"""
慧眼识灾 · 任务 C：串行任务队列的持久化
=====================================

只依赖标准库（``sqlite3``）的串行队列，供界面/脚本在没有后台线程时驱动：

    store = JobStore("outputs/jobs.db")
    store.recover_interrupted()          # 启动时显式做一次，构造函数不做
    job_id = store.enqueue({"request": {...}})
    for event in QueueRunner(store, my_runner).run_pending():
        ...                              # 每个事件带任务快照，界面据此刷新

设计要点：``claim_next()`` 在 ``BEGIN IMMEDIATE`` 事务里先查 running，有就返回
None（同一数据库同时最多一个 running）；``pending`` 取消直接转 ``cancelled``
（绝不执行），``running`` 取消只置 ``cancel_requested``，由 runner 在安全边界退出；
``retry()`` 只接受 ``failed/cancelled/interrupted``，同一 job_id 重置为 ``pending``
且 ``attempt`` 加一；``recover_interrupted()`` 只把 ``running`` 改成 ``interrupted``，
终止态一律不动；状态只由 runner 是否正常返回（且未被取消）决定，**绝不从
zip/落盘产物推断成功**。

runner 契约::

    def runner(payload, job_id, progress, is_cancelled) -> JSON-safe result

``payload`` 是 enqueue 的 JSON-safe dict；``progress(msg, stage=None)`` 写有界日志并
更新 stage；``is_cancelled()`` 读取取消标志（安全边界轮询）。
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional

__all__ = [
    "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED",
    "STATUSES", "TERMINAL_STATUSES", "RETRYABLE_STATUSES", "SUMMARY_KEYS",
    "EVENT_STARTED", "EVENT_SUCCEEDED", "EVENT_FAILED", "EVENT_CANCELLED",
    "MAX_LOG_MESSAGE", "JobCancelled", "JobStore", "QueueRunner",
]

PENDING, RUNNING, SUCCEEDED = "pending", "running", "succeeded"
FAILED, CANCELLED, INTERRUPTED = "failed", "cancelled", "interrupted"
STATUSES = (PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED, INTERRUPTED)
TERMINAL_STATUSES = frozenset({SUCCEEDED, FAILED, CANCELLED, INTERRUPTED})
RETRYABLE_STATUSES = frozenset({FAILED, CANCELLED, INTERRUPTED})
EVENT_STARTED, EVENT_SUCCEEDED = "started", "succeeded"
EVENT_FAILED, EVENT_CANCELLED = "failed", "cancelled"
#: get()/list_jobs() 快照里稳定的顶层键；get() 额外带有界 ``logs``。
SUMMARY_KEYS = ("job_id", "status", "payload", "result", "error", "stage", "attempt",
                "cancel_requested", "created_at", "updated_at", "started_at", "finished_at")
MAX_LOG_MESSAGE = 2000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, status TEXT NOT NULL,
    result_json TEXT, error TEXT, stage TEXT, attempt INTEGER NOT NULL DEFAULT 1,
    cancel_requested INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, started_at TEXT, finished_at TEXT);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS job_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, ts TEXT NOT NULL,
    stage TEXT, message TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_job_logs_job ON job_logs(job_id, id);
"""


class JobCancelled(Exception):
    """runner 在安全边界确认取消时主动抛出；与 ``cancel_requested`` 标志等价。"""


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _dumps(value: Any) -> str:
    """严格 JSON 序列化：不可序列化的对象直接报错，不做 default=str 兜底。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(text: Optional[str], default: Any = None) -> Any:
    if text is None:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def _snapshot(row: sqlite3.Row, logs: Optional[List[sqlite3.Row]] = None) -> Dict[str, Any]:
    job: Dict[str, Any] = {
        "job_id": row["job_id"], "status": row["status"],
        "payload": _loads(row["payload_json"], {}), "result": _loads(row["result_json"], None),
        "error": row["error"], "stage": row["stage"], "attempt": int(row["attempt"]),
        "cancel_requested": bool(row["cancel_requested"]), "created_at": row["created_at"],
        "updated_at": row["updated_at"], "started_at": row["started_at"], "finished_at": row["finished_at"]}
    if logs is not None:
        job["logs"] = [{"ts": r["ts"], "stage": r["stage"], "message": r["message"]} for r in logs]
    return job


class JobStore:
    """SQLite 串行任务队列；每次操作独立连接，跨线程/跨进程都安全。"""

    def __init__(self, db_path: str, *, log_limit: int = 200, timeout: float = 15.0) -> None:
        self.db_path, self.log_limit, self.timeout = (
            os.fspath(db_path), max(1, int(log_limit)), float(timeout))
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.timeout, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = %d" % int(self.timeout * 1000))
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode = WAL"); conn.executescript(_SCHEMA)
        finally:
            conn.close()

    @contextmanager
    def worker_lock(self) -> Iterator[bool]:
        """跨进程独占队列执行权；非阻塞获取，供执行与恢复共同使用。"""
        lock_path = self.db_path + ".worker.lock"
        with open(lock_path, "a+b") as fh:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"0")
                fh.flush()
            fh.seek(0)
            acquired = False
            try:
                if os.name == "nt":
                    import msvcrt
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                    except OSError:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except BlockingIOError:
                        pass
                yield acquired
            finally:
                if acquired:
                    fh.seek(0)
                    if os.name == "nt":
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """BEGIN IMMEDIATE：写事务从读取队首起就占锁，避免重复领取。"""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    def enqueue(self, payload: Dict[str, Any]) -> str:
        """入队一个 JSON-safe dict，返回全局唯一 job_id。"""
        if not isinstance(payload, dict):
            raise TypeError("payload 必须是 JSON-safe 的 dict")
        payload_json = _dumps(payload)  # 非 JSON-safe 在这里就报错
        job_id, now = uuid.uuid4().hex, _now()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, payload_json, status, attempt, cancel_requested,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (job_id, payload_json, PENDING, 1, 0, now, now),
            )
        return job_id

    def request_cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        """pending 直接转 cancelled；running 只置 cancel_requested；终止态不动。"""
        now = _now()
        with self._write() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if row["status"] == PENDING:
                conn.execute("UPDATE jobs SET status=?, finished_at=?, updated_at=?"
                             " WHERE job_id=? AND status=?",
                             (CANCELLED, now, now, job_id, PENDING))
            elif row["status"] == RUNNING:
                conn.execute("UPDATE jobs SET cancel_requested=1, updated_at=?"
                             " WHERE job_id=? AND status=?", (now, job_id, RUNNING))
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return _snapshot(row)

    def retry(self, job_id: str) -> Dict[str, Any]:
        """failed/cancelled/interrupted → pending，attempt+1，保持同一 job_id。"""
        now = _now()
        with self._write() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] not in RETRYABLE_STATUSES:
                raise ValueError("只有 failed/cancelled/interrupted 可重试，当前状态：%s"
                                 % row["status"])
            conn.execute("UPDATE jobs SET status=?, attempt=attempt+1, cancel_requested=0,"
                         " result_json=NULL, error=NULL, stage=NULL, started_at=NULL,"
                         " finished_at=NULL, updated_at=? WHERE job_id=?",
                         (PENDING, now, job_id))
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return _snapshot(row)

    def recover_interrupted(self) -> int:
        """启动维护：只把 running 标成 interrupted，其它状态（含 succeeded）不动。"""
        now = _now()
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, updated_at=?, error=COALESCE(error, ?)"
                " WHERE status=?",
                (INTERRUPTED, now, now, "进程中断，任务未完成（recover_interrupted）", RUNNING))
            return int(cur.rowcount)

    def claim_next(self) -> Optional[Dict[str, Any]]:
        """领取最老的 pending 任务并置为 running；已有 running 时返回 None。"""
        now = _now()
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE status=? LIMIT 1", (RUNNING,)).fetchone():
                return None
            row = conn.execute("SELECT * FROM jobs WHERE status=? AND cancel_requested=0"
                               " ORDER BY rowid ASC LIMIT 1", (PENDING,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE jobs SET status=?, started_at=?, updated_at=?"
                         " WHERE job_id=? AND status=?",
                         (RUNNING, now, now, row["job_id"], PENDING))
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()
            return _snapshot(row)

    def finish(self, job_id: str, status: str, result: Any = None,
               error: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """收尾一个 running 任务；成功时才落 result_json。"""
        if status not in TERMINAL_STATUSES:
            raise ValueError("finish 只接受终止状态：%s" % status)
        result_json = _dumps(result) if (status == SUCCEEDED and result is not None) else None
        now = _now()
        with self._write() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status=?, result_json=?, error=?, finished_at=?, updated_at=?"
                " WHERE job_id=? AND status=?",
                (status, result_json, error, now, now, job_id, RUNNING))
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            if cur.rowcount == 0 and row["status"] != status:
                return _snapshot(row)  # 已被并发恢复/取消，返回真实状态
            return _snapshot(row)

    def progress(self, job_id: str, message: Any, stage: Optional[str] = None) -> None:
        """追加一条有界日志，并可选更新 stage。"""
        text, stage_text, now = str(message)[:MAX_LOG_MESSAGE], None, _now()
        if stage is not None:
            stage_text = str(stage)
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() is None:
                raise KeyError(job_id)
            conn.execute("INSERT INTO job_logs (job_id, ts, stage, message) VALUES (?,?,?,?)",
                         (job_id, now, stage_text, text))
            conn.execute("DELETE FROM job_logs WHERE job_id=? AND id NOT IN"
                         " (SELECT id FROM job_logs WHERE job_id=? ORDER BY id DESC LIMIT ?)",
                         (job_id, job_id, self.log_limit))
            if stage_text is not None:
                conn.execute("UPDATE jobs SET stage=?, updated_at=? WHERE job_id=?",
                             (stage_text, now, job_id))
            else:
                conn.execute("UPDATE jobs SET updated_at=? WHERE job_id=?", (now, job_id))

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            logs = conn.execute("SELECT ts, stage, message FROM job_logs WHERE job_id=?"
                                " ORDER BY id ASC", (job_id,)).fetchall()
            return _snapshot(row, logs)
        finally:
            conn.close()

    def list_jobs(self) -> List[Dict[str, Any]]:
        """按入队先后（最老在前）返回全部任务快照，顺序稳定；不含 logs。"""
        conn = self._connect()
        try:
            return [_snapshot(r) for r in
                    conn.execute("SELECT * FROM jobs ORDER BY rowid ASC").fetchall()]
        finally:
            conn.close()

    def is_cancel_requested(self, job_id: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute("SELECT cancel_requested FROM jobs WHERE job_id=?",
                               (job_id,)).fetchone()
            return bool(row["cancel_requested"]) if row is not None else False
        finally:
            conn.close()

    def summary(self) -> Dict[str, int]:
        """各状态计数 + total，便于界面顶栏展示。"""
        conn = self._connect()
        try:
            counts = {status: 0 for status in STATUSES}
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
                counts[row["status"]] = int(row["n"])
            counts["total"] = sum(counts[status] for status in STATUSES)
            return counts
        finally:
            conn.close()


class QueueRunner:
    """驱动 ``JobStore`` 的串行执行器（生成器，无后台线程）。

    ``run_pending()`` 每完成一次状态迁移就 yield
    ``{"event": "started"|"succeeded"|"failed"|"cancelled", "job": <快照>}``。
    单任务抛异常只记 failed 并继续下一个；runner 正常返回但 ``cancel_requested``
    已置位（或抛 ``JobCancelled``）时记 cancelled，其结果不会落成 success。
    """

    def __init__(self, store: JobStore, runner: Callable[..., Any]) -> None:
        if not isinstance(store, JobStore) or not callable(runner):
            raise TypeError("store 必须是 JobStore，runner 必须可调用")
        self.store, self.runner = store, runner

    def _progress(self, job_id: str) -> Callable[..., None]:
        return lambda message, stage=None: self.store.progress(job_id, message, stage)

    def _is_cancelled(self, job_id: str) -> Callable[[], bool]:
        return lambda: self.store.is_cancel_requested(job_id)

    def run_pending(self) -> Iterator[Dict[str, Any]]:
        with self.store.worker_lock() as acquired:
            if not acquired:
                return
            yield from self._run_locked()

    def _run_locked(self) -> Iterator[Dict[str, Any]]:
        while True:
            job = self.store.claim_next()
            if job is None:
                return
            job_id = job["job_id"]
            yield {"event": EVENT_STARTED, "job": self.store.get(job_id)}
            result: Any = None
            error: Optional[str] = None
            cancelled = False
            try:
                result = self.runner(job["payload"], job_id,
                                     self._progress(job_id), self._is_cancelled(job_id))
            except JobCancelled:
                cancelled = True
            except Exception as exc:  # noqa: BLE001 - 单任务失败不拖垮整条队列
                error = "%s: %s" % (type(exc).__name__, exc)
            if not cancelled and error is None and not self.store.is_cancel_requested(job_id):
                try:
                    _dumps(result)
                except (TypeError, ValueError) as exc:
                    error = "结果不是 JSON-safe：%s: %s" % (type(exc).__name__, exc)
            if cancelled or self.store.is_cancel_requested(job_id):
                snap = self.store.finish(job_id, CANCELLED) or self.store.get(job_id)
                yield {"event": EVENT_CANCELLED, "job": snap}
            elif error is not None:
                snap = self.store.finish(job_id, FAILED, error=error) or self.store.get(job_id)
                yield {"event": EVENT_FAILED, "job": snap}
            else:
                snap = self.store.finish(job_id, SUCCEEDED, result=result) or self.store.get(job_id)
                yield {"event": EVENT_SUCCEEDED, "job": snap}
