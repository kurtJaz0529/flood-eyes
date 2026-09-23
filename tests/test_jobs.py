"""
慧眼识灾 · 任务 C 队列持久化回归测试
=====================================

运行：
    python tests/test_jobs.py
    python -m pytest tests/test_jobs.py -q

覆盖任务卡 C 的验收点：

    · enqueue/list/get 顺序稳定，payload 原样往返，job_id 唯一；
    · 串行执行：中间任务失败不中断，后续任务照常成功；
    · 持久化 + 显式恢复：构造函数不自动恢复，recover_interrupted() 只动 running，
      不碰 succeeded/pending；
    · 排队中取消的任务绝不执行；running 取消置标志，安全边界不算 success；
    · retry 只对 failed/cancelled/interrupted 生效，同一 job_id、attempt+1；
    · 并发领取只有一个 worker 成功（同一数据库最多一个 running）；
    · progress 日志有界、stage 持久化；结果非 JSON-safe 记为 failed 并继续；
    · 有 zip/结果文件但 runner 抛异常时仍是 failed（不凭产物推断成功）。

用例只使用标准库与临时目录，不联网、不装依赖。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

from src import jobs  # noqa: E402


class JobStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="huiyan_jobs_")
        self.db = os.path.join(self.tmp, "jobs.db")
        self.store = jobs.JobStore(self.db)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_worker_lock_is_exclusive_across_store_instances(self):
        other = jobs.JobStore(self.db)
        with self.store.worker_lock() as acquired:
            self.assertTrue(acquired)
            with other.worker_lock() as second:
                self.assertFalse(second)
        with other.worker_lock() as acquired_again:
            self.assertTrue(acquired_again)

    # -- 辅助 -------------------------------------------------------------

    def _run(self, runner):
        return list(jobs.QueueRunner(self.store, runner).run_pending())

    def _events(self, events):
        return [e["event"] for e in events]

    def _ok_runner(self, payload, job_id, progress, is_cancelled):
        progress("running", stage="work")
        return {"ok": payload.get("n")}


class TestEnqueueAndRead(JobStoreTestCase):
    def test_enqueue_unique_and_list_order_stable(self):
        ids = [self.store.enqueue({"n": i}) for i in range(3)]
        self.assertEqual(len(set(ids)), 3)
        rows = self.store.list_jobs()
        self.assertEqual([r["job_id"] for r in rows], ids)
        self.assertEqual([r["payload"] for r in rows], [{"n": 0}, {"n": 1}, {"n": 2}])
        for row in rows:
            self.assertEqual(row["status"], jobs.PENDING)
            self.assertEqual(row["attempt"], 1)
            self.assertFalse(row["cancel_requested"])
            self.assertIsNone(row["result"])
            self.assertIsNone(row["started_at"])
            self.assertIsNone(row["finished_at"])

    def test_get_returns_summary_keys_and_logs(self):
        job_id = self.store.enqueue({"n": 1, "nested": {"a": [1, 2]}})
        snap = self.store.get(job_id)
        for key in jobs.SUMMARY_KEYS:
            self.assertIn(key, snap)
        self.assertEqual(snap["payload"], {"n": 1, "nested": {"a": [1, 2]}})
        self.assertEqual(snap["logs"], [])
        self.assertIsNone(self.store.get("does-not-exist"))

    def test_payload_must_be_json_safe_dict(self):
        with self.assertRaises(TypeError):
            self.store.enqueue(["not", "a", "dict"])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.store.enqueue({"bad": object()})

    def test_summary_counts(self):
        self.store.enqueue({"n": 1})
        self.store.enqueue({"n": 2})
        summary = self.store.summary()
        self.assertEqual(summary["pending"], 2)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["succeeded"], 0)


class TestSerialExecution(JobStoreTestCase):
    def test_middle_failure_third_succeeds(self):
        ids = [self.store.enqueue({"n": n}) for n in (1, 2, 3)]
        seen = []

        def runner(payload, job_id, progress, is_cancelled):
            seen.append(payload["n"])
            if payload["n"] == 2:
                raise RuntimeError("boom")
            return {"ok": payload["n"]}

        events = self._run(runner)
        self.assertEqual(seen, [1, 2, 3], "失败任务之后必须继续执行后续任务")
        self.assertEqual(
            self._events(events),
            [jobs.EVENT_STARTED, jobs.EVENT_SUCCEEDED,
             jobs.EVENT_STARTED, jobs.EVENT_FAILED,
             jobs.EVENT_STARTED, jobs.EVENT_SUCCEEDED],
        )
        self.assertEqual(
            [r["status"] for r in self.store.list_jobs()],
            [jobs.SUCCEEDED, jobs.FAILED, jobs.SUCCEEDED],
        )
        self.assertEqual(self.store.get(ids[0])["result"], {"ok": 1})
        self.assertIsNone(self.store.get(ids[1])["result"])
        self.assertEqual(self.store.get(ids[1])["error"], "RuntimeError: boom")
        self.assertEqual(self.store.get(ids[2])["result"], {"ok": 3})

    def test_empty_store_yields_nothing(self):
        self.assertEqual(self._run(self._ok_runner), [])

    def test_progress_logs_bounded_and_stage_persisted(self):
        store = jobs.JobStore(self.db, log_limit=5)
        job_id = store.enqueue({"n": 1})

        def runner(payload, jid, progress, is_cancelled):
            for i in range(20):
                progress("m%d" % i, stage="s%d" % (i % 2))
            return {"ok": True}

        list(jobs.QueueRunner(store, runner).run_pending())
        snap = store.get(job_id)
        self.assertEqual(len(snap["logs"]), 5)
        self.assertEqual(snap["logs"][0]["message"], "m15")
        self.assertEqual(snap["logs"][-1]["message"], "m19")
        self.assertEqual(snap["stage"], "s1")
        self.assertEqual(snap["status"], jobs.SUCCEEDED)

    def test_progress_unknown_job_rejected(self):
        with self.assertRaises(KeyError):
            self.store.progress("missing", "hello")

    def test_non_json_safe_result_marks_failed_and_continues(self):
        self.store.enqueue({"n": 1})
        self.store.enqueue({"n": 2})

        def runner(payload, job_id, progress, is_cancelled):
            if payload["n"] == 1:
                return object()
            return {"ok": True}

        self._run(runner)
        first, second = self.store.list_jobs()
        self.assertEqual(first["status"], jobs.FAILED)
        self.assertIn("JSON-safe", first["error"])
        self.assertEqual(second["status"], jobs.SUCCEEDED)

    def test_zip_presence_is_not_success(self):
        """runner 抛异常时，即使结果目录里出现了 zip，也必须是 failed。"""
        zip_path = os.path.join(self.tmp, "job_result.zip")
        job_id = self.store.enqueue({"n": 1})

        def runner(payload, jid, progress, is_cancelled):
            with open(zip_path, "w", encoding="utf-8") as fh:
                fh.write("fake zip")
            raise RuntimeError("产物存在但流程失败")

        self._run(runner)
        self.assertTrue(os.path.isfile(zip_path))
        snap = self.store.get(job_id)
        self.assertEqual(snap["status"], jobs.FAILED)
        self.assertIsNone(snap["result"])


class TestCancellation(JobStoreTestCase):
    def test_cancelled_pending_never_runs(self):
        first = self.store.enqueue({"n": 1})
        second = self.store.enqueue({"n": 2})
        cancelled = self.store.request_cancel(second)
        self.assertEqual(cancelled["status"], jobs.CANCELLED)
        self.assertIsNotNone(cancelled["finished_at"])

        seen = []

        def runner(payload, job_id, progress, is_cancelled):
            seen.append(payload["n"])
            return {"ok": payload["n"]}

        self._run(runner)
        self.assertEqual(seen, [1], "排队中被取消的任务绝不能执行")
        self.assertEqual(self.store.get(first)["status"], jobs.SUCCEEDED)
        self.assertEqual(self.store.get(second)["status"], jobs.CANCELLED)

    def test_cancel_running_sets_flag_and_boundary_is_not_success(self):
        job_id = self.store.enqueue({"n": 1})
        saw_flag = {"value": False}

        def runner(payload, jid, progress, is_cancelled):
            progress("halfway", stage="mid")
            self.store.request_cancel(jid)
            saw_flag["value"] = is_cancelled()
            return {"should": "not be success"}

        events = self._run(runner)
        self.assertTrue(saw_flag["value"], "is_cancelled() 必须读到 cancel_requested")
        snap = self.store.get(job_id)
        self.assertEqual(snap["status"], jobs.CANCELLED)
        self.assertIsNone(snap["result"], "取消边界返回的结果不能落成 success")
        self.assertTrue(snap["cancel_requested"])
        self.assertEqual(snap["stage"], "mid")
        self.assertEqual(self._events(events)[-1], jobs.EVENT_CANCELLED)

    def test_job_cancelled_exception_is_cancelled(self):
        job_id = self.store.enqueue({"n": 1})

        def runner(payload, jid, progress, is_cancelled):
            raise jobs.JobCancelled("user cancelled")

        events = self._run(runner)
        self.assertEqual(self.store.get(job_id)["status"], jobs.CANCELLED)
        self.assertEqual(self._events(events)[-1], jobs.EVENT_CANCELLED)

    def test_cancel_terminal_job_is_noop(self):
        job_id = self.store.enqueue({"n": 1})
        self._run(self._ok_runner)
        snap = self.store.request_cancel(job_id)
        self.assertEqual(snap["status"], jobs.SUCCEEDED)
        self.assertFalse(snap["cancel_requested"])
        self.assertIsNone(self.store.request_cancel("missing"))


class TestPersistenceAndRecovery(JobStoreTestCase):
    def test_constructor_does_not_recover(self):
        job_id = self.store.enqueue({"n": 1})
        self.store.claim_next()
        reopened = jobs.JobStore(self.db)
        self.assertEqual(reopened.get(job_id)["status"], jobs.RUNNING)

    def test_recover_only_running_leaves_succeeded_and_pending(self):
        first = self.store.enqueue({"n": 1})
        second = self.store.enqueue({"n": 2})
        third = self.store.enqueue({"n": 3})
        self.store.claim_next()
        self.store.finish(first, jobs.SUCCEEDED, result={"ok": 1})
        self.store.claim_next()  # second -> running
        self.assertEqual(self.store.get(second)["status"], jobs.RUNNING)

        reopened = jobs.JobStore(self.db)
        self.assertEqual(reopened.recover_interrupted(), 1)
        self.assertEqual(reopened.get(first)["status"], jobs.SUCCEEDED)
        self.assertEqual(reopened.get(first)["result"], {"ok": 1})
        self.assertEqual(reopened.get(second)["status"], jobs.INTERRUPTED)
        self.assertEqual(reopened.get(third)["status"], jobs.PENDING)

    def test_recover_again_is_idempotent(self):
        self.store.enqueue({"n": 1})
        self.store.claim_next()
        self.assertEqual(self.store.recover_interrupted(), 1)
        self.assertEqual(self.store.recover_interrupted(), 0)

    def test_interrupted_job_retry_runs_again(self):
        job_id = self.store.enqueue({"n": 1})
        self.store.claim_next()
        self.store.recover_interrupted()
        snap = self.store.retry(job_id)
        self.assertEqual(snap["status"], jobs.PENDING)
        self.assertEqual(snap["attempt"], 2)
        self._run(self._ok_runner)
        self.assertEqual(self.store.get(job_id)["status"], jobs.SUCCEEDED)


class TestRetry(JobStoreTestCase):
    def test_failed_retry_resets_same_id_and_bumps_attempt(self):
        job_id = self.store.enqueue({"n": 1})

        def boom(payload, jid, progress, is_cancelled):
            progress("trying", stage="s1")
            raise ValueError("bad input")

        self._run(boom)
        snap = self.store.get(job_id)
        self.assertEqual(snap["status"], jobs.FAILED)
        retried = self.store.retry(job_id)
        self.assertEqual(retried["job_id"], job_id)
        self.assertEqual(retried["status"], jobs.PENDING)
        self.assertEqual(retried["attempt"], 2)
        self.assertIsNone(retried["error"])
        self.assertIsNone(retried["stage"])
        self.assertIsNone(retried["started_at"])
        self.assertFalse(retried["cancel_requested"])

        self._run(self._ok_runner)
        self.assertEqual(self.store.get(job_id)["status"], jobs.SUCCEEDED)

    def test_retry_only_for_retryable_states(self):
        pending = self.store.enqueue({"n": 1})
        with self.assertRaises(ValueError):
            self.store.retry(pending)

        running = self.store.enqueue({"n": 2})
        self.store.claim_next()
        with self.assertRaises(ValueError):
            self.store.retry(running)
        self.store.finish(running, jobs.FAILED, error="x")

        succeeded = self.store.enqueue({"n": 3})
        self.store.claim_next()
        self.store.finish(succeeded, jobs.SUCCEEDED, result={"ok": True})
        with self.assertRaises(ValueError):
            self.store.retry(succeeded)

    def test_cancelled_retry_allowed(self):
        job_id = self.store.enqueue({"n": 1})
        self.store.request_cancel(job_id)
        snap = self.store.retry(job_id)
        self.assertEqual(snap["status"], jobs.PENDING)
        self.assertEqual(snap["attempt"], 2)
        self.assertFalse(snap["cancel_requested"])

    def test_retry_missing_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.store.retry("missing")


class TestConcurrentClaim(JobStoreTestCase):
    def test_concurrent_claims_yield_single_winner(self):
        job_id = self.store.enqueue({"n": 1})
        stores = [jobs.JobStore(self.db) for _ in range(4)]
        barrier = threading.Barrier(len(stores))
        results = []
        errors = []
        lock = threading.Lock()

        def worker(store):
            try:
                barrier.wait(timeout=10)
                claimed = store.claim_next()
                with lock:
                    results.append(claimed)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(s,)) for s in stores]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [])
        winners = [r for r in results if r is not None]
        self.assertEqual(len(winners), 1, "同一数据库同时只能有一个 running 任务")
        self.assertEqual(winners[0]["job_id"], job_id)
        self.assertEqual(len(stores), len(results))

    def test_running_job_blocks_new_claim(self):
        first = self.store.enqueue({"n": 1})
        self.store.enqueue({"n": 2})
        self.assertEqual(self.store.claim_next()["job_id"], first)
        self.assertIsNone(self.store.claim_next())
        self.store.finish(first, jobs.SUCCEEDED, result={"ok": 1})
        self.assertEqual(self.store.claim_next()["payload"], {"n": 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
