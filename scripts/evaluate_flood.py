"""Compare a water_mask.tif with independent, co-registered 0/1/255 labels."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import evaluate_water
from src.task_contracts import TERRAIN_PROFILES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--terrain-profile", choices=TERRAIN_PROFILES, default="unspecified")
    parser.add_argument("--out", help="新建 JSON 结果文件；存在则拒绝覆盖")
    args = parser.parse_args(argv)
    result = evaluate_water(args.prediction, args.reference, args.terrain_profile)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.out:
        with open(args.out, "x", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
