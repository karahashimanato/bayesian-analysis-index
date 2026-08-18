#!/usr/bin/env python3
"""演習をその場で採点するオンデマンドCLI。git操作は一切不要。

使い方:
    python exercises/grade.py                # 演習一覧を表示
    python exercises/grade.py rhat            # 名前の一部が一致する演習を採点
    python exercises/grade.py all             # 全演習を採点
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXERCISES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXERCISES_DIR.parent


def discover_exercises() -> list[Path]:
    return sorted(p.parent for p in EXERCISES_DIR.glob("*/*/problem.md"))


def format_name(path: Path) -> str:
    return path.relative_to(EXERCISES_DIR).as_posix()


def print_list(exercises: list[Path]) -> None:
    print("演習一覧(名前の一部を指定して採点できます):\n")
    for ex in exercises:
        print(f"  {format_name(ex)}")
    print('\n例: python exercises/grade.py rhat_classification')


def run_pytest(target: Path) -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-v", "--tb=short"],
        cwd=REPO_ROOT,
    )
    return result.returncode


def main() -> int:
    exercises = discover_exercises()

    if len(sys.argv) < 2:
        print_list(exercises)
        return 0

    query = sys.argv[1]

    if query == "all":
        return run_pytest(EXERCISES_DIR)

    matches = [ex for ex in exercises if query in format_name(ex)]

    if not matches:
        print(f"'{query}' に一致する演習が見つかりません。\n")
        print_list(exercises)
        return 1

    if len(matches) > 1:
        print(f"'{query}' に複数の演習が一致しました。もう少し絞り込んでください:\n")
        for ex in matches:
            print(f"  {format_name(ex)}")
        return 1

    target = matches[0]
    print(f"=== 採点対象: {format_name(target)} ===\n", flush=True)
    returncode = run_pytest(target)

    print()
    if returncode == 0:
        print("結果: 全テストpass ✅")
    else:
        print("結果: fail ❌ (上のエラーメッセージを参照)")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
