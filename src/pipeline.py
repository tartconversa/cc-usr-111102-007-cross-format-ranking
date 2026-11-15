"""离线研究流水线入口。

用法：
    python -m src.pipeline --fixtures fixtures --output .workspace-output
    python -m src.pipeline --fixtures fixtures --check

默认只打印摘要；--output 会把 report.json 与 summary.md 写到指定目录（建议 .workspace-output）。
--check 在摘要之后执行内建自检，任一失败返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluation import run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨业态推荐离线评估与约束重排")
    parser.add_argument("--fixtures", default="fixtures", help="资料目录")
    parser.add_argument("--output", default=None, help="输出目录（建议 .workspace-output）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 policies.json 的种子")
    parser.add_argument("--max-exact-lists", type=int, default=50000,
                        help="精确枚举的合法列表上限，超出改用拒绝采样")
    parser.add_argument("--check", action="store_true", help="运行内建自检")
    args = parser.parse_args(argv)

    result = run_pipeline(
        Path(args.fixtures),
        Path(args.output) if args.output else None,
        seed_override=args.seed,
        max_exact_lists=args.max_exact_lists,
    )
    print(result.text_summary)
    if args.check:
        from .selfcheck import run_self_check

        report = run_self_check(result.payload)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
