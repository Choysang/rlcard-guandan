"""Export human expert GUI decisions to question/sentence JSONL.

Example:
    python examples/export_human_expert_dataset.py \
        --input logs/gui/guandan_gui.jsonl \
        --output dataset/human_expert.jsonl
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.backend.expert_data import export_expert_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='logs/gui/guandan_gui.jsonl')
    parser.add_argument('--output', default='dataset/human_expert.jsonl')
    args = parser.parse_args()

    summary = export_expert_samples(args.input, args.output)
    print(
        f"read={summary['read']} written={summary['written']} "
        f"skipped={summary['skipped']} output={args.output}"
    )


if __name__ == '__main__':
    main()
