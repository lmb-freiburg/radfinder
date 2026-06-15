import argparse
import re
from pathlib import Path

from rate.reportstructuring_validation import load_question_pairs_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify question consistency constraints directly on modality config YAML files."
    )
    parser.add_argument(
        "--config-root",
        type=str,
        default="config",
        help="Root path containing modalities_en/ and modalities_de/",
    )
    parser.add_argument(
        "--show-order-diff",
        action="store_true",
        help="Also print positional order differences for shared questions.",
    )
    return parser.parse_args()


def _pairs(config_root: Path, lang: str, modality: str) -> list[tuple[str, str]]:
    path = config_root / f"modalities_{lang}" / f"{modality}_ct.yaml"
    return load_question_pairs_from_config(path)


def _duplicate_category_keys(config_path: Path) -> dict[str, int]:
    text = config_path.read_text(encoding="utf-8")
    in_categories = False
    category_counts: dict[str, int] = {}
    for line in text.splitlines():
        if line.strip() == "categories:" or line.strip() == '"categories":':
            in_categories = True
            continue
        if not in_categories:
            continue
        # End once indentation returns to top-level non-empty line.
        if line and not line.startswith(" "):
            break
        match = re.match(r'^ {2}(?! )"?([^":]+)"?:\s*$', line)
        if match:
            category = match.group(1)
            category_counts[category] = category_counts.get(category, 0) + 1
    return {k: v for k, v in category_counts.items() if v > 1}


def main() -> int:
    args = parse_args()
    config_root = Path(args.config_root)
    ok = True

    print("=== Duplicate category keys in raw config files ===")
    for lang in ["en", "de"]:
        for modality in ["abdomen", "chest", "abdomen_chest"]:
            path = config_root / f"modalities_{lang}" / f"{modality}_ct.yaml"
            duplicates = _duplicate_category_keys(path)
            if duplicates:
                ok = False
                print(f"{path.as_posix()}: DUPLICATES {duplicates}")
            else:
                print(f"{path.as_posix()}: none")

    print("=== Cross-language count/order checks on configs ===")
    for modality in ["abdomen", "chest", "abdomen_chest"]:
        en = _pairs(config_root, "en", modality)
        de = _pairs(config_root, "de", modality)
        same_count = len(en) == len(de)
        status = "OK" if same_count else "FAIL"
        print(f"{modality}: {status} en_n={len(en)} de_n={len(de)}")
        if not same_count:
            ok = False

    print("\n=== Union checks inside each language ===")
    for lang in ["en", "de"]:
        abdomen = _pairs(config_root, lang, "abdomen")
        chest = _pairs(config_root, lang, "chest")
        abdomen_chest = _pairs(config_root, lang, "abdomen_chest")

        union = list(dict.fromkeys(abdomen + chest))
        union_set = set(union)
        ac_set = set(abdomen_chest)
        missing = union_set - ac_set
        extra = ac_set - union_set
        order_equal = union == abdomen_chest

        status = "OK" if not missing and not extra else "FAIL"
        print(
            f"{lang}: {status} union_n={len(union_set)} abdomen_chest_n={len(ac_set)} "
            f"missing={len(missing)} extra={len(extra)} order_equal={order_equal}"
        )
        if missing:
            print("  missing questions:")
            for category, question in sorted(missing):
                print(f"  missing: [{category}] {question}")
        if extra:
            print("  extra questions:")
            for category, question in sorted(extra):
                print(f"  extra: [{category}] {question}")
        if args.show_order_diff and not order_equal:
            ac_pos = {pair: i for i, pair in enumerate(abdomen_chest)}
            union_pos = {pair: i for i, pair in enumerate(union)}
            mismatched_shared = [
                (pair, union_pos[pair], ac_pos[pair])
                for pair in union
                if pair in ac_pos and union_pos[pair] != ac_pos[pair]
            ]
            print(f"  order mismatches among shared questions: {len(mismatched_shared)}")
            for (category, question), u_i, ac_i in mismatched_shared:
                print(
                    f"  order_diff: [{category}] {question} "
                    f"union_idx={u_i} abdomen_chest_idx={ac_i}"
                )
        if missing or extra:
            ok = False

    print(f"\nOVERALL: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
