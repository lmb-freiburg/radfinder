"""
Verify question map consistency across languages and modalities.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from attrs import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs


@dataclass
class MapData:
    rows: list[dict[str, str]]
    by_pair: dict[tuple[str, str], str]
    by_qid: dict[str, tuple[str, str]]


def load_map(path: Path) -> MapData:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_pair: dict[tuple[str, str], str] = {}
    by_qid: dict[str, tuple[str, str]] = {}
    for row in rows:
        qid = row["qid"]
        pair = (row["category"], row["question"])
        by_pair[pair] = qid
        by_qid[qid] = pair
    return MapData(rows=rows, by_pair=by_pair, by_qid=by_qid)


def modality_map_path(root: Path, lang: str, modality: str) -> Path:
    return root / f"modalities_{lang}" / "question_maps" / f"question_map_{modality}.csv"


def verify_cross_language(root: Path) -> bool:
    print("\n=== Cross-language structural check (EN vs DE) ===")
    ok = True
    for modality in ["abdomen", "chest", "abdomen_chest"]:
        en_map = load_map(modality_map_path(root, "en", modality))
        de_map = load_map(modality_map_path(root, "de", modality))
        en_qids = [row["qid"] for row in en_map.rows]
        de_qids = [row["qid"] for row in de_map.rows]
        same_qid_sequence = en_qids == de_qids
        same_count = len(en_map.rows) == len(de_map.rows)
        status = "OK" if same_qid_sequence and same_count else "FAIL"
        print(
            f"{modality}: {status} "
            f"en_n={len(en_map.rows)} de_n={len(de_map.rows)} "
            f"same_qid_sequence={same_qid_sequence}"
        )
        if not (same_qid_sequence and same_count):
            ok = False
    return ok


def verify_union_per_language(root: Path, lang: str) -> bool:
    print(f"\n=== Union check within {lang.upper()} ===")
    ok = True
    abdomen = load_map(modality_map_path(root, lang, "abdomen"))
    chest = load_map(modality_map_path(root, lang, "chest"))
    abdomen_chest = load_map(modality_map_path(root, lang, "abdomen_chest"))

    union_by_pair = dict(abdomen.by_pair)
    union_by_pair.update(chest.by_pair)

    union_pairs = set(union_by_pair.keys())
    ac_pairs = set(abdomen_chest.by_pair.keys())

    missing_in_ac = sorted(union_pairs - ac_pairs)
    extra_in_ac = sorted(ac_pairs - union_pairs)

    qid_mismatch: list[tuple[str, str, str, str]] = []
    for pair in sorted(union_pairs & ac_pairs):
        union_qid = union_by_pair[pair]
        ac_qid = abdomen_chest.by_pair[pair]
        if union_qid != ac_qid:
            qid_mismatch.append((pair[0], pair[1], union_qid, ac_qid))

    print(
        f"union_size={len(union_pairs)} abdomen_chest_size={len(ac_pairs)} "
        f"missing_in_abdomen_chest={len(missing_in_ac)} extra_in_abdomen_chest={len(extra_in_ac)} "
        f"qid_mismatch={len(qid_mismatch)}"
    )

    if missing_in_ac:
        ok = False
        print("missing examples (up to 10):")
        for category, question in missing_in_ac[:10]:
            print(f"  - [{category}] {question}")
    if extra_in_ac:
        ok = False
        print("extra examples (up to 10):")
        for category, question in extra_in_ac[:10]:
            print(f"  - [{category}] {question}")
    if qid_mismatch:
        ok = False
        print("qid mismatches (up to 10):")
        for category, question, union_qid, ac_qid in qid_mismatch[:10]:
            print(f"  - [{category}] {question}: union={union_qid} abdomen_chest={ac_qid}")

    if ok:
        print("OK: abdomen_chest is an exact union with consistent qids.")

    return ok


@define
class Args(VerboseQuietArgs):
    pass


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    root = Path(RADFINDER_REPO_DIR) / "configs/rate"
    if not root.exists():
        raise FileNotFoundError(f"Maps root does not exist: {root.as_posix()}")

    cross_ok = verify_cross_language(root)
    en_union_ok = verify_union_per_language(root, "en")
    de_union_ok = verify_union_per_language(root, "de")

    all_ok = cross_ok and en_union_ok and de_union_ok
    print(f"\nOVERALL: {'OK' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    raise SystemExit(main())
