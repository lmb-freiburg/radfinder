import csv
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rate.rate_common_utils import resolve_question_category_compat

LANG_LABELS = {
    "en": ("Yes", "No"),
    "de": ("Ja", "Nein"),
}


@dataclass
class FileStats:
    file_path: str
    total_rows: int = 0
    changed_rows: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)

    def add_reason(self, reason: str) -> None:
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1


@dataclass
class VectorizationStats:
    file_path: str
    rows_read: int = 0
    reports_written: int = 0
    unknown_questions: int = 0
    conflicting_answers: int = 0


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_yes_no_answer(raw_answer: str, lang: str) -> tuple[str, str]:
    assert lang in LANG_LABELS, f"Unsupported language: {lang}"
    yes_label, no_label = LANG_LABELS[lang]
    valid_labels = {yes_label.lower(): yes_label, no_label.lower(): no_label}

    answer = raw_answer.strip()
    if not answer:
        return no_label, "empty_fallback_to_no"

    answer_lc = answer.lower()
    if answer_lc in valid_labels:
        return valid_labels[answer_lc], "exact"

    exactish = re.fullmatch(r"[^\w]*(yes|no|ja|nein)[^\w]*", answer_lc)
    if exactish:
        token = exactish.group(1)
        if token in valid_labels:
            return valid_labels[token], "punctuation_wrapped"

    tag_match = re.search(r"<answer>\s*(yes|no|ja|nein)\s*</answer>", answer_lc)
    if tag_match:
        token = tag_match.group(1)
        if token in valid_labels:
            return valid_labels[token], "xml_tag_wrapped"

    # Try to recover a consistent answer from noisy text.
    found = re.findall(r"\b(yes|no|ja|nein)\b", answer_lc)
    in_lang = [token for token in found if token in valid_labels]
    unique = set(in_lang)

    if len(unique) == 1:
        token = next(iter(unique))
        return valid_labels[token], "embedded_single_label"

    return no_label, "unparseable_fallback_to_no"


def sanitize_findings_text(raw_findings: str, max_chars: int = 280) -> tuple[str, list[str]]:
    findings = raw_findings.strip()
    reasons: list[str] = []

    if "\n" in findings:
        first_non_empty_line = next(
            (line.strip() for line in findings.splitlines() if line.strip()), ""
        )
        findings = first_non_empty_line
        reasons.append("cut_at_newline")

    findings = _normalize_space(findings)

    # If text still looks too long, keep only the first sentence.
    if len(findings) > max_chars:
        sentence_split = re.split(r"(?<=[.!?])\s+", findings)
        if sentence_split and sentence_split[0]:
            findings = sentence_split[0]
            reasons.append("kept_first_sentence")

    findings = _normalize_space(findings)

    if len(findings) > max_chars:
        findings = findings[: max_chars - 3].rstrip() + "..."
        reasons.append("hard_truncate")

    return findings, reasons


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _sample_random_row(path: Path, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    chosen: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=1):
            if rng.randrange(i) == 0:
                chosen = row
    return chosen


def show_random_examples(data_root: Path, sample_size: int, seed: int) -> None:
    rng = random.Random(seed)
    question_files = sorted(data_root.glob("p0rate_*/questions_*.csv"))
    findings_files = sorted(data_root.glob("p0rate_*/category_findings_*.csv"))

    print("\n=== Random Question Samples ===")
    for path in rng.sample(question_files, k=min(sample_size, len(question_files))):
        row = _sample_random_row(path, seed=seed + hash(path.as_posix()) % 100000)
        print(f"{path.as_posix()}: {row}")

    print("\n=== Random Findings Samples ===")
    for path in rng.sample(findings_files, k=min(sample_size, len(findings_files))):
        row = _sample_random_row(path, seed=seed + hash(path.as_posix()) % 100000)
        print(f"{path.as_posix()}: {row}")


def verify_question_file(
    src_path: Path, dst_path: Path | None, lang: str, write_cleaned: bool
) -> FileStats:
    stats = FileStats(file_path=src_path.as_posix())
    dst_writer = None
    dst_handle = None

    if write_cleaned:
        assert dst_path is not None
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_handle = dst_path.open("w", encoding="utf-8", newline="")
        dst_writer = csv.DictWriter(
            dst_handle, fieldnames=["report_id", "category", "question", "answer"]
        )
        dst_writer.writeheader()

    try:
        with src_path.open("r", encoding="utf-8", newline="") as src_file:
            for row in csv.DictReader(src_file):
                stats.total_rows += 1
                normalized, reason = normalize_yes_no_answer(row["answer"], lang=lang)
                stats.add_reason(reason)
                if normalized != row["answer"]:
                    stats.changed_rows += 1
                    row["answer"] = normalized
                if dst_writer is not None:
                    dst_writer.writerow(row)
    finally:
        if dst_handle is not None:
            dst_handle.close()

    return stats


def verify_findings_file(
    src_path: Path, dst_path: Path | None, write_cleaned: bool, max_chars: int
) -> FileStats:
    stats = FileStats(file_path=src_path.as_posix())
    dst_writer = None
    dst_handle = None

    if write_cleaned:
        assert dst_path is not None
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_handle = dst_path.open("w", encoding="utf-8", newline="")
        dst_writer = csv.DictWriter(dst_handle, fieldnames=["report_id", "category", "findings"])
        dst_writer.writeheader()

    try:
        with src_path.open("r", encoding="utf-8", newline="") as src_file:
            for row in csv.DictReader(src_file):
                stats.total_rows += 1
                cleaned, reasons = sanitize_findings_text(row["findings"], max_chars=max_chars)
                if not reasons:
                    stats.add_reason("unchanged")
                else:
                    for reason in reasons:
                        stats.add_reason(reason)
                if cleaned != row["findings"]:
                    stats.changed_rows += 1
                    row["findings"] = cleaned
                if dst_writer is not None:
                    dst_writer.writerow(row)
    finally:
        if dst_handle is not None:
            dst_handle.close()

    return stats


def load_question_pairs_from_config(config_path: Path) -> list[tuple[str, str]]:
    content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    categories = content.get("categories") or {}
    pairs: list[tuple[str, str]] = []
    for category, category_info in categories.items():
        for q_obj in category_info.get("questions", []):
            question = q_obj["question"] if isinstance(q_obj, dict) else q_obj
            pairs.append((category, question))
    return pairs


def build_question_id_map(config_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    return {pair: f"q{i:04d}" for i, pair in enumerate(config_pairs, start=1)}


def build_aligned_global_question_registries(
    canonical_pairs_by_lang: dict[str, list[tuple[str, str]]],
) -> dict[str, dict[tuple[str, str], str]]:
    if not canonical_pairs_by_lang:
        raise ValueError("canonical_pairs_by_lang is empty")

    lengths = {lang: len(pairs) for lang, pairs in canonical_pairs_by_lang.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(
            "Canonical question counts differ across languages: "
            + ", ".join(f"{lang}={count}" for lang, count in sorted(lengths.items()))
        )

    registries: dict[str, dict[tuple[str, str], str]] = {}
    for lang, pairs in canonical_pairs_by_lang.items():
        registry: dict[tuple[str, str], str] = {}
        for i, pair in enumerate(pairs, start=1):
            if pair in registry:
                raise ValueError(f"Duplicate canonical question pair for {lang}: {pair}")
            registry[pair] = f"q{i:04d}"
        registries[lang] = registry
    return registries


def map_pairs_to_registry_qids(
    config_pairs: list[tuple[str, str]],
    registry: dict[tuple[str, str], str],
    registry_categories: dict | None = None,
) -> dict[tuple[str, str], str]:
    qid_map: dict[tuple[str, str], str] = {}
    missing: list[tuple[str, str]] = []
    for category, question in config_pairs:
        pair = (category, question)
        qid = registry.get(pair)
        if qid is None and registry_categories is not None:
            mapped_category, _ = resolve_question_category_compat(
                source_category=category,
                question=question,
                categories=registry_categories,
            )
            qid = registry.get((mapped_category, question))
        if qid is None:
            missing.append(pair)
            continue
        qid_map[pair] = qid
    if missing:
        sample = ", ".join(f"[{c}] {q}" for c, q in missing[:3])
        raise ValueError(
            f"{len(missing)} questions are missing from global registry. " f"Examples: {sample}"
        )
    return qid_map


def write_question_map_csv(question_id_map: dict[tuple[str, str], str], dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "category", "question"])
        writer.writeheader()
        for (category, question), qid in sorted(question_id_map.items(), key=lambda item: item[1]):
            writer.writerow({"qid": qid, "category": category, "question": question})


def vectorize_question_file(
    src_path: Path,
    dst_path: Path,
    question_id_map: dict[tuple[str, str], str],
    lang: str,
    categories: dict | None = None,
) -> VectorizationStats:
    yes_label, _ = LANG_LABELS[lang]
    qid_order = [qid for _, qid in sorted(question_id_map.items(), key=lambda item: item[1])]
    stats = VectorizationStats(file_path=src_path.as_posix())
    report_vectors: dict[str, dict[str, int]] = {}

    with src_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            stats.rows_read += 1
            report_id = row["report_id"]
            category = row["category"]
            if categories is not None:
                category, _ = resolve_question_category_compat(
                    source_category=row["category"],
                    question=row["question"],
                    categories=categories,
                )
            key = (category, row["question"])
            qid = question_id_map.get(key)
            if qid is None:
                stats.unknown_questions += 1
                continue

            normalized, _ = normalize_yes_no_answer(row["answer"], lang=lang)
            value = 1 if normalized == yes_label else 0
            if report_id not in report_vectors:
                report_vectors[report_id] = {q: 0 for q in qid_order}

            previous = report_vectors[report_id][qid]
            if previous != value and (previous == 1 or value == 1):
                if previous != value:
                    stats.conflicting_answers += 1
                report_vectors[report_id][qid] = 1
            else:
                report_vectors[report_id][qid] = value

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["report_id"] + qid_order
    with dst_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report_id in sorted(report_vectors.keys()):
            row = {"report_id": report_id}
            row.update(report_vectors[report_id])
            writer.writerow(row)

    stats.reports_written = len(report_vectors)
    return stats
