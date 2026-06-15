import csv

from rate.reportstructuring_validation import (
    build_aligned_global_question_registries,
    map_pairs_to_registry_qids,
    normalize_yes_no_answer,
    sanitize_findings_text,
    vectorize_question_file,
)


def test_normalize_yes_no_exact_en() -> None:
    normalized, reason = normalize_yes_no_answer("Yes", lang="en")
    assert normalized == "Yes"
    assert reason == "exact"


def test_normalize_yes_no_with_punctuation_en() -> None:
    normalized, reason = normalize_yes_no_answer("No.", lang="en")
    assert normalized == "No"
    assert reason == "punctuation_wrapped"


def test_normalize_yes_no_xml_tag_en() -> None:
    normalized, reason = normalize_yes_no_answer("<answer>Yes</answer>", lang="en")
    assert normalized == "Yes"
    assert reason == "xml_tag_wrapped"


def test_normalize_yes_no_embedded_single_label_en() -> None:
    noisy = "Long report text with many details. Final output: No"
    normalized, reason = normalize_yes_no_answer(noisy, lang="en")
    assert normalized == "No"
    assert reason == "embedded_single_label"


def test_normalize_yes_no_unparseable_fallback_de() -> None:
    noisy = "Das ist ein langer Befund ohne klares Antwortlabel."
    normalized, reason = normalize_yes_no_answer(noisy, lang="de")
    assert normalized == "Nein"
    assert reason == "unparseable_fallback_to_no"


def test_sanitize_findings_cut_at_newline() -> None:
    raw = "No relevant findings.\nAdditional report noise that should be removed."
    cleaned, reasons = sanitize_findings_text(raw, max_chars=280)
    assert cleaned == "No relevant findings."
    assert "cut_at_newline" in reasons


def test_sanitize_findings_hard_truncate() -> None:
    raw = "A" * 600
    cleaned, reasons = sanitize_findings_text(raw, max_chars=100)
    assert len(cleaned) == 100
    assert cleaned.endswith("...")
    assert "hard_truncate" in reasons


def test_vectorize_question_file_one_row_per_report(tmp_path) -> None:
    src = tmp_path / "questions_abdomen_train.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["report_id", "category", "question", "answer"])
        writer.writeheader()
        writer.writerow(
            {
                "report_id": "r1",
                "category": "Pancreas",
                "question": "Is there acute pancreatitis?",
                "answer": "Yes",
            }
        )
        writer.writerow(
            {
                "report_id": "r1",
                "category": "Liver",
                "question": "Are there any hepatic cysts?",
                "answer": "No",
            }
        )
        writer.writerow(
            {
                "report_id": "r2",
                "category": "Pancreas",
                "question": "Is there acute pancreatitis?",
                "answer": "<answer>No</answer>",
            }
        )

    qmap = {
        ("Pancreas", "Is there acute pancreatitis?"): "q0001",
        ("Liver", "Are there any hepatic cysts?"): "q0002",
    }
    dst = tmp_path / "vector.csv"
    stats = vectorize_question_file(src_path=src, dst_path=dst, question_id_map=qmap, lang="en")
    assert stats.rows_read == 3
    assert stats.reports_written == 2
    assert stats.unknown_questions == 0

    with dst.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["report_id"] == "r1"
    assert rows[0]["q0001"] == "1"
    assert rows[0]["q0002"] == "0"
    assert rows[1]["report_id"] == "r2"
    assert rows[1]["q0001"] == "0"
    assert rows[1]["q0002"] == "0"


def test_map_pairs_to_registry_qids_produces_global_unique_qids() -> None:
    registries = build_aligned_global_question_registries(
        {
            "en": [("Pancreas", "Q1"), ("Heart", "Q2"), ("Lung", "Q3")],
            "de": [("Pankreas", "F1"), ("Herz", "F2"), ("Lunge", "F3")],
        }
    )

    abdomen_map = map_pairs_to_registry_qids([("Pancreas", "Q1")], registries["en"])
    chest_map = map_pairs_to_registry_qids([("Heart", "Q2")], registries["en"])

    assert abdomen_map[("Pancreas", "Q1")] == "q0001"
    assert chest_map[("Heart", "Q2")] == "q0002"
    assert abdomen_map[("Pancreas", "Q1")] != chest_map[("Heart", "Q2")]


def test_map_pairs_to_registry_qids_supports_category_compat_mapping() -> None:
    registry = {
        ("Devices", "Is heart valve replacement present?"): "q0001",
        ("Great Vessels", "Is pulmonary embolism suspected?"): "q0002",
    }
    registry_categories = {
        "Devices": {"questions": [{"question": "Is heart valve replacement present?"}]},
        "Great Vessels": {"questions": [{"question": "Is pulmonary embolism suspected?"}]},
    }
    source_pairs = [
        ("Devices in the Thorax", "Is heart valve replacement present?"),
        ("Great Vessels in the Thorax", "Is pulmonary embolism suspected?"),
    ]
    qmap = map_pairs_to_registry_qids(
        config_pairs=source_pairs,
        registry=registry,
        registry_categories=registry_categories,
    )
    assert qmap[("Devices in the Thorax", "Is heart valve replacement present?")] == "q0001"
    assert qmap[("Great Vessels in the Thorax", "Is pulmonary embolism suspected?")] == "q0002"


def test_build_aligned_global_question_registries_requires_equal_lengths() -> None:
    try:
        build_aligned_global_question_registries(
            {"en": [("A", "Q1"), ("B", "Q2")], "de": [("A", "F1")]}
        )
        assert False, "Expected ValueError for unequal canonical lengths"
    except ValueError as exc:
        assert "Canonical question counts differ" in str(exc)
