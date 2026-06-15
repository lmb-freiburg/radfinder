import pandas as pd
from rate.rate_common_utils import resolve_question_category_compat, unstack_questions


def test_resolve_question_category_by_unique_question() -> None:
    categories = {
        "Great Vessel in the abdomen": {
            "questions": [{"question": "Are there any aortic stents?"}]
        },
        "Devices in the thorax": {"questions": [{"question": "Is endotracheal tube present?"}]},
    }
    category, reason = resolve_question_category_compat(
        source_category="Great Vessel",
        question="Are there any aortic stents?",
        categories=categories,
    )
    assert category == "Great Vessel in the abdomen"
    assert reason.startswith("by_")


def test_unstack_questions_remaps_categories() -> None:
    categories = {
        "Great Vessel in the abdomen": {"questions": [{"question": "Are there any aortic stents?"}]}
    }
    ques = pd.DataFrame(
        [
            {
                "report_id": "r1",
                "category": "Great Vessel",
                "question": "Are there any aortic stents?",
                "answer": "No",
            }
        ]
    )
    out = unstack_questions(
        ques,
        reports_index_set={"r1"},
        categories=categories,
        do_category_compat_remap=True,
        verbose=False,
    )
    assert out["r1"]["Great Vessel in the abdomen"]["Are there any aortic stents?"] == "No"
