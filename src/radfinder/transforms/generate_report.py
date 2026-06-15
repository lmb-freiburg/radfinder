from copy import deepcopy
from typing import Any, Hashable, Mapping

from monai.config import KeysCollection
from monai.transforms import MapTransform, Randomizable
from radfinder.transforms.shared_utils import Language

from packg.constclass import Const


class RandomReportTransformd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        language: str = Language.EN,
        max_num_icd10: int = 20,
        keep_original_prob: float = 0.5,
        drop_prob: float = 0.3,
        allow_missing_keys: bool = False,
        allow_missing_icd10: bool = True,
        organ_text_replace_prob: float = 0.0,
        no_comp_replace_prob: float = 0.0,
        drop_impressions_prob: float = 0.0,
        drop_prefix: bool = False,
    ):
        assert all(
            str(key) in ["findings", "impressions", "icd10"] for key in keys
        ), "keys must be one of ['findings', 'impressions', 'icd10']"

        language = Language.verify_value(language)

        super().__init__(keys, allow_missing_keys)
        self.language = language
        self.max_num_icd10 = max_num_icd10
        self.keep_original_prob = keep_original_prob
        self.allow_missing_icd10 = allow_missing_icd10
        self.drop_prob = drop_prob
        self.organ_text_replace_prob = organ_text_replace_prob
        self.no_comp_replace_prob = no_comp_replace_prob
        self.drop_impressions_prob = drop_impressions_prob
        self.drop_prefix = drop_prefix
        self._rand_state = {}

    def randomize(self, data: Any = None) -> None:
        self._rand_state.clear()

        organ_text = data.get("organ_text", "") if data else ""
        self._rand_state["use_organ_text"] = (
            bool(organ_text)
            and self.organ_text_replace_prob > 0
            and self.R.random() < self.organ_text_replace_prob
        )

        self._rand_state["use_no_comp"] = (
            self.no_comp_replace_prob > 0 and self.R.random() < self.no_comp_replace_prob
        )

        # Select language if "both"
        if self.language == Language.BOTH:
            self._rand_state["selected_language"] = self.R.choice([Language.EN, Language.DE])
        else:
            self._rand_state["selected_language"] = self.language

        for key in self.keys:
            if str(key) == "findings":
                self._rand_state["drop_findings"] = self.R.random() < self.drop_prob
                self._rand_state["keep_findings_original"] = (
                    self.R.random() < self.keep_original_prob
                )

            elif str(key) == "impressions":
                self._rand_state["drop_impressions"] = self.R.random() < self.drop_impressions_prob
                self._rand_state["keep_impressions_original"] = (
                    self.R.random() < self.keep_original_prob
                )

            elif str(key) == "icd10":
                self._rand_state["drop_icd10"] = self.R.random() < self.drop_prob

    def _get_impressions(self, data: Mapping[Hashable, Any]) -> str:
        selected_lang = self._rand_state["selected_language"]
        lang_key = f"{selected_lang}_impressions"
        if lang_key not in data:
            lang_key = "impressions"
            if lang_key not in data:
                return None
        texts = data[lang_key]
        if self._rand_state.get("use_no_comp", False):
            no_comp = data.get("no_comp_impressions", "")
            if no_comp:
                texts = [no_comp]
        if not texts:
            return ""
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 1 or self._rand_state.get("keep_impressions_original", True):
            text = texts[0]
        else:
            text = self.R.choice(texts[1:])
        if self.drop_prefix:
            return f"{text}\n"
        return f"Impressions: {text}\n"

    def __call__(self, data: Mapping[Hashable, Any]) -> dict[Hashable, Any]:
        ret = dict()
        # deep copy all the unmodified data (remove report generation input)
        remove_keys = []
        for key in self.keys:
            remove_keys.append(key)
            for lang in [Language.EN, Language.DE]:
                remove_keys.append(f"{lang}_{key}")
        remove_keys.append("organ_text")
        remove_keys.append("no_comp_findings")
        remove_keys.append("no_comp_impressions")
        for key in set(data.keys()).difference(set(remove_keys)):
            ret[key] = deepcopy(data[key])

        self.randomize(data)

        if self._rand_state.get("use_organ_text", False):
            lines = data["organ_text"].split("\n")
            self.R.shuffle(lines)
            organ_findings = "Findings per organ:\n" + "\n".join(lines) + "\n"
            impressions = self._get_impressions(data)
            ret["report"] = f"{organ_findings}{impressions}"
            return ret

        selected_lang = self._rand_state["selected_language"]

        findings = ""
        impressions = ""
        icd10 = ""

        for key in self.keys:
            if str(key) == "findings":
                if self._rand_state.get("drop_findings", False):
                    continue
                # Try with language prefix first, then without prefix as fallback
                lang_key = f"{selected_lang}_findings"
                if lang_key not in data:
                    lang_key = "findings"
                    if lang_key not in data:
                        if not self.allow_missing_keys:
                            raise KeyError(
                                f"Key 'findings' or '{selected_lang}_findings' not found in data. Available keys: {list(data.keys())}"
                            )
                        continue
                texts = data[lang_key]
                if self._rand_state.get("use_no_comp", False):
                    no_comp = data.get("no_comp_findings", "")
                    if no_comp:
                        texts = [no_comp]
                if not texts:
                    continue
                if isinstance(texts, str):
                    texts = [texts]

                if len(texts) == 1 or self._rand_state.get("keep_findings_original", True):
                    text = texts[0]
                else:
                    text = self.R.choice(texts[1:])
                if self.drop_prefix:
                    findings = f"{text}\n"
                else:
                    findings = f"Findings: {text}\n".replace("Impressions", "").replace(
                        "impressions", ""
                    )

            elif str(key) == "impressions":
                if self._rand_state.get("drop_impressions", False):
                    continue
                impressions = self._get_impressions(data)
                if impressions is None and not self.allow_missing_keys:
                    selected_lang = self._rand_state["selected_language"]
                    raise KeyError(
                        f"Key 'impressions' or '{selected_lang}_impressions' not found in data. Available keys: {list(data.keys())}"
                    )

            elif str(key) == "icd10":
                if "icd10" not in data:
                    if not self.allow_missing_keys and not self.allow_missing_icd10:
                        raise KeyError(
                            f"Key 'icd10' not found in data. Available keys: {list(data.keys())}"
                        )
                    continue
                codes = data["icd10"]
                if isinstance(codes, str):
                    codes = codes.split(";")
                if (
                    not isinstance(codes, list)
                    or not codes
                    or self._rand_state.get("drop_icd10", False)
                ):
                    continue

                num_codes = (
                    len(codes) if self.max_num_icd10 < 0 else min(self.max_num_icd10, len(codes))
                )
                if len(codes) <= self.max_num_icd10:
                    selected_codes = codes
                else:
                    selected_codes = self.R.choice(codes, size=num_codes, replace=False)
                icd10 = f"ICD10: {'; '.join(selected_codes)}\n"

        ret["report"] = f"{findings}{impressions}{icd10}"
        # print(f"\nReport: {ret['report']}\n")
        return ret
