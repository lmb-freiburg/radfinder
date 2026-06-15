---
license: cc-by-nc-sa-4.0
base_model: cclaess/SPECTRE
language:
  - en
datasets:
  - ibrahimhamamci/CT-RATE
tags:
  - medical-imaging
  - computed-tomography
  - ct
  - vision-language
  - 3d-medical-imaging
  - radiology
  - retrieval
---

# RadFinder

Links: 
<a href="https://radfinder.github.io">Project page</a> —
<a href="https://arxiv.org/abs/2603.02026">Paper</a> —
<a href="https://github.com/lmb-freiburg/radfinder">Code</a> —
<a href="https://huggingface.co/collections/lmb-freiburg/radfinder">Models</a>

_Disease-Aware Vision–Language Pretraining for 3D CT_

We pretrain a 3D CT vision–language model on 159k report–volume pairs with two new supervision signals:
**prompt-based disease labels** for classification and **intra-scan snippet localization** for axial depth grounding.
A single unified model reaches state-of-the-art retrieval on CT-RATE, competitive disease
classification, and slice-level localization at 12 mm resolution.

## Usage

See the [GitHub repository](https://github.com/lmb-freiburg/radfinder).

## Training data

- RefCT (internal): ~98k report–volume pairs from ~50k patients at a single hospital; in-house clinical data, not publicly released.
- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) (CC BY-NC-SA 4.0)
- [Merlin](https://stanfordaimi.azurewebsites.net/) (Stanford AIMI non-commercial research DUA)
- [INSPECT](https://stanfordaimi.azurewebsites.net/) (Stanford AIMI non-commercial research DUA)

## Further acknowledgements

- The model and parts of the SigLIP training framework in `src/radfinder` are based on [SPECTRE](https://github.com/cclaess/SPECTRE)
- The text processing pipeline in `src/rate` is used to create binary labels based on text reports and is based on [RATE](https://github.com/YalaLab/rate)
- We thank the [MONAI](https://project-monai.github.io/), [timm](https://timm.fast.ai/), and
[Hugging Face transformers](https://github.com/huggingface/transformers) maintainers for the libraries
and all other package maintainers listed in `requirements.txt`
- The demo scan under `assets/demo/s0859/` is case `s0859` from [TotalSegmentator v2](https://zenodo.org/records/10047292) (Wasserthal et al., CC-BY-4.0).
- Funding, additional acknowledgements, full citations: see paper.

## License

- All code is MIT (see `LICENSE`) unless a file header says otherwise. Files in
  `src/rate/` that carry a `# Vendored from YalaLab/rate ... (ECL 2.0)` header
  are derivatives of the upstream rate package and are licensed under ECL 2.0
  (see `LICENSE_RATE`).
- RadFinder model weights are CC BY-NC-SA 4.0, see `LICENSE_MODELS`.
  - Note: the weights are subject to the original dataset licenses. Users intending to use RadFinder in commercial settings should verify dataset and model licensing and obtain any required permissions.

## Citation

If you use this code, models, or results, please cite:

```bibtex
@inproceedings{ging2026radfinder,
  author    = {Simon Ging and Philipp Arnold and Sebastian Walter and Hani Alnahas and Hannah Bast and Elmar Kotter and Jiancheng Yang and Behzad Bozorgtabar and Thomas Brox},
  title     = {Learning to Read Where to Look: Disease-Aware Vision--Language Pretraining for 3{D} {CT}},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- {MICCAI} 2026, Strasbourg, France, September 27 -- October 1, 2026, Proceedings},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear},
}
```
