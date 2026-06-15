"""
Load the exported HF model and run inference on a real scan.

The two radiology reports below are synthetic, written only to illustrate
retrieval; they are not real reports for this scan.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

SEG_DIR = Path("assets/demo/s0859/segmentations")

# Synthetic reports for the retrieval demo (image -> which report matches).
MATCHING_REPORT = (
    "CT of the chest, abdomen, and pelvis. There is consolidation in the right lower lobe; "
    "the remaining lungs are clear without pleural effusion. Heart size is normal. A simple "
    "benign cyst is noted in the left kidney, and the liver, spleen, pancreas, and adrenal "
    "glands are unremarkable. No free fluid in the abdomen or pelvis."
)
UNRELATED_REPORT = (
    "Non-contrast CT of the head. There is no acute intracranial hemorrhage, mass effect, "
    "or territorial infarct. The ventricles and cortical sulci are normal for the patient's "
    "age, and the visualized paranasal sinuses and mastoid air cells are clear."
)


def main():
    MODEL_PATH = "lmb-freiburg/radfinder"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model.config.text_tokenizer_name)
    print(f"supports_localization={model.check_supports_localization()}\n")

    # The demo scan is a whole-body CT; crop it to chest+abdomen (lungs down to
    # kidneys), the region the chest-trained model is in distribution for, then
    # preprocess: resample to 0.75 x 0.75 x 3.0 mm and split into 128x128x32 windows.
    z0, z1 = chest_abdomen_zrange()
    with NamedTemporaryFile(suffix=".nii.gz") as tmp:
        nib.save(
            nib.as_closest_canonical(nib.load("assets/demo/s0859/ct.nii.gz")).slicer[:, :, z0:z1],
            tmp.name,
        )
        inputs = processor(tmp.name)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=model.dtype)  # (N, C, H, W, D)
    grid_size = inputs["grid_size"]  # (B, 3): (Hg, Wg, Dg)

    # --- Disease classification: consolidation vs lung nodule (T3 prompts) ---
    diseases = ["consolidation", "lung nodule"]
    prompts = [t.format(d=d) for d in diseases for t in ("There is {d}.", "There is no {d}.")]
    tok = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = F.normalize(model.encode_image_disease(pixel_values, grid_size), dim=-1)
        text_emb = F.normalize(model.encode_text(tok["input_ids"], tok["attention_mask"]), dim=-1)
    sims = (image_emb @ text_emb.T).reshape(len(diseases), 2)  # (n_disease, 2): [pos, neg]
    scale = model.model.criterion.t.exp().item()  # learned contrastive temperature
    print("disease classification:")
    for disease, (pos, neg) in zip(diseases, sims):
        margin = (pos - neg).item()
        call = "positive" if margin > 0 else "negative"
        pseudoprob = torch.sigmoid(torch.tensor(scale * margin)).item()
        print(
            f"  {disease:13s}: {call:<8} margin: {margin:+.4f}, "
            f"pseudo probability: {pseudoprob:4.0%}"
        )
    print()

    # --- Retrieval ---
    reports = tokenizer(
        [MATCHING_REPORT, UNRELATED_REPORT], padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        scan_emb = F.normalize(model.encode_image_retrieval(pixel_values, grid_size), dim=-1)
        report_emb = F.normalize(
            model.encode_text(reports["input_ids"], reports["attention_mask"]), dim=-1
        )
    report_sims = (scan_emb @ report_emb.T)[0]  # (2,)
    bias = model.model.criterion.b.item()  # learned contrastive bias
    report_probs = torch.softmax(report_sims * scale + bias, dim=0)  # (2,)
    labels = ["matching (chest+abdomen)", "unrelated (head CT)"]
    print("retrieval:")
    for label, sim, prob in zip(labels, report_sims, report_probs):
        print(f"  {label:20}: sim={sim.item():+.4f}, pseudo probability: {prob.item():4.0%}")

    # --- Localization ---
    # The model localizes only along the depth axis (axis2, inferior->superior)
    if model.check_supports_localization():
        snippet = "a cyst in the kidney"
        snippet = "kidney"
        snip = tokenizer([snippet], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.localize(pixel_values, grid_size, snip["input_ids"], snip["attention_mask"])
        slice_emb = F.normalize(out["scan_slice_emb"][0], dim=-1)  # (D2, 512)
        snip_emb = F.normalize(out["snippet_emb"][0], dim=-1)  # (512,)
        valid = out["scan_valid_depth_mask"][0]  # (D2,)
        scores = (slice_emb @ snip_emb).masked_fill(~valid, float("-inf"))  # (D2,)
        peak = int(scores.argmax())
        d2 = slice_emb.shape[0]

        gt_bin, lo, hi, zc = kidney_centroid_bin(
            d2, processor.pixdim[2], processor.sliding_window_size[2], z0, z1
        )
        mm_per_bin = processor.pixdim[2] * zc / d2
        err_mm = (peak - gt_bin) * mm_per_bin
        print(
            f"\nlocalization '{snippet}': peak depth bin {peak}/{d2 - 1} "
            f"(kidney spans bins {lo:.0f}-{hi:.0f}) "
            f"distance to kidney centroid {err_mm:+.1f} mm (gt bin {gt_bin:.1f})"
        )


def chest_abdomen_zrange(margin_mm: float = 15.0) -> tuple[int, int]:
    """Depth crop (RAS axis2 voxels) from the lung apex down to the bottom of the
    pelvis, i.e. everything except the neck/head above and the legs below."""
    lungs = [
        "lung_upper_lobe_left",
        "lung_upper_lobe_right",
        "lung_lower_lobe_left",
        "lung_lower_lobe_right",
        "lung_middle_lobe_right",
    ]
    (_, lung_top), zoom_z, depth = _mask_depth_span(lungs)
    (pelvis_bottom, _), _, _ = _mask_depth_span(["hip_left", "hip_right"])
    margin = round(margin_mm / zoom_z)
    return max(pelvis_bottom - margin, 0), min(lung_top + margin + 1, depth)


def kidney_centroid_bin(
    d2: int, pixdim_z: float, window_z: int, z0: int, z1: int
) -> tuple[float, float, float, int]:
    """Kidney depth extent within the chest+abdomen crop [z0, z1), in localization bins.

    Reorients the kidney masks to RAS (depth = last axis), restricts to the crop,
    resamples to the model's depth spacing, applies the same largest-multiple center
    crop, and scales onto the D2 localization bins. Returns the (fractional) center
    bin, the lower and upper extent bins, and the cropped depth in voxels.
    """
    profile, zoom_z = _kidney_depth_profile()
    profile = profile[z0:z1]
    spaced_depth = round(len(profile) * zoom_z / pixdim_z)  # depth after resampling
    zc = (spaced_depth // window_z) * window_z  # largest-multiple center crop
    crop_low = (spaced_depth - zc) // 2

    def to_bin(voxel: float) -> float:  # cropped depth voxel -> localization bin
        return (voxel * zoom_z / pixdim_z - crop_low) * d2 / zc

    centroid = float((np.arange(len(profile)) * profile).sum() / profile.sum())
    nonzero = np.nonzero(profile)[0]
    return to_bin(centroid), to_bin(int(nonzero.min())), to_bin(int(nonzero.max())), zc


def _kidney_depth_profile() -> tuple[np.ndarray, float]:
    union, zoom_z = None, None
    for side in ("kidney_left", "kidney_right"):
        img = nib.as_closest_canonical(nib.load(str(SEG_DIR / f"{side}.nii.gz")))
        zoom_z = float(img.header.get_zooms()[2])
        mask = np.asarray(img.dataobj) > 0.5
        union = mask if union is None else (union | mask)
    return union.sum(axis=(0, 1)).astype(float), zoom_z  # voxels per depth slice


def _mask_depth_span(names: list[str]) -> tuple[tuple[int, int], float, int]:
    union, zoom_z = None, None
    for name in names:
        img = nib.as_closest_canonical(nib.load(str(SEG_DIR / f"{name}.nii.gz")))
        zoom_z = float(img.header.get_zooms()[2])
        mask = np.asarray(img.dataobj) > 0.5
        union = mask if union is None else (union | mask)
    profile = union.sum(axis=(0, 1))
    nz = np.nonzero(profile)[0]
    return (int(nz.min()), int(nz.max())), zoom_z, len(profile)


if __name__ == "__main__":
    main()
