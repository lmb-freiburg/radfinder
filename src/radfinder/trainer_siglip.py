"""Trainer for SigLIP with support for gradient accumulation."""

import gc
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from attrs import define
from radfinder.losses.prompt_rate_loss import PromptRateLoss, PromptSamplingC
from radfinder.models.modeling import last_token_pool
from radfinder.models.vision_language import (
    GlobalContrastiveOutput,
    GlobalResC,
    LocalizationOutput,
    SigLIP,
    SnippetAlignmentModeC,
)
from radfinder.models.vision_transformer import get_default_snippet_alignment
from radfinder.tasks.run_task import BOOTSTRAPPABLE_TASK_TYPES, run_task_by_type
from radfinder.utils.logging_utils import log_debug, log_info, log_warning
from radfinder.utils.scheduler import cosine_warmup_schedule
from torch.optim import AdamW

from packg.constclass import Const
from packg.iotools import dump_json, load_json
from packg.tqdmext import tqdm_max_ncols
from typedparser.objects import repr_value
from visiontext.distutils import is_main_process

_PROMPT_RATE_REF_BATCH_SIZE = 256
_LOCALIZATION_REF_BATCH_SIZE = 16


class ProjGradClipC(Const):
    """
    How to handle gradient clipping for copy projection heads vs global model.

    Config field: do_snippet_alignment.proj_grad_clip
    """

    TOGETHER = "together"  # clip all parameters together (default, original behavior)
    SEPARATE = "separate"  # clip copy heads and global model separately
    ONLY_GLOBAL = "only_global"  # only clip global model, skip copy heads
    ONLY_PROJ_COPY = "only_proj_copy"  # only clip copy heads, skip global model
    NONE = "none"  # no gradient clipping at all


@define(slots=False)
class Trainer:
    """Trainer for SigLIP with support for gradient accumulation."""

    model: SigLIP
    optimizer: AdamW
    dataloader: object
    accelerator: Accelerator
    train_config: dict
    model_config: dict | None = None
    val_dict: dict | None = None
    # Prompt-rate auxiliary loss components (None when disabled)
    prompt_rate_loss: PromptRateLoss | None = None
    prompt_rate_labels: dict | None = None  # scan_key -> torch.Tensor(319,)
    prompt_rate_tokenizer: object | None = None
    prompt_rate_pos_prompts: list | None = None
    prompt_rate_neg_prompts: list | None = None
    bootstrap: bool = False

    def _get_copy_head_param_ids(self) -> set[int]:
        """Get parameter ids of copy projection heads."""
        ids = set()
        unwrapped = self.accelerator.unwrap_model(self.model)
        for name in ("projection_image_copy", "projection_text_copy"):
            mod = getattr(unwrapped, name, None)
            if mod is not None:
                for p in mod.parameters():
                    ids.add(id(p))
        return ids

    def clip_grad_norm(self, max_norm: float) -> torch.Tensor | None:
        """
        Clip gradient norm respecting proj_grad_clip config.

        Returns the gradient norm (of the global parameters) or None if clipping is disabled.
        """
        if max_norm <= 0:
            return None

        snippet_cfg = self.train_config["train"].get(
            "do_snippet_alignment", get_default_snippet_alignment()
        )
        mode = snippet_cfg.get("proj_grad_clip", ProjGradClipC.TOGETHER)
        ProjGradClipC.verify_value(mode)

        if mode == ProjGradClipC.NONE:
            return None

        if mode == ProjGradClipC.TOGETHER:
            return self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm)

        copy_ids = self._get_copy_head_param_ids()
        global_params = [p for p in self.model.parameters() if id(p) not in copy_ids]
        copy_params = [p for p in self.model.parameters() if id(p) in copy_ids]

        grad_norm = None
        if mode == ProjGradClipC.SEPARATE:
            grad_norm = self.accelerator.clip_grad_norm_(global_params, max_norm)
            if copy_params:
                self.accelerator.clip_grad_norm_(copy_params, max_norm)
        elif mode == ProjGradClipC.ONLY_GLOBAL:
            grad_norm = self.accelerator.clip_grad_norm_(global_params, max_norm)
        elif mode == ProjGradClipC.ONLY_PROJ_COPY:
            if copy_params:
                grad_norm = self.accelerator.clip_grad_norm_(copy_params, max_norm)
        return grad_norm

    def update_lr_and_wd(self) -> float:
        """
        Update learning rate and weight decay for all optimizer parameter groups.

        Returns:
            Current learning rate
        """
        lr = cosine_warmup_schedule(
            self.global_step,
            max_steps=self.total_num_steps,
            start_value=self.train_config["optim"]["lr"],
            end_value=self.train_config["optim"]["min_lr"],
            warmup_steps=self.warmup_num_steps,
            warmup_start_value=0.0,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group["lr_mult"]
            param_group["weight_decay"] = (
                self.train_config["optim"]["weight_decay"] * param_group["wd_mult"]
            )
        return lr

    def start_training(self, start_epoch_num: int = 0):
        """Main training loop."""
        max_epochs = self.train_config["optim"]["epochs"]
        if start_epoch_num >= max_epochs:
            log_info(f"{start_epoch_num=} > {max_epochs=}. Skipping training.")
            return
        # Calculate training steps
        self.accum_steps = self.train_config["train"]["accum_steps"]
        self.batches_per_epoch = len(self.dataloader)
        self.steps_per_epoch = self.batches_per_epoch // self.accum_steps
        self.total_num_steps = self.train_config["optim"]["epochs"] * self.steps_per_epoch
        self.warmup_num_steps = self.train_config["optim"]["warmup_epochs"] * self.steps_per_epoch
        self.global_step = start_epoch_num * self.steps_per_epoch
        self.accum_dict = {"batch_data": [], "image_embeddings": [], "text_embeddings": []}

        # self.save_checkpoint(0, val_score_lower_better=99999)
        # raise
        if start_epoch_num == 0 and self.train_config["train"]["val_first"] <= 0:
            _ = self.validate_epoch(start_epoch_num)
        for epoch in range(start_epoch_num, max_epochs):
            epoch_loss = self.run_training_epoch(epoch)
            epoch_time = time.time() - self.epoch_start
            log_info(f"Epoch {epoch+1} completed in {epoch_time:.1f}s - Avg loss: {epoch_loss:.4f}")

            # Log epoch metrics
            self.accelerator.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch_time": epoch_time,
                },
                step=self.global_step,
            )

            # Aggressive memory cleanup at epoch end
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            n_epochs_trained = epoch + 1
            val_meanr = self.validate_epoch(n_epochs_trained)
            self.save_checkpoint(n_epochs_trained, val_score_lower_better=val_meanr)

        # End training
        self.accelerator.end_training()
        log_info("Training completed!")

    def run_training_epoch(self, epoch: int) -> float:
        """Run one epoch of training."""
        self.model.train()
        self.model.set_frozen_state(
            self.model_config, self.train_config, is_init=False, epoch=epoch
        )
        self.accum_batch_list = []
        self.accum_output_dict = {}
        self._accum_loc_loss = 0.0
        self._accum_loc_steps = 0
        self._accum_pr_loss = 0.0
        self._accum_pr_steps = 0
        accum_steps = self.train_config["train"]["accum_steps"]
        self.global_epoch = epoch
        epoch_loss = 0.0
        self.epoch_start = time.time()

        pbar = tqdm_max_ncols(
            total=len(self.dataloader) // accum_steps,
            desc=f"Epoch {epoch+1}/{self.train_config['optim']['epochs']}",
            smoothing=0.0,
            disable=not is_main_process(),
        )

        for batch_idx, batch in enumerate(self.dataloader):
            # here the batch is already on the correct device due to accelerator.prepare(dataloader)
            if self.accum_steps == 1:
                losses, lr = self.run_batch_regular(batch, batch_idx)
            else:
                losses, lr = self.run_batch_with_accum(batch, batch_idx)
            if losses is None:
                # accumulation step, no loss yet
                continue
            loss_value = losses.pop("loss")
            epoch_loss += loss_value

            # Log metrics to wandb
            log_dict = {
                "train/loss": loss_value,
                "train/lr": lr,
                "train/epoch": epoch + 1,
                "train/step": self.global_step,
            }
            for k, v in losses.items():
                log_dict[f"train/{k}"] = v
            try:
                loss_t = self.model.criterion.t.detach().item()
                loss_b = self.model.criterion.b.detach().item()
                log_dict["train/temperature"] = loss_t
                log_dict["train/bias"] = loss_b
            except AttributeError:
                pass
            self.accelerator.log(log_dict, step=self.global_step)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_value:.4f}", "lr": f"{lr:.2e}"})
            if ((batch_idx + 1) % accum_steps) == 0:
                pbar.update(1)

        pbar.close()

        # Average loss over optimizer steps, not batches
        avg_loss = epoch_loss / self.steps_per_epoch
        return avg_loss

    def run_training_forward_pass(self, batch: dict, batch_num: int = 0) -> GlobalContrastiveOutput:
        """
        Global contrastive forward pass for training.

        Args:
            batch: Input batch dictionary
            batch_num: Current batch number (only prints debug info when 0-4, -1 to suppress)
        """
        do_print = batch_num == 0 and self.accelerator.is_main_process and self.global_epoch == 0
        if do_print:
            log_debug(f"Batch {batch_num} input to model")
            for k, v in batch.items():
                if k in {"image_metadata", "slices"}:
                    continue
                log_debug(f"   {k}: {repr_value(v, depth=1, key=k)}")
        output = self.model.forward_global_contrastive(batch)
        if do_print:
            log_debug(
                f"Batch {batch_num} output from model: "
                f"{', '.join(f'{k}: {v.shape}' for k, v in output.items())}"
            )
        return output

    def run_localization_forward_pass(self, batch: dict) -> LocalizationOutput:
        """Localization forward pass: returns scan_slice_emb / snippet_emb / etc."""
        return self.model.forward_localization(batch)

    def run_batch_regular(self, batch: dict, batch_idx: int) -> tuple[dict[str, float], float]:
        """Regular optimization without batch accumulation."""
        # raise NotImplementedError("disabled for simplicity")
        return self.run_batch_with_accum(batch, batch_idx)

    def run_batch_with_accum(
        self, batch: dict, batch_idx: int
    ) -> tuple[dict[str, float] | None, float]:
        """
        Accumulate gradients over multiple steps.

        Two-phase approach to save memory:
          Phase 1 (per batch, as they arrive):
            - If localization mode: forward WITH gradients, backward localization loss
              immediately (intra-sample, no cross-batch negatives needed), cache only
              detached global embeddings.
            - Otherwise: forward without gradients, cache embeddings for phase 2.
          Phase 2 (after all batches accumulated):
            - Re-forward each batch with gradients for global SigLIP loss
              (+ local siglip loss if applicable). Last micro-batch syncs DDP gradients.
          Then optimizer step.
        """
        acc_steps = self.accum_steps
        assert not self.train_config["train"]["accum_cpu_offload"], "removed for simplicity"

        # loss config
        snippet_cfg = self.model.do_snippet_alignment
        snippet_mode = snippet_cfg.get("snippet_mode", SnippetAlignmentModeC.NONE)
        model_settings = self.train_config.get("train", {}).get("model_settings", {})
        global_res = model_settings.get("global_res", GlobalResC.LOW_RES)
        global_loss_weight = model_settings.get("global_loss_weight", 1.0)
        run_loc_loss = (
            self._get_effective_loc_weight() > 0.0
            and snippet_mode == SnippetAlignmentModeC.AXIS_LOCALIZATION
        )
        run_phase2 = global_loss_weight != 0.0

        batch_size = self.train_config["train"]["batch_size"]
        loc_correction = batch_size / _LOCALIZATION_REF_BATCH_SIZE
        pr_correction = batch_size / _PROMPT_RATE_REF_BATCH_SIZE
        # run_phase3_pr = run_loc_loss and self.prompt_rate_loss is not None

        do_print = (
            batch_idx + 1 <= acc_steps
            and self.global_epoch == 0
            and self.accelerator.is_main_process
        )
        if do_print and batch_idx == 0:
            log_debug(f"Phase1 corrections: loc={loc_correction:.4f} pr={pr_correction:.4f}")

        # Save batch for phase 2 re-forward
        _input_accum_whitelist = [
            "image_backbone_cls",
            "image_backbone_patch_average",
            "window_mask",
            "image_grid_shape",
            "image_metadata",
            "report",
            "report_input_ids",
            "report_hidden_state_mask",
            "features_dir",
            "filename",
            "scan_key",
        ]
        # axis2 features must be re-forwarded in phase 2 if the global loss runs on them.
        if global_res == GlobalResC.AXIS2:
            _input_accum_whitelist.extend(["image_backbone_patch_axis2", "slice_axis2_mask"])

        if run_phase2:
            batch_subset = {}
            for k, v in batch.items():
                if k not in _input_accum_whitelist:
                    continue
                if not isinstance(v, torch.Tensor):
                    batch_subset[k] = v
                    continue
                batch_subset[k] = v.detach()
            self.accum_batch_list.append(batch_subset)

        # Encode prompt-rate prompts once after every gradient update
        prompt_rate_cfg = self.train_config["train"].get("prompt_rate", {})
        prl_text_grad = prompt_rate_cfg.get("text_grad", False)

        # Prompt-rate auxiliary loss (per-sample, uses current micro-batch only)
        if run_loc_loss:
            # Phase 1a: localization (intra-sample loss, backward immediately).
            # Phase 1b: global contrastive forward, detached for cross-batch reuse in phase 2.
            # (Only run the global forward if phase 2 will actually consume the cache.)
            with self.accelerator.no_sync(self.model), self.accelerator.autocast():
                valid_slices = batch.get("valid_slices")
                batch_has_loc_data = valid_slices is not None and bool(valid_slices.any().item())
                if batch_has_loc_data:
                    o_loc = self.run_localization_forward_pass(batch)
                    loc_loss = self._compute_localization_loss(o_loc)
                    if loc_loss is None:
                        raise RuntimeError(
                            "Localization loss is enabled and the batch has valid slices, "
                            "but the model did not return localization outputs."
                        )
                    loc_loss_val = loc_loss.detach().item()
                    self._accum_loc_loss += loc_loss_val
                    self._accum_loc_steps += 1 if loc_loss_val > 0 else 0
                    self.accelerator.backward(loc_loss * loc_correction)
                o = None
                if run_phase2:
                    with torch.no_grad():
                        o = self.run_training_forward_pass(batch, batch_idx)
        else:
            # No localization: forward without gradients
            phase1_no_grad = self.train_config["train"].get("accum_phase1_no_grad", True)
            no_grad_ctx = torch.no_grad() if phase1_no_grad else nullcontext()
            with no_grad_ctx:
                with self.accelerator.autocast():
                    o = self.run_training_forward_pass(batch, batch_idx)

        if not run_phase2:
            # check for early exists before phase 2
            if (batch_idx + 1) % acc_steps != 0:
                return None, 0.0
            lr = self.update_lr_and_wd()
            if do_print:
                log_debug("Global loss disabled, skipping phase 2.")
            # no local siglip loss and global loss disabled, skip phase 2
            # just do optimizer step with whatever gradients we have from phase 1 (e.g. localization)
            losses = self._optimizer_step(0.0, acc_steps, {}, {}, do_print)
            return losses, lr

        # Find reports with empty/one-word text, to later drop them from the contrastive loss.
        # Empty text will be 'Impressions: \n<|endoftext|>' which is 6 tokens
        # so we use >10 as a "has text" heuristic to be safe
        report_mask = batch.get("report_hidden_state_mask")
        if report_mask is not None:
            has_text = report_mask.sum(dim=-1) > 10
        else:
            has_text = o.image_embeddings.new_ones(o.image_embeddings.shape[0], dtype=torch.bool)
        cached = {
            "image_embeddings": o.image_embeddings.detach(),
            "text_embeddings": o.text_embeddings.detach(),
            "has_text": has_text,
        }
        del o
        for k, v in cached.items():
            if k not in self.accum_output_dict:
                self.accum_output_dict[k] = []
            self.accum_output_dict[k].append(v)
        del cached

        if do_print and (batch_idx <= 5 or batch_idx >= acc_steps - 5):
            log_debug(
                f"Accum: {batch_idx=:3d}: "
                + ", ".join(
                    f"{k}: {v[-1].shape if v[-1] is not None else None}"
                    for k, v in self.accum_output_dict.items()
                )
            )

        if (batch_idx + 1) % acc_steps != 0:
            return None, 0.0

        # --- Phase 2: global SigLIP loss with cross-batch accumulation ---
        if do_print:
            log_debug(f"Accum backward with {len(self.accum_batch_list)} batches")
        lr = self.update_lr_and_wd()
        total_loss = 0.0

        loss_accum = defaultdict(float)
        loss_steps = defaultdict(int)
        run_phase3_pr = self.prompt_rate_loss is not None
        # Full-batch text-validity mask. has_text carries no gradient and the
        # concatenation order is fixed (batches 0..acc_steps-1), so it lines up with
        # all_i/all_t for every j and is computed once here.
        all_has_text = torch.cat(self.accum_output_dict["has_text"])
        assert all_has_text.any(), "all reports empty, empty contrastive batch would NaN weights"
        n_excluded = int((~all_has_text).sum().item())
        if n_excluded > 0:
            log_debug(
                f"Excluding {n_excluded}/{all_has_text.numel()} empty-report samples "
                f"from the contrastive batch"
            )
        for j in range(acc_steps):
            batch_j = self.accum_batch_list[j]
            ctx = (
                self.accelerator.no_sync(self.model)
                if j < acc_steps - 1 or run_phase3_pr
                else nullcontext()
            )
            with ctx:
                loss_j = {}
                with self.accelerator.autocast():
                    o_j = self.run_training_forward_pass(batch_j, batch_num=-1)
                    if do_print and (j <= 5 or j >= acc_steps - 5):
                        log_debug(
                            f"[Accum Debug]   Micro-batch {j}/{acc_steps} "
                            f"output: {', '.join(f'{k}: {v.shape}' for k, v in o_j.items())}"
                        )

                    # Concatenate: accumulated features from other batches + current batch.
                    # The one from the current batch will have the gradient, while all the
                    # others are detached and used for negatives.
                    accum_i = self.accum_output_dict["image_embeddings"]
                    accum_t = self.accum_output_dict["text_embeddings"]
                    all_i = torch.cat(accum_i[:j] + [o_j.image_embeddings] + accum_i[j + 1 :])
                    all_t = torch.cat(accum_t[:j] + [o_j.text_embeddings] + accum_t[j + 1 :])
                    all_i = all_i[all_has_text]
                    all_t = all_t[all_has_text]

                    global_loss_j = self.model.criterion(all_i, all_t)
                    loss_j["global_loss"] = global_loss_j
                    loss_steps["global_loss"] += 1

                total_loss_j = sum(loss_j.values())
                total_loss_j_item = total_loss_j.item()  # pyright: ignore
                total_loss += total_loss_j_item
                if do_print and (j <= 5 or j >= acc_steps - 5):  # print only first/last 5 batches
                    log_debug(
                        f"[Accum Debug] Micro-batch {j}/{acc_steps}: "
                        f"{', '.join(f'{k}: {v.item():.4f}' for k, v in loss_j.items())} "
                        f"total={total_loss_j_item:.4f}"
                    )
                self.accelerator.backward(total_loss_j)
            for loss_j_key, loss_j_value in loss_j.items():
                loss_accum[loss_j_key] += loss_j_value.detach().item()

        if run_phase3_pr:
            self._run_phase3_prompt_rate(acc_steps, do_print)

        losses = self._optimizer_step(total_loss, acc_steps, loss_accum, loss_steps, do_print)
        return losses, lr

    def _optimizer_step(self, total_loss, acc_steps, loss_accum, loss_steps, do_print):
        loss_accum = dict(loss_accum)
        loss_steps = dict(loss_steps)
        assert loss_accum.keys() == loss_steps.keys(), f"{loss_accum.keys()=} {loss_steps.keys()}="
        max_norm = self.train_config["optim"].get("clip_grad_norm", 0)
        grad_norm = self.clip_grad_norm(max_norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        losses = {"loss": total_loss / acc_steps}
        if grad_norm is not None:
            losses["grad_norm"] = grad_norm.item()
        for k, v in loss_accum.items():
            steps = loss_steps[k]
            if steps > 0:
                losses[k] = v / steps
        for loss_name, loss_value, loss_steps in (
            ("localization_loss", self._accum_loc_loss, self._accum_loc_steps),
            ("prompt_rate_loss", self._accum_pr_loss, self._accum_pr_steps),
        ):
            if loss_steps > 0:
                # no division because the loss is not divided at the end, but scaled on the go
                losses[f"{loss_name}_fix"] = loss_value
        if do_print:
            for k, v in losses.items():
                log_debug(f"Accum loss: {k}: {v:.4f}")

        # Reset accumulation buffers
        self.accum_batch_list = []
        self.accum_output_dict = {}
        self._accum_loc_loss = 0.0
        self._accum_loc_steps = 0
        self._accum_pr_loss = 0.0
        self._accum_pr_steps = 0

        return losses

    #################### prompt rate ####################

    def _encode_prompt_rate_embeddings(self) -> torch.Tensor:
        """
        Encode prompts through text model.

        Returns:
            emb shape (n_prompts, n_variants, 2, D) where 2 is for pos/neg.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        all_texts = []
        n_prompts = len(self.prompt_rate_pos_prompts)
        n_variants = 3
        for pos_list, neg_list in zip(
            self.prompt_rate_pos_prompts, self.prompt_rate_neg_prompts, strict=True
        ):
            assert len(pos_list) == len(neg_list) == n_variants
            for pos_prompt, neg_prompt in zip(pos_list, neg_list, strict=True):
                all_texts += [pos_prompt, neg_prompt]
        # all_text is now length n_prompts * n_variants * 2
        tok = self.prompt_rate_tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(self.accelerator.device)
        attention_mask = tok["attention_mask"].to(self.accelerator.device)
        out = unwrapped.backbone_text(input_ids=input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(out.last_hidden_state, attention_mask)
        emb = unwrapped.projection_text(pooled)
        if unwrapped.criterion.normalize:
            emb = F.normalize(emb, dim=-1)
        emb_shaped = emb.reshape(n_prompts, n_variants, 2, -1)
        return emb_shaped

    def _compute_prompt_rate_loss(
        self,
        batch: dict,
        image_embeddings: torch.Tensor,
        prompt_emb: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute prompt-rate BCE loss for a batch. Returns weighted loss or None."""
        pr_labels = self._get_prompt_rate_labels(batch)
        if pr_labels is None:
            return None
        unwrapped = self.accelerator.unwrap_model(self.model)
        img_emb = image_embeddings
        if unwrapped.criterion.normalize:
            img_emb = F.normalize(img_emb, dim=-1)
        prompt_sampling_mode = self.train_config["train"]["prompt_rate"].get(
            "prompt_sampling_mode", PromptSamplingC.FIRST_ONLY
        )
        return self.prompt_rate_loss(
            img_emb,
            prompt_emb,
            pr_labels,
            unwrapped.criterion.t,
            unwrapped.criterion.b,
            prompt_sampling_mode,
        )

    def _get_prompt_rate_labels(self, batch: dict) -> torch.Tensor | None:
        """Look up RaTE labels for scan_keys in batch. Returns (B, Q) or None."""
        scan_keys = batch.get("scan_key")
        if scan_keys is None:
            return None
        device = self.accelerator.device
        num_q = self.prompt_rate_loss.pos_weight_per_q.shape[0]
        labels = []
        for sk in scan_keys:
            lbl = self.prompt_rate_labels.get(sk)
            if lbl is not None:
                labels.append(lbl)
            else:
                # only 2 / ~78000 are missing labels, ignore
                labels.append(torch.full((num_q,), -1, dtype=torch.long))
        return torch.stack(labels).to(device)

    #################### phase 3: prompt-rate with mega-batches ####################

    def _run_phase3_prompt_rate(self, acc_steps_phase1: int, do_print: bool) -> None:
        prompt_rate_cfg = self.train_config["train"]["prompt_rate"]
        assert prompt_rate_cfg.get("text_grad", False), "Phase 3 requires text_grad=True"
        batch_size_p1 = self.train_config["train"]["batch_size"]
        phase3_B = prompt_rate_cfg["phase3_batch_size"]
        total_samples = acc_steps_phase1 * batch_size_p1
        if phase3_B > total_samples:
            # global batch size is smaller than the desired phase 3 batch size, so just use that
            phase3_B = total_samples
        phase3_accum_steps = total_samples // phase3_B
        assert (
            total_samples % phase3_B == 0
        ), f"Phase 3: {total_samples} not divisible by phase3_batch_size={phase3_B}"
        micro_per_mega = phase3_B // batch_size_p1
        pr_weight = prompt_rate_cfg.get("weight", 1.0)
        phase3_pr_correction = phase3_B / _PROMPT_RATE_REF_BATCH_SIZE

        if do_print:
            log_debug(
                f"Phase 3: {phase3_accum_steps} mega-batches of {phase3_B} "
                f"(pr_correction={phase3_pr_correction:.4f})"
            )

        for m_idx in range(phase3_accum_steps):
            start = m_idx * micro_per_mega
            end = start + micro_per_mega
            mega = self._merge_micro_batches(self.accum_batch_list[start:end])

            ctx = (
                self.accelerator.no_sync(self.model)
                if m_idx < phase3_accum_steps - 1
                else nullcontext()
            )
            with ctx, self.accelerator.autocast():
                o = self.model.forward_image_only(mega)

                prompt_emb = self._encode_prompt_rate_embeddings()
                pr_loss = self._compute_prompt_rate_loss(
                    mega, o.image_embeddings_secondary, prompt_emb
                )
                if pr_loss is not None:
                    scaled_loss = pr_loss * pr_weight * phase3_pr_correction
                    self._accum_pr_loss += scaled_loss.detach().item()
                    self._accum_pr_steps += 1
                    self.accelerator.backward(scaled_loss)

    def _merge_micro_batches(self, micro_list: list[dict]) -> dict:
        all_grids = torch.cat([m["image_grid_shape"] for m in micro_list], dim=0)
        Hp_max = all_grids[:, 0].max().item()
        Wp_max = all_grids[:, 1].max().item()
        Dp_max = all_grids[:, 2].max().item()
        total_B = all_grids.shape[0]

        ref = micro_list[0]["image_backbone_cls"]
        E = ref.shape[-1]
        device = ref.device
        dtype = ref.dtype

        out_cls = torch.zeros(total_B, Hp_max, Wp_max, Dp_max, E, device=device, dtype=dtype)
        out_pavg = torch.zeros_like(out_cls)
        out_mask = torch.zeros(total_B, Hp_max, Wp_max, Dp_max, device=device, dtype=torch.bool)

        all_scan_keys = []
        offset = 0
        for m in micro_list:
            cls = m["image_backbone_cls"]
            B_m, Hp_m, Wp_m, Dp_m, _ = cls.shape
            out_cls[offset : offset + B_m, :Hp_m, :Wp_m, :Dp_m, :] = cls
            out_pavg[offset : offset + B_m, :Hp_m, :Wp_m, :Dp_m, :] = m[
                "image_backbone_patch_average"
            ]
            out_mask[offset : offset + B_m, :Hp_m, :Wp_m, :Dp_m] = m["window_mask"]
            all_scan_keys.extend(m["scan_key"])
            offset += B_m

        return {
            "image_backbone_cls": out_cls,
            "image_backbone_patch_average": out_pavg,
            "window_mask": out_mask,
            "image_grid_shape": all_grids,
            "scan_key": all_scan_keys,
        }

    #################### localization loss ####################

    def _compute_localization_loss(self, output: LocalizationOutput) -> torch.Tensor | None:
        """Compute weighted localization loss from output fields. Returns None if no loc data."""
        if output.snippet_emb is None:
            return None
        loc_weight = self._get_effective_loc_weight()
        if loc_weight == 0.0:
            return None
        for field in (
            "scan_slice_emb",
            "scan_valid_depth_mask",
            "slice_target_depth_mask",
            "slice_batch_idx_valid",
        ):
            if output.get(field) is None:
                raise RuntimeError(f"Localization output missing {field}")
        # convert the vision features from scan-level to slice-level by repeating them
        # (e.g. 3 slices from the same scan, all need the scan inputs for the localization loss)
        slice_emb = output.scan_slice_emb[output.slice_batch_idx_valid]
        slice_valid_depth_mask = output.scan_valid_depth_mask[output.slice_batch_idx_valid]
        loc_loss = self.model.loc_criterion(
            slice_emb,
            output.snippet_emb,
            output.slice_target_depth_mask,
            slice_valid_depth_mask,
        )
        return loc_weight * loc_loss

    def _get_effective_loc_weight(self) -> float:
        """
        Compute effective localization weight with optional skip and linear warmup.

        Config fields (under do_snippet_alignment):
            localization_weight: base weight (default 0.3)
            localization_skip_epochs: epochs with zero loc loss (default 0)
            localization_warmup_epochs: epochs to linearly ramp from 0 to base weight (default 0)

        Returns 0 during skip, linear ramp during warmup, constant afterwards.
        """
        snippet_cfg = self.model.do_snippet_alignment
        base_weight = snippet_cfg.get("localization_weight", 0.0)
        skip_epochs = snippet_cfg.get("localization_skip_epochs", 0)
        warmup_epochs = snippet_cfg.get("localization_warmup_epochs", 0)

        epoch = self.global_epoch

        if epoch < skip_epochs:
            return 0.0

        if warmup_epochs > 0 and epoch < skip_epochs + warmup_epochs:
            warmup_start_step = skip_epochs * self.steps_per_epoch
            warmup_total_steps = warmup_epochs * self.steps_per_epoch
            steps_into_warmup = self.global_step - warmup_start_step
            progress = min(steps_into_warmup / max(warmup_total_steps, 1), 1.0)
            return base_weight * progress

        return base_weight

    #################### validation, checkpoint ####################

    def does_epoch_need_validating(self, n_epochs_trained: int) -> float | None:
        """Check if the epoch needs validating."""
        if not self.accelerator.is_main_process or not self.val_dict:
            return False
        for task_name, (_task_config, _, _) in self.val_dict.items():
            if not self.get_val_output_file(n_epochs_trained, task_name).is_file():
                return True
            if self.bootstrap:
                task_type = _task_config.get("task_type", "retrieval")
                if task_type in (
                    "retrieval",
                    "pool_retrieval",
                    "volume_retrieval",
                    "binary_zs",
                    "localization",
                ):
                    if not self.get_bootstrap_output_file(n_epochs_trained, task_name).is_file():
                        return True
        return False

    def get_val_output_file(self, n_epochs_trained: int, task_name: str) -> Path:
        val_output_dir = Path(self.train_config["train"]["output_dir"]) / "val_output"
        return val_output_dir / f"epoch_{n_epochs_trained:04d}_{task_name}.json"

    def get_bootstrap_output_file(self, n_epochs_trained: int, task_name: str) -> Path:
        val_output_dir = Path(self.train_config["train"]["output_dir"]) / "val_output"
        return val_output_dir / f"epoch_{n_epochs_trained:04d}_{task_name}_bootstrap.json"

    def validate_epoch(self, n_epochs_trained: int, accelerator_log: bool = True) -> float | None:
        """Run validation and return meanr for checkpoint naming."""
        if not self.accelerator.is_main_process or not self.val_dict:
            return None
        log_info(f"Running validation for epoch {n_epochs_trained}...")

        self.model.eval()
        # Keep full snippet config during eval — forward() naturally no-ops when
        # eval batches lack slices/axis2 features, no need to disable anything

        val_meanr = None
        for task_name, (_task_config, dataloader, dataset) in self.val_dict.items():
            log_info(f"Evaluating =====> {task_name} <=====")
            task_type = _task_config["task_type"]
            output_file = self.get_val_output_file(n_epochs_trained, task_name)
            os.makedirs(output_file.parent, exist_ok=True)
            bootstrap_file = self.get_bootstrap_output_file(n_epochs_trained, task_name)
            need_bootstrap = (
                self.bootstrap
                and task_type in BOOTSTRAPPABLE_TASK_TYPES
                and not bootstrap_file.is_file()
            )
            if output_file.is_file() and not need_bootstrap:
                log_info(f"  {output_file} already exists, skipping")
                all_metrics = load_json(output_file)
                log_dict = None
            else:
                if need_bootstrap:
                    assert (
                        output_file.is_file()
                    ), f"Output file {output_file} must exist for bootstrap validation"
                with torch.inference_mode(), self.accelerator.autocast():
                    all_metrics, aux_metrics, bootstrap_metrics = run_task_by_type(
                        task_config=_task_config,
                        model=self.accelerator.unwrap_model(self.model),
                        dataloader=dataloader,
                        dataset=dataset,
                        model_config=self.model_config,
                        device=self.accelerator.device,
                        bootstrap=need_bootstrap,
                        verbose=False,
                    )
                if need_bootstrap:
                    _assert_metrics_match(load_json(output_file), all_metrics, task_name)
                    dump_json(bootstrap_metrics, bootstrap_file, indent=2, verbose=False)
                    log_info(f"  Saved bootstrap CIs to {bootstrap_file}")
                else:
                    _save_main, _save_aux = _save_kwargs_for_task(task_type)
                    dump_json(all_metrics, output_file, indent=2, verbose=False, **_save_main)
                    if aux_metrics is not None and _aux_should_be_saved(
                        task_type, self.train_config
                    ):
                        aux_file = output_file.with_name(output_file.stem + "_aux.json")
                        dump_json(aux_metrics, aux_file, indent=2, verbose=False, **_save_aux)
                log_dict = _log_dict_for_task(task_name, task_type, all_metrics)

            if accelerator_log and log_dict is not None:
                self.accelerator.log(log_dict, step=self.global_step)

            # Print task-specific summary
            if task_type == "binary_zs":
                log_info(
                    f"    auc: {all_metrics['mean_auroc']:.4f}  "
                    f"prec: {all_metrics['mean_prec']:.4f}  "
                    f"f1: {all_metrics['mean_f1']:.4f}  "
                    f"acc: {all_metrics['mean_acc']:.4f}"
                )
            elif task_type == "binary_zs_rate":
                n_eval = all_metrics.get("n_questions_evaluated", 0)
                n_skip = all_metrics.get("n_questions_skipped", 0)
                log_info(
                    f"    auc: {all_metrics['mean_auroc']:.4f}  "
                    f"prec: {all_metrics.get('mean_precision', 0):.4f}  "
                    f"f1w: {all_metrics.get('mean_f1_w', 0):.4f}  "
                    f"acc: {all_metrics.get('mean_acc', 0):.4f}  "
                    f"({n_eval} questions, {n_skip} skipped)"
                )
            elif task_type == "localization":
                log_info(
                    f"    loc MAE: {all_metrics['loc_mae_mm']:.1f}mm, "
                    f"exact: {all_metrics['loc_acc_exact']*100:.1f}%, "
                    f"<=24mm: {all_metrics['loc_acc_within_24mm']*100:.1f}%"
                )
            elif task_type == "volume_retrieval":
                log_info(
                    f"    vol MAP@5: {all_metrics['vol_map5']:.4f}, "
                    f"MAP@10: {all_metrics['vol_map10']:.4f}, "
                    f"MAP@50: {all_metrics['vol_map50']:.4f}, "
                    f"n_abnormal: {all_metrics['vol_n_abnormal']}"
                )
            elif task_type == "pool_retrieval":
                log_info(
                    f"    pool R@1: find={all_metrics.get('find_pool128_r1', 0):.4f} "
                    f"impr={all_metrics.get('impr_pool128_r1', 0):.4f} "
                    f"full={all_metrics.get('full_pool128_r1', 0):.4f}, "
                    f"n={all_metrics.get('n', 0)}"
                )
            else:
                log_info(
                    f"    t2i meanr: {all_metrics['t2i_meanr']:.2f}, "
                    f"t2i r@10: {all_metrics['t2i_r10']*100:.2f}%, "
                    f"loss_nonaccum: {all_metrics['loss_nonaccum']:.2f}"
                )

            if val_meanr is None:
                val_meanr = all_metrics.get("t2i_meanr", None)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val_meanr

    def save_checkpoint(self, n_epochs_trained: int, val_score_lower_better: float):
        """
        Save checkpoint at the end of epoch.

        Only saves trainable parameters (those with requires_grad=True) to avoid
        saving frozen backbone weights. This keeps checkpoint size small and allows
        loading into models with different frozen/pretrained components.
        """
        assert val_score_lower_better is not None, "val_score_lower_better must be provided"
        output_dir = Path(self.train_config["train"]["output_dir"])

        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Note: filtering to requires_grad only somehow breaks trained checkpoints.
        # Just save the full state dict and condense it later.
        checkpoint = {
            "epoch": n_epochs_trained,
            "global_step": self.global_step,
            "model": unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_config": self.train_config,
            "model_config": self.model_config,
        }

        torch.save(checkpoint, output_dir / "checkpoint.pt")
        log_info(f"Saved checkpoint to {output_dir / 'checkpoint.pt'}")

        # Save periodic checkpoints
        if n_epochs_trained % self.train_config["train"].get("saveckp_freq", 5) == 0:
            checkpoint_path = (
                output_dir / f"checkpoint_epoch_{n_epochs_trained:04d}_meanr_"
                f"{val_score_lower_better:.2f}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            log_info(f"Saved checkpoint to {checkpoint_path}")


def _assert_metrics_match(
    existing: dict, new: dict, task_name: str, rtol: float = 5e-2, atol: float = 5e-2
):
    for key in existing:
        if key not in new or not isinstance(existing[key], (int, float)):
            continue
        if existing[key] is None or new[key] is None:
            continue
        if key.startswith("loss"):
            continue
        if key.endswith("medr"):
            key_rtol, key_atol = 1e-1, 2.0
        elif key == "mean_sens":
            key_rtol, key_atol = 1e-1, 1e-1
        else:
            key_rtol, key_atol = rtol, atol
        ref = abs(existing[key]) if abs(existing[key]) > 1e-8 else 1.0
        err = abs(existing[key] - new[key])
        if err > key_rtol * ref and err > key_atol:
            raise AssertionError(
                f"Bootstrap cross-validation failed for {task_name}: "
                f"{key}={new[key]} vs existing={existing[key]} "
                f"(diff={abs(existing[key] - new[key]):.2e}, rtol={key_rtol}, atol={key_atol})"
            )


def _save_kwargs_for_task(task_type: str) -> tuple[dict, dict]:
    """Return `(dump_json_kwargs_main, dump_json_kwargs_aux)` for the given task."""
    # binary_zs main + aux both contain NaN cells that need the custom encoder.
    if task_type == "binary_zs":
        kw = {"custom_format_nan_to_none": True}
        return kw, kw
    return {}, {}


def _aux_should_be_saved(task_type: str, train_config: dict) -> bool:
    """Whether `aux_metrics` should be dumped to a separate `_aux.json` for this task."""
    if task_type in ("binary_zs", "binary_zs_rate"):
        return True
    if task_type == "localization":
        return train_config["train"].get("save_outputs", False)
    return False


def _log_dict_for_task(task_name: str, task_type: str, all_metrics: dict) -> dict:
    """Build the per-task wandb log_dict from the standard metrics."""
    if task_type == "retrieval":
        return {
            f"val/{task_name}/t2i_meanr": all_metrics["t2i_meanr"],
            f"val/{task_name}/t2i_r10": all_metrics["t2i_r10"],
            f"val/{task_name}/loss_nonaccum": all_metrics["loss_nonaccum"],
        }
    if task_type == "binary_zs":
        return {
            f"val/{task_name}/mean_auroc": all_metrics["mean_auroc"],
            f"val/{task_name}/mean_prec": all_metrics["mean_prec"],
            f"val/{task_name}/mean_f1": all_metrics["mean_f1"],
            f"val/{task_name}/mean_acc": all_metrics["mean_acc"],
        }
    if task_type == "binary_zs_rate":
        return {
            f"val/{task_name}/mean_auroc": all_metrics["mean_auroc"],
            f"val/{task_name}/mean_precision": all_metrics["mean_precision"],
            f"val/{task_name}/mean_f1_w": all_metrics["mean_f1_w"],
            f"val/{task_name}/mean_acc": all_metrics["mean_acc"],
        }
    if task_type == "localization":
        return {
            f"val/{task_name}/loc_mae_mm": all_metrics["loc_mae_mm"],
            f"val/{task_name}/loc_acc_within_24mm": all_metrics["loc_acc_within_24mm"],
        }
    if task_type == "volume_retrieval":
        return {
            f"val/{task_name}/vol_map5": all_metrics["vol_map5"],
            f"val/{task_name}/vol_map10": all_metrics["vol_map10"],
            f"val/{task_name}/vol_map50": all_metrics["vol_map50"],
        }
    if task_type == "pool_retrieval":
        return {f"val/{task_name}/full_pool128_r1": all_metrics.get("full_pool128_r1", 0)}
    raise ValueError(f"Unknown validation task type: {task_type}")
