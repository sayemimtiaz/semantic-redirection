import torch
import torch.nn.functional as F
import copy

from trainer.unlearn.base import UnlearnTrainer
from sentence_transformers import SentenceTransformer


class SemanticRedirection(UnlearnTrainer):
    def __init__(
        self,
        gamma=1.0,
        alpha=1.0,
        ref_model_name='all-MiniLM-L6-v2',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.gamma = gamma
        self.alpha = alpha

        # Fixed external semantic encoder
        self.semantic_encoder = SentenceTransformer(ref_model_name)
        self.semantic_encoder = _prepare_ref_model(self.semantic_encoder)

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------

    @torch.no_grad()
    def _decode_labels(self, labels):
        """
        Decode ground-truth labels into text.
        """
        labels = labels.clone()
        labels[labels == -100] = self.tokenizer.pad_token_id
        return self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

    @torch.no_grad()
    def _decode_model(self, model, inputs):
        """
        Decode model response given inputs.
        """
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=inputs["labels"].size(1),
        )
        return self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    def _cosine_similarity(self, cur_text, gt_text):
        """
        Cosine similarity between semantic embeddings.
        """
        gt_emb = self.semantic_encoder.encode(
            gt_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        cur_emb = self.semantic_encoder.encode(
            cur_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        return F.cosine_similarity(cur_emb, gt_emb, dim=-1)

    # ------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------

    def compute_forget_loss(self, model, forget_inputs):
        """
        Forget loss:
        Minimize cosine similarity between model response
        and ground-truth forget response.
        """
        gt_text = self._decode_labels(forget_inputs["labels"])
        cur_text = self._decode_model(model, forget_inputs)

        forget_loss = self._cosine_similarity(cur_text, gt_text)
        return forget_loss.mean()

    def compute_retain_loss(self, model, retain_inputs):
        """
        Retain loss:
        Minimize semantic drift from ground-truth retain response.
        """
        gt_text = self._decode_labels(retain_inputs["labels"])
        cur_text = self._decode_model(model, retain_inputs)

        retain_loss = 1.0 - self._cosine_similarity(cur_text, gt_text)
        return retain_loss.mean()

    # ------------------------------------------------------------
    # Main training objective
    # ------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Total loss:
            gamma * forget_loss + alpha * retain_loss
        """

        # Forget batch
        forget_inputs = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }

        forget_loss = self.compute_forget_loss(
            model=model,
            forget_inputs=forget_inputs,
        )

        # Retain batch
        retain_inputs = {
            "input_ids": inputs["retain"]["input_ids"],
            "attention_mask": inputs["retain"]["attention_mask"],
            "labels": inputs["retain"]["labels"],
        }

        retain_loss = self.compute_retain_loss(
            model=model,
            retain_inputs=retain_inputs,
        )

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return loss
