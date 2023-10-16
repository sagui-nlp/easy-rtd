from dataclasses import dataclass

import srsly
import torch

from transformers import (
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2ForTokenClassification,
)


from args import targs, ModelPaths, TrainArgs


def initialize_generator(mpaths: ModelPaths) -> DebertaV2ForMaskedLM:
    generator_config = DebertaV2Config(
        **srsly.read_json(mpaths.generator_config)
    )
    generator = DebertaV2ForMaskedLM(generator_config)

    generator_weights = torch.load(
        mpaths.generator_weights, map_location=torch.device("cpu")
    )

    delete_keys = [
        "deberta.embeddings.word_embeddings.weight",  # because we use a different vocab
        "deberta.embeddings.position_embeddings.weight",
        "lm_predictions.lm_head.bias",
    ]
    for key in delete_keys:
        del generator_weights[key]

    rename_keys = {
        "lm_predictions.lm_head.dense.weight": "cls.predictions.transform.dense.weight",
        "lm_predictions.lm_head.dense.bias": "cls.predictions.transform.dense.bias",
        "lm_predictions.lm_head.LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
        "lm_predictions.lm_head.LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
    }
    for old_key, new_key in rename_keys.items():
        generator_weights[new_key] = generator_weights.pop(old_key)

    print(generator.load_state_dict(generator_weights, strict=False))

    return generator


def initialize_discriminator(
    mpaths: ModelPaths,
) -> DebertaV2ForTokenClassification:
    discriminator_config = DebertaV2Config(
        **srsly.read_json(mpaths.discriminator_config)
    )
    discriminator_config.num_labels = 1
    discriminator = DebertaV2ForTokenClassification(discriminator_config)

    discriminator_weights = torch.load(
        mpaths.discriminator_weights, map_location=torch.device("cpu")
    )

    delete_keys = [
        "deberta.embeddings.word_embeddings.weight",  # because we use a different vocab
    ]
    for key in delete_keys:
        del discriminator_weights[key]

    print(discriminator.load_state_dict(discriminator_weights, strict=False))

    return discriminator


class DeBERTinhaV2:
    targs: TrainArgs = targs

    def __post_init__(self):
        self.generator = initialize_generator(self.targs)
        self.discriminator = initialize_discriminator(self.targs)
        self.discriminator.register_forward_pre_hook()

    @staticmethod
    def _set_param(module, param_name, value):
        if hasattr(module, param_name):
            delattr(module, param_name)
            module.register_buffer(param_name, value)

    def register_discriminator_fw_hook(self, *wargs):
        def fw_hook(module, *inputs):
            g_w_ebd = self.generator.deberta.embeddings.word_embeddings
            d_w_ebd = self.discriminator.deberta.embeddings.word_embeddings
            self._set_param(
                d_w_ebd, "weight", g_w_ebd.weight.detach() + d_w_ebd.weight
            )
            return None

        self.discriminator.register_forward_pre_hook(fw_hook)

    @staticmethod
    def topk_sampling(logits, topk=1, temp=1):
        top_p = torch.nn.functional.softmax(logits / temp, dim=-1)
        topk = max(1, topk)
        next_tokens = torch.multinomial(top_p, topk)
        return next_tokens, top_p

    def forward_generator(self, batch):
        """Output already contains loss"""
        return self.generator(**batch)

    def forward_discriminator(self, batch):
        return self.discriminator(**batch)

    def new_token_ids_from_generator(self, logits, mlm_labels, input_ids):
        logits = logits.view(-1, logits.size(-1))
        topk_labels, _ = self.topk_sampling(
            logits, topk=1, temp=targs.temperature
        )
        mask_index = (mlm_labels.view(-1) > 0).nonzero().view(-1)
        top_ids = torch.zeros_like(mlm_labels.view(-1))
        top_ids.scatter_(
            index=mask_index.long(), src=topk_labels.view(-1).long(), dim=-1
        )
        top_ids = top_ids.view(mlm_labels.size())
        new_ids = torch.where(mlm_labels > 0, top_ids, input_ids)
        return new_ids.detach()

    def get_discriminator_logits_and_labels(self, batch):
        mlm_labels = batch["labels"].pop()
        disc_outputs = self.forward_discriminator(batch)
        disc_logits = disc_outputs.logits
        mask_logits = disc_logits.view(-1)

        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]

        _input_mask = attention_mask.view(-1).to(mask_logits)
        input_idx = (_input_mask > 0).nonzero().view(-1)
        mask_labels = ((mlm_labels > 0) & (mlm_labels != input_ids)).view(-1)
        mask_labels = torch.gather(mask_labels.to(mask_logits), 0, input_idx)
        mask_logits = torch.gather(mask_logits, 0, input_idx).float()
        return mask_logits, mask_labels
