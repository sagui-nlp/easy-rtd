import math
import os
import shutil

from datasets import load_from_disk
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler
from transformers import (
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2ForTokenClassification,
)

import torch
from torch.utils.data import DataLoader

import srsly
from tqdm.auto import tqdm


class TrainArgs:
    generator_config = "deberta-v3-xsmall-changed/generator_config.json"
    generator_weights = "deberta-v3-xsmall-changed/pytorch_model.generator.bin"
    discriminator_config = "deberta-v3-xsmall-changed/config.json"
    discriminator_weights = "deberta-v3-xsmall-changed/pytorch_model.bin"
    per_device_train_batch_size: int = 1
    temperature: float = 1.0
    rtd_lambda: float = 20.0
    tokenizer_name: str = "debertinha-v2-tokenizer"
    learning_rate: float = 5e-5
    mixed_precision: str = "no"
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    num_warmup_steps: int = 10_000
    lr_scheduler_type: str = "linear"
    num_train_epochs: int = 1
    cpu: bool = False
    log_with: str = "tensorboard"
    project_dir: str = "debertinha-v2-accelerate"
    max_train_steps: int = None
    checkpointing_steps: int = 10
    output_dir: str = "debertinha-v2-checkpoints"
    save_total_limit: int = 1
    max_grad_norm: float = 1.0
    dataset_path: str = "ds_subset_encoded"


targs = TrainArgs()

accelerator = Accelerator(
    mixed_precision=targs.mixed_precision,
    gradient_accumulation_steps=targs.gradient_accumulation_steps,
    cpu=targs.cpu,
    log_with=targs.log_with,
    project_dir=targs.project_dir,
)

tokenizer = AutoTokenizer.from_pretrained(targs.tokenizer_name)


def get_train_dataloader(targs, tokenizer, dataset):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=targs.per_device_train_batch_size,
        num_workers=os.cpu_count(),
    )
    return train_dataloader


dataset = load_from_disk(targs.dataset_path)
dataset = dataset.select(range(100))

train_loader = get_train_dataloader(targs, tokenizer, dataset)


def initialize_generator(targs) -> DebertaV2ForMaskedLM:
    generator_config = DebertaV2Config(
        **srsly.read_json(targs.generator_config)
    )
    generator = DebertaV2ForMaskedLM(generator_config)

    generator_weights = torch.load(
        targs.generator_weights, map_location=torch.device("cpu")
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
    targs,
) -> DebertaV2ForTokenClassification:
    discriminator_config = DebertaV2Config(
        **srsly.read_json(targs.discriminator_config)
    )
    discriminator_config.num_labels = 1
    discriminator = DebertaV2ForTokenClassification(discriminator_config)

    discriminator_weights = torch.load(
        targs.discriminator_weights, map_location=torch.device("cpu")
    )

    delete_keys = [
        "deberta.embeddings.word_embeddings.weight",  # because we use a different vocab
    ]
    for key in delete_keys:
        del discriminator_weights[key]

    print(discriminator.load_state_dict(discriminator_weights, strict=False))

    return discriminator


discriminator = initialize_discriminator(targs)
generator = initialize_generator(targs)


def _set_param(module, param_name, value):
    if hasattr(module, param_name):
        delattr(module, param_name)
    module.register_buffer(param_name, value)


def disentangled_hook(module, *inputs):
    g_w_ebd = generator.deberta.embeddings.word_embeddings
    d_w_ebd = discriminator.deberta.embeddings.word_embeddings
    _set_param(d_w_ebd, "weight", g_w_ebd.weight.detach() + d_w_ebd.weight)


discriminator.register_forward_pre_hook(disentangled_hook)


def get_optimizer_and_scheduler(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": targs.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=targs.learning_rate
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / targs.gradient_accumulation_steps
    )
    targs.max_train_steps = targs.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=targs.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=targs.num_warmup_steps
        * targs.gradient_accumulation_steps,
        num_training_steps=targs.max_train_steps
        * targs.gradient_accumulation_steps,
    )
    return optimizer, lr_scheduler


generator_optimizer, generator_lr_scheduler = get_optimizer_and_scheduler(
    generator
)
(
    discriminator_optimizer,
    discriminator_lr_scheduler,
) = get_optimizer_and_scheduler(discriminator)

(
    generator,
    generator_optimizer,
    generator_lr_scheduler,
    discriminator,
    discriminator_optimizer,
    discriminator_lr_scheduler,
    train_loader,
) = accelerator.prepare(
    generator,
    generator_optimizer,
    generator_lr_scheduler,
    discriminator,
    discriminator_optimizer,
    discriminator_lr_scheduler,
    train_loader,
)

num_update_steps_per_epoch = math.ceil(
    len(train_loader) / targs.gradient_accumulation_steps
)
targs.num_train_epochs = math.ceil(
    targs.max_train_steps / num_update_steps_per_epoch
)
experiment_config = {
    "per_device_train_batch_size": targs.per_device_train_batch_size,
    "temperature": targs.temperature,
    "rtd_lambda": targs.rtd_lambda,
    "tokenizer_name": targs.tokenizer_name,
    "learning_rate": targs.learning_rate,
    "mixed_precision": targs.mixed_precision,
    "weight_decay": targs.weight_decay,
    "gradient_accumulation_steps": targs.gradient_accumulation_steps,
    "num_warmup_steps": targs.num_warmup_steps,
    "lr_scheduler_type": targs.lr_scheduler_type,
    "num_train_epochs": targs.num_train_epochs,
    "cpu": targs.cpu,
    "log_with": targs.log_with,
    "project_dir": targs.project_dir,
    "max_train_steps": targs.max_train_steps,
    "checkpointing_steps": targs.checkpointing_steps,
    "output_dir": targs.output_dir,
    "save_total_limit": targs.save_total_limit,
    "max_grad_norm": targs.max_grad_norm,
    "dataset_path": targs.dataset_path,
}
accelerator.init_trackers("mlm_no_trainer", experiment_config)


def topk_sampling(logits, topk=1, temp=1):
    top_p = torch.nn.functional.softmax(logits / temp, dim=-1)
    topk = max(1, topk)
    next_tokens = torch.multinomial(top_p, topk)
    return next_tokens, top_p


progress_bar = tqdm(
    range(targs.max_train_steps), disable=not accelerator.is_local_main_process
)

completed_steps = 0
saved_states = []
for epoch in range(0, targs.num_train_epochs):
    generator.train()
    discriminator.train()

    total_generator_loss = 0
    total_discriminator_loss = 0
    total_loss = 0
    active_dataloader = train_loader

    for step, batch in enumerate(active_dataloader):
        with accelerator.accumulate(generator, discriminator):
            mlm_labels = batch["labels"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            ## GENERATOR STEP
            gen_outputs = generator(**batch)
            gen_loss = gen_outputs.loss
            accelerator.backward(gen_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    generator.parameters(), targs.max_grad_norm
                )

            generator_optimizer.step()
            generator_lr_scheduler.step()
            generator_optimizer.zero_grad()

            total_generator_loss += gen_loss.detach().float()
            ## GENERATOR STEP

            ## DISCRIMINATOR BATCH
            gen_logits = gen_outputs.logits
            gen_logits = gen_logits.view(-1, gen_logits.size(-1))
            topk_labels, _ = topk_sampling(
                gen_logits, topk=1, temp=targs.temperature
            )
            mask_index = (mlm_labels.view(-1) > 0).nonzero().view(-1)
            top_ids = torch.zeros_like(mlm_labels.view(-1))
            top_ids.scatter_(
                index=mask_index.long(),
                src=topk_labels.view(-1).long(),
                dim=-1,
            )
            top_ids = top_ids.view(mlm_labels.size())
            new_ids = torch.where(mlm_labels > 0, top_ids, input_ids).detach()
            disc_batch = {
                "input_ids": new_ids,
                "attention_mask": attention_mask,
            }
            ## DISCRIMINATOR BATCH

            ## DISCRIMINATOR STEP
            disc_outputs = discriminator(**disc_batch)
            disc_logits = disc_outputs.logits
            mask_logits = disc_logits.view(-1)
            _input_mask = attention_mask.view(-1).to(accelerator.device)
            input_idx = (_input_mask > 0).nonzero().view(-1)
            mask_labels = ((mlm_labels > 0) & (mlm_labels != input_ids)).view(
                -1
            )
            mask_labels = torch.gather(
                mask_labels.to(accelerator.device), 0, input_idx
            )
            mask_loss_fn = torch.nn.BCEWithLogitsLoss()
            mask_logits = torch.gather(mask_logits, 0, input_idx).float()
            disc_loss = targs.rtd_lambda * mask_loss_fn(
                mask_logits, mask_labels
            )
            accelerator.backward(disc_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    discriminator.parameters(), targs.max_grad_norm
                )
            discriminator_optimizer.step()
            discriminator_lr_scheduler.step()
            discriminator_optimizer.zero_grad()

            total_discriminator_loss += disc_loss.detach().float()
            ## DISCRIMINATOR STEP

            total_loss += (gen_loss + disc_loss).detach().float()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if completed_steps % targs.checkpointing_steps == 0:
            output_dir = f"step_{completed_steps}"
            if targs.output_dir is not None:
                output_dir = os.path.join(targs.output_dir, output_dir)
            accelerator.save_state(output_dir)
            saved_states.append(output_dir)

            # remove old states directory
            if len(saved_states) > targs.save_total_limit:
                old_state = saved_states.pop(0)
                shutil.rmtree(old_state)

        if completed_steps % 100 == 0:
            accelerator.log(
                {
                    "train_loss": total_loss.item() / 100,
                    "discriminator_loss": total_discriminator_loss.item()
                    / 100,
                    "generator_loss": total_generator_loss.item() / 100,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
            total_generator_loss = 0
            total_discriminator_loss = 0
            total_loss = 0

        if completed_steps >= targs.max_train_steps:
            break

    output_dir = f"epoch_{epoch}"
    if targs.output_dir is not None:
        output_dir = os.path.join(targs.output_dir, output_dir)
    accelerator.save_state(output_dir)

accelerator.end_training()

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(generator)
unwrapped_model.save_pretrained(
    "generator_final",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)
unwrapped_model = accelerator.unwrap_model(discriminator)
unwrapped_model.save_pretrained(
    "discriminator_final",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)
