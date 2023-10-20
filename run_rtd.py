from dataclasses import dataclass, field
import math
import os
import shutil
from typing import Optional

from datasets import load_from_disk, concatenate_datasets
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import get_scheduler
from transformers import (
    DebertaV2Config,
    DebertaV2ForMaskedLM,
    DebertaV2ForTokenClassification,
    HfArgumentParser,
)

import torch
from torch.utils.data import DataLoader

import srsly
from tqdm.auto import tqdm


@dataclass
class TrainArgs:
    generator_config: Optional[str] = field(
        default="deberta-v3-xsmall-changed/generator_config.json",
        metadata={"help": "Path to the generator config file."},
    )
    generator_weights: Optional[str] = field(
        default="deberta-v3-xsmall-changed/pytorch_model.generator.bin",
        metadata={"help": "Path to the generator weights file."},
    )
    discriminator_config: Optional[str] = field(
        default="deberta-v3-xsmall-changed/config.json",
        metadata={"help": "Path to the discriminator config file."},
    )
    discriminator_weights: Optional[str] = field(
        default="deberta-v3-xsmall-changed/pytorch_model.bin",
        metadata={"help": "Path to the discriminator weights file."},
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help": "Temperature for top-k sampling."}
    )
    rtd_lambda: Optional[float] = field(
        default=20.0, metadata={"help": "Lambda for RTD loss."}
    )
    tokenizer_name: Optional[str] = field(
        default="debertinha-v2-tokenizer",
        metadata={"help": "Tokenizer name or path"},
    )
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={"help": "Learning rate"}
    )
    mixed_precision: Optional[str] = field(
        default="no", metadata={"help": "Mixed precision training"}
    )
    weight_decay: Optional[float] = field(
        default=0.0, metadata={"help": "Weight decay"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    num_warmup_steps: Optional[int] = field(
        default=10_000, metadata={"help": "Number of warmup steps"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "LR scheduler type"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "Number of training epochs"}
    )
    cpu: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use CPU"}
    )
    log_with: Optional[str] = field(
        default="tensorboard", metadata={"help": "Logging framework"}
    )
    project_dir: Optional[str] = field(
        default="debertinha-v2-accelerate",
        metadata={"help": "Project directory"},
    )
    max_train_steps: Optional[int] = field(
        default=None, metadata={"help": "Max train steps"}
    )
    checkpointing_steps: Optional[int] = field(
        default=10, metadata={"help": "Checkpointing steps"}
    )
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": "Save total limit"}
    )
    max_grad_norm: Optional[float] = field(
        default=1.0, metadata={"help": "Max grad norm"}
    )
    dataset_paths: Optional[str] = field(
        default="brwac_encoded_firsthalf,brwac_encoded_secondhalf,mc4pt_subset_encoded",
        metadata={"help": "Path to the dataset"}
    )
    run_name: Optional[str] = field(
        default="debertinha-v2-runs",
        metadata={"help": "Name of the run"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the saved state to load"}
    )
    skip_batches: Optional[int] = field(
        default=0,
        metadata={"help": "number of steps to skip from the dataloader"}
    )


def get_train_dataloader(targs, tokenizer, dataset):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=targs.per_device_train_batch_size,
        num_workers=1,
    )
    return train_dataloader


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


def _set_param(module, param_name, value):
    if hasattr(module, param_name):
        delattr(module, param_name)
    module.register_buffer(param_name, value)


def disentangled_hook(module, *inputs):
    g_w_ebd = generator.deberta.embeddings.word_embeddings
    d_w_ebd = discriminator.deberta.embeddings.word_embeddings
    _set_param(d_w_ebd, "weight", g_w_ebd.weight.detach() + d_w_ebd.weight)


def get_optimizer_and_scheduler(model, targs):
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


def topk_sampling(logits, topk=1, temp=1):
    top_p = torch.nn.functional.softmax(logits / temp, dim=-1)
    topk = max(1, topk)
    next_tokens = torch.multinomial(top_p, topk)
    return next_tokens, top_p


def generator_step(generator, generator_optimizer, generator_lr_scheduler, accelerator, batch, targs):
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

    return gen_loss.cpu().detach().float(), gen_outputs.logits

def discriminator_batch(gen_logits, batch, targs):
    mlm_labels = batch.pop("labels")
    input_ids = batch.pop("input_ids")

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
    new_ids = torch.where(
        mlm_labels > 0, top_ids, input_ids
    ).detach()
    return {
        "input_ids": new_ids,
        "labels": mlm_labels,
        "attention_mask": batch.pop("attention_mask"),
    }

def discriminator_step(discriminator, discriminator_optimizer, discriminator_lr_scheduler, accelerator, batch, targs):
    mlm_labels = batch.pop("labels")
    disc_outputs = discriminator(**batch)

    mask_logits = disc_outputs.logits.view(-1)
    _input_mask = batch.pop("attention_mask")
    _input_mask = _input_mask.view(-1).to(mask_logits)
    input_ids = batch.pop("input_ids")

    input_idx = (_input_mask > 0).nonzero().view(-1)
    mask_labels = (
        (mlm_labels > 0) & (mlm_labels != input_ids)
    ).view(-1)
    mask_labels = torch.gather(
        mask_labels.to(mask_logits), 0, input_idx
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

    return disc_loss.cpu().detach().float()

if __name__ == "__main__":
    parser = HfArgumentParser(TrainArgs)
    targs = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(
        mixed_precision=targs.mixed_precision,
        gradient_accumulation_steps=targs.gradient_accumulation_steps,
        cpu=targs.cpu,
        log_with=targs.log_with,
        project_dir=targs.project_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(targs.tokenizer_name)

    dataset = concatenate_datasets([load_from_disk(dspath) for dspath in targs.dataset_paths.split(",")])

    train_loader = get_train_dataloader(targs, tokenizer, dataset)

    discriminator = initialize_discriminator(targs)
    generator = initialize_generator(targs)

    discriminator.register_forward_pre_hook(disentangled_hook)

    generator_optimizer, generator_lr_scheduler = get_optimizer_and_scheduler(
        generator, targs
    )

    (
        discriminator_optimizer,
        discriminator_lr_scheduler,
    ) = get_optimizer_and_scheduler(discriminator, targs)

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
    experiment_config = vars(targs)
    accelerator.init_trackers(targs.run_name, experiment_config)

    if targs.resume_from_checkpoint is not None:
        accelerator.load_state(targs.resume_from_checkpoint)
        train_loader = accelerator.skip_first_batches(train_loader, targs.skip_batches)
        print("LOADED STATE")

    progress_bar = tqdm(
        range(targs.max_train_steps),
        disable=not accelerator.is_local_main_process,
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
                ## GENERATOR STEP
                gen_loss, gen_logits = generator_step(generator, generator_optimizer, generator_lr_scheduler, accelerator, batch, targs)
                total_generator_loss += gen_loss
                ## GENERATOR STEP

                ## DISCRIMINATOR BATCH
                batch = discriminator_batch(gen_logits, batch, targs)
                del gen_logits
                ## DISCRIMINATOR BATCH

                ## DISCRIMINATOR STEP
                disc_loss = discriminator_step(discriminator, discriminator_optimizer, discriminator_lr_scheduler, accelerator, batch, targs)
                total_discriminator_loss += disc_loss
                ## DISCRIMINATOR STEP

                total_loss += (gen_loss + disc_loss)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % targs.checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                output_dir = os.path.join(targs.project_dir, output_dir)
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
        output_dir = os.path.join(targs.project_dir, output_dir)
        accelerator.save_state(output_dir)

    accelerator.end_training()

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(generator)
    unwrapped_model.save_pretrained(
        os.path.join(targs.project_dir, "generator_pretrained"),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    unwrapped_model = accelerator.unwrap_model(discriminator)
    unwrapped_model.save_pretrained(
        os.path.join(targs.project_dir, "discriminator_pretrained"),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
