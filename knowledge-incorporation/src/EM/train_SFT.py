"""
SFT trainer with LoRAMoe integration

Dataset format expected:
{"prompt": "...", "completion": "..."}
"""

import os
import argparse
from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig
from peft import get_peft_model
from loramae import LoRAMoeConfig, get_loramoe_model  # Make sure to implement get_loramoe_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--logging_steps", type=int, default=10)
    return p.parse_args()

def tokenize_batch(batch, tokenizer, max_length):
    texts = [ex["prompt"] + ex["completion"] for ex in batch]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    # Labels are input_ids shifted, here just use input_ids and mask padding tokens
    encodings["labels"] = encodings.input_ids.clone()
    # Optionally, mask padding tokens loss
    encodings["labels"][encodings["attention_mask"] == 0] = -100
    return encodings

class SFTTrainer:
    def __init__(self, model, args, train_dataset, tokenizer, peft_config=None):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.peft_config = peft_config

        if peft_config is not None:
            if isinstance(peft_config, LoRAMoeConfig):
                self.model = get_loramoe_model(self.model, peft_config).to(self.device)
            else:
                self.model = get_peft_model(self.model, peft_config).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

    def train(self):
        self.model.train()
        dataloader = DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, shuffle=True)
        global_step = 0
        for epoch in range(self.args.num_train_epochs):
            for batch in dataloader:
                batch = tokenize_batch(batch, self.tokenizer, max_length=self.args.max_length)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss

                # Add MoE auxiliary loss if exists
                if hasattr(self.model, "moe_loss"):
                    loss = loss + self.peft_config.moe_loss_coef * self.model.moe_loss()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if global_step % self.args.logging_steps == 0:
                    print(f"Epoch {epoch} Step {global_step} Loss: {loss.item():.4f}")
                global_step += 1

def longest_seq_len(dataset, tok):
    return max(
        len(tok(example["prompt"] + example["completion"]).input_ids)
        for example in dataset
    )

def main() -> None:
    args = parse_args()

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_length=longest_seq_len(dataset, tokenizer),
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    loramoe_cfg = LoRAMoeConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_experts=4,
        moe_loss_coef=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=loramoe_cfg,
    )

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    trainer.train()

    peft_model = trainer.model
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
