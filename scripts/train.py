#!/usr/bin/env python3
"""
Fine-tuning script for Ecommerce Language Model (ELM)
Uses LoRA/QLoRA to fine-tune Llama-3.1-8B-Instruct for ecommerce tasks
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import yaml

# Load configuration
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load and format training data
def load_dataset(file_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
    """Load JSONL dataset and format for training"""
    print(f"📂 Loading dataset from {file_path}...")

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)

    print(f"   Loaded {len(data)} samples")

    # Format as conversations
    def format_sample(sample):
        # Create chat format similar to Llama-3.1-Instruct
        messages = [
            {"role": "system", "content": "You are ELM, an expert ecommerce AI assistant specialized in product recommendations, customer support, and sales."},
            {"role": "user", "content": sample["instruction"] + ("\n\n" + sample["input"] if "input" in sample else "")},
            {"role": "assistant", "content": sample["output"]}
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )

        # Add labels (for training)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Process all samples
    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    return dataset

# Model setup
def setup_model_and_tokenizer(model_name: str, config: dict):
    """Initialize model, tokenizer, and LoRA configuration"""
    print(f"🤖 Setting up model: {model_name}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"   ✓ Loaded tokenizer")
    print(f"   - Vocab size: {tokenizer.vocab_size}")
    print(f"   - Model max length: {tokenizer.model_max_length}")

    # Quantization configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config["quantization"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["quantization"]["bnb_4bit_use_double_quant"],
    )

    # Load model
    print(f"\n💾 Loading model with quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        use_cache=False  # Required for gradient checkpointing
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    print(f"   ✓ Loaded and prepared model")
    print(f"   - Model dtype: {model.dtype}")
    print(f"   - Device map: {model.hf_device_map}")

    # LoRA configuration
    print(f"\n⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias="none",
        inference_mode=False
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"   ✓ LoRA configured")
    print(f"   - Rank (r): {config['lora']['r']}")
    print(f"   - Alpha: {config['lora']['lora_alpha']}")
    print(f"   - Dropout: {config['lora']['lora_dropout']}")
    print(f"   - Target modules: {', '.join(config['lora']['target_modules'][:3])}...")

    return model, tokenizer

# Training arguments
def create_training_args(config: dict, output_dir: str):
    """Create training arguments"""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        adam_beta1=config["training"]["adam_beta1"],
        adam_beta2=config["training"]["adam_beta2"],
        adam_epsilon=config["training"]["adam_epsilon"],
        max_grad_norm=config["training"]["max_grad_norm"],
        logging_steps=config["logging"]["logging_steps"],
        eval_steps=config["logging"]["eval_steps"],
        save_steps=config["logging"]["save_steps"],
        evaluation_strategy=config["logging"]["eval_strategy"],
        save_strategy=config["logging"]["save_strategy"],
        load_best_model_at_end=config["logging"]["load_best_model_at_end"],
        metric_for_best_model=config["logging"]["metric_for_best_model"],
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        run_name=f"elm-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        group_by_length=True,
        length_column_name="length",
    )

# Main training function
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.1-8B for Ecommerce")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../config/training_config.yaml",
        help="Training configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../outputs/elm-llama-3b",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/processed",
        help="Directory containing training data"
    )

    args = parser.parse_args()

    # Load configuration
    print("📋 Loading training configuration...")
    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    print(f"\n🎯 Training Configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Epochs: {config['training']['num_train_epochs']}")
    print(f"   Batch size: {config['training']['per_device_train_batch_size']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   LoRA rank: {config['lora']['r']}")
    print(f"   Max length: {config['data']['max_length']}")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, config)

    # Load datasets
    train_file = os.path.join(args.data_dir, "ecommerce_train.json")
    eval_file = os.path.join(args.data_dir, "ecommerce_eval.json")

    train_dataset = load_dataset(train_file, tokenizer, config["data"]["max_length"])
    eval_dataset = load_dataset(eval_file, tokenizer, config["data"]["max_length"])

    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Evaluation samples: {len(eval_dataset)}")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    # Training arguments
    training_args = create_training_args(config, args.output_dir)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Save configuration
    config_file = os.path.join(args.output_dir, "training_config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    print(f"\n💾 Saved configuration to {config_file}")

    # Training
    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    print("=" * 60)

    # Resume from checkpoint if exists
    if os.path.exists(os.path.join(args.output_dir, "checkpoint-*")):
        print("📂 Resuming from checkpoint...")

    # Train
    trainer.train()

    # Save final model
    print(f"\n💾 Saving final model to {args.output_dir}...")
    trainer.save_model()

    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)

    # Save training summary
    summary = {
        "model_name": args.model_name,
        "base_model": args.model_name,
        "training_time": str(datetime.now()),
        "total_steps": trainer.state.global_step,
        "final_eval_loss": trainer.state.log_history[-2]["eval_loss"] if len(trainer.state.log_history) > 1 else "N/A",
        "lora_config": config["lora"],
        "training_config": config["training"]
    }

    summary_file = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {args.output_dir}")
    print(f"📊 Training summary: {summary_file}")
    print(f"\n🎉 Your Ecommerce Language Model is ready!")
    print("\nNext steps:")
    print(f"  1. Test the model: python scripts/inference.py --model_path {args.output_dir}")
    print(f"  2. Run evaluation: python scripts/evaluate.py --model_path {args.output_dir}")
    print(f"  3. Push to Hub: huggingface-cli login && python scripts/push_to_hub.py --model_path {args.output_dir}")

if __name__ == "__main__":
    main()
