#!/usr/bin/env python3
"""
Hugging Face Training Script for ELM
Optimized for HF Spaces, Hub training, and Infrastructure
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", default="./outputs/elm-hf")
    parser.add_argument("--data_dir", default="./data/processed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    print(f"🚀 Training {args.model_name}")
    print(f"Output: {args.output_dir}")

    # Setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    train_file = os.path.join(args.data_dir, "ecommerce_train.json")
    if not os.path.exists(train_file):
        # Use default synthetic data location
        train_file = "data/processed/ecommerce_train.json"

    dataset = load_dataset('json', data_files=train_file)['train']

    # Tokenize
    def tokenize_function(examples):
        text = tokenizer(
            examples['instruction'] + "\n" + examples.get('input', ''),
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )
        text["labels"] = text["input_ids"].copy()
        return text

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=250,
        eval_steps=250,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=True,
        gradient_checkpointing=True,
        report_to="tensorboard"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    print(f"✅ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
