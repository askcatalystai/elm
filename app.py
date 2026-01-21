#!/usr/bin/env python3
"""
Hugging Face Spaces App for ELM
Interactive demo of the Ecommerce Language Model
"""

import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import json

# Model loading function
def load_model(model_path: str):
    """Load the fine-tuned ELM model"""
    try:
        # Check if it's a PEFT model
        if "adapter_config.json" in os.listdir(model_path):
            config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Load model (change path to your model)
MODEL_PATH = "askcatalystai/elm"  # Hugging Face Hub model
model, tokenizer = load_model(MODEL_PATH)

if model is None:
    print("⚠️  Model not loaded. Using base model.")
    model, tokenizer = load_model("meta-llama/Llama-3.1-8B-Instruct")

def chat_completion(message: str, history: list):
    """Generate response using ELM"""
    if model is None or tokenizer is None:
        return "Error: Model not loaded"

    # Convert history to messages format
    messages = [
        {"role": "system", "content": "You are ELM, an expert ecommerce AI assistant."}
    ]
    for msg in history:
        messages.append({"role": "user", "content": msg[0]})
        if len(msg) > 1 and msg[1]:
            messages.append({"role": "assistant", "content": msg[1]})
    messages.append({"role": "user", "content": message})

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("assistant")[-1].strip()

    return response

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛍️ ELM - Ecommerce Language Model")
    gr.Markdown("""
    ## Fine-tuned Llama-3.1-8B for Ecommerce

    ### Features:
    - ✅ Product description generation
    - ✅ Conversational commerce
    - ✅ Product categorization
    - ✅ Tool calling for ecommerce tasks

    ### Try these examples:
    - "Find wireless headphones under $150"
    - "Write a product description for a laptop"
    - "Classify: LED Desk Lamp"
    - "I want to return my order #12345"
    """)

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your message", placeholder="Ask me anything about ecommerce...")
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        if history and history[-1][1] is None:
            response = chat_completion(history[-1][0], history[:-1])
            history[-1][1] = response
        return history

    chatbot.change(bot, chatbot, chatbot)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: [], outputs=[chatbot], queue=False)

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
