#!/usr/bin/env python3
"""
Deploy fine-tuned Ecommerce Language Model to HuggingFace Hub
"""

import os
import argparse
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

def push_model_to_hub(model_path: str, repo_id: str, private: bool = False):
    """
    Push fine-tuned model to HuggingFace Hub

    Args:
        model_path: Path to the fine-tuned model directory
        repo_id: HuggingFace repository ID (e.g., "username/elm-llama-3b")
        private: Whether to make the repository private
    """
    print("=" * 60)
    print("🚀 DEPLOYING MODEL TO HUGGINGFACE HUB")
    print("=" * 60)

    # Initialize HF API
    api = HfApi()

    # Login
    print("\n🔐 Please login to HuggingFace Hub...")
    api.login()

    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path '{model_path}' not found")
        return False

    # Check if it's a PEFT model
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_peft:
        print(f"\n📦 Detected PEFT/LoRA model")
        print(f"   Loading base model configuration...")

        # Load PEFT config to get base model name
        config = PeftConfig.from_pretrained(model_path)
        base_model_name = config.base_model_name_or_path

        print(f"   Base model: {base_model_name}")

        # Create model card for PEFT model
        create_model_card(model_path, is_peft=True, base_model=base_model_name)
    else:
        print(f"\n📦 Detected full model")
        create_model_card(model_path, is_peft=False)

    # Upload to Hub
    print(f"\n📤 Uploading to HuggingFace Hub...")
    print(f"   Repository ID: {repo_id}")
    print(f"   Private: {private}")

    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            private=private,
            commit_message="Upload Ecommerce Language Model (ELM)"
        )

        print("\n✅ Model successfully uploaded!")
        print(f"\n🔗 Repository URL: https://huggingface.co/{repo_id}")
        print(f"\n💡 To use this model:")
        print(f"   from transformers import AutoTokenizer, AutoModelForCausalLM")
        print(f"   from peft import PeftModel, PeftConfig")
        print(f"   ")
        print(f"   model_name = '{repo_id}'")
        print(f"   tokenizer = AutoTokenizer.from_pretrained(model_name)")
        print(f"   model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)")
        print(f"   # If using LoRA:")
        print(f"   # model = PeftModel.from_pretrained(model, model_name)")

        return True

    except Exception as e:
        print(f"\n❌ Error uploading model: {str(e)}")
        return False

def create_model_card(model_path: str, is_peft: bool, base_model: str = None):
    """Create a model card (README.md) for the repository"""

    model_card = f"""# Ecommerce Language Model (ELM)

A fine-tuned {f'LoRA adapter for {base_model}' if is_peft else 'language model'} optimized for ecommerce applications.

## Model Details

{f'- **Base Model**: {base_model}' if base_model else '- **Architecture**: Custom fine-tuned model'}
- **Task**: Ecommerce assistant (product recommendations, customer support, categorization)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Language**: English

## Capabilities

This model excels at:

### 🛍️ Product Description Generation
- Creates compelling, SEO-optimized product descriptions
- Highlights key features and benefits
- Adapts to different brand voices

### 💬 Conversational Commerce
- Handles customer inquiries professionally
- Provides product recommendations
- Assists with order status and returns

### 🏷️ Product Categorization
- Automatically categorizes products
- Generates relevant tags
- Maintains category consistency

### 🔧 Tool Calling
- Uses tools to fetch product information
- Checks inventory and pricing
- Calculates shipping costs

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('{base_model if base_model else 'your-username/elm-llama-3b'}')

# Load model
model = AutoModelForCausalLM.from_pretrained(
    '{base_model if base_model else 'your-username/elm-llama-3b'}',
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter (if applicable)
{f"model = PeftModel.from_pretrained(model, '{base_model if base_model else 'your-username/elm-llama-3b'}')" if is_peft else "# Full model - no adapter needed"}

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Generate response
def chat_completion(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|end_of_signal|>")[0].strip()

# Example conversation
messages = [
    {{"role": "system", "content": "You are ELM, an expert ecommerce assistant."}},
    {{"role": "user", "content": "Hi, I'm looking for wireless headphones under $150"}}
]

response = chat_completion(messages)
print(response)
```

### Product Description Example

```python
messages = [
    {{"role": "system", "content": "You are ELM, an expert ecommerce copywriter."}},
    {{"role": "user", "content": "Write a product description for: Wireless Headphones, $99, features: noise canceling, 30hr battery, wireless"}}
]

response = chat_completion(messages)
print(response)
```

### Tool Calling Example

```python
messages = [
    {{"role": "system", "content": "You are an ecommerce assistant. Use tools when needed."}},
    {{"role": "user", "content": "Check if the Sony WH-1000XM5 is in stock"}}
]

response = chat_completion(messages)
# Output will include: <tool_call>get_inventory(product_id='Sony WH-1000XM5')</tool_call>
```

## Training Details

### Dataset
- **Size**: ~4,000 instruction-response pairs
- **Categories**:
  - Product descriptions (40%)
  - Conversational commerce (30%)
  - Product categorization (20%)
  - Tool calling examples (10%)

### Fine-tuning Configuration
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank (r)**: 64
- **Alpha**: 128
- **Dropout**: 0.1
- **Target Modules**: Q/K/V/O projections, gate/up/down projections
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Batch Size**: 4 (per device)
- **Context Length**: 2,048 tokens

### Hardware Requirements
- **Training**: 1x A100 (40GB) or RTX 4090 (24GB)
- **Inference**: RTX 3090/4090 or CPU (slower)

## Evaluation Metrics

| Task | Score |
|------|-------|
| Product Descriptions | TBD |
| Categorization Accuracy | TBD |
| Conversational Quality | TBD |
| Tool Calling | TBD |

*Note: Run `python scripts/evaluate.py` to generate evaluation metrics for your instance*

## Limitations

- English language only
- Trained on syntheticecommerce data
- May hallucinate product information
- Tool calls are simulated (not actual API calls)

## License

{f'- Base Model: {base_model} License' if base_model else '- Custom License'}
- LoRA Adapters: Apache 2.0

## Citation

If you use this model, please cite:

```bibtex
@software{{elm_2024,
  title = {{Ecommerce Language Model (ELM)}},
  author = {{Your Name}},
  year = {{2024}},
  url = {{https://huggingface.co/{base_model if base_model else 'your-username/elm-llama-3b'}}}
}}
```

## Acknowledgments

- Base model: {base_model if base_model else 'N/A'}
- Fine-tuning framework: PEFT (HuggingFace)
- Training infrastructure: HuggingFace/RunPod

---

**Made with ❤️ for the ecommerce community**
"""

    readme_path = os.path.join(model_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(model_card)

    print(f"✓ Created model card: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description="Deploy model to HuggingFace Hub")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repository ID (e.g., 'username/elm-llama-3b')")
    parser.add_argument("--private", action="store_true",
                        help="Make repository private (default: public)")

    args = parser.parse_args()

    success = push_model_to_hub(args.model_path, args.repo_id, args.private)

    if success:
        print("\n🎉 Deployment successful!")
        print("\nNext steps:")
        print(f"  1. View your model: https://huggingface.co/{args.repo_id}")
        print(f"  2. Test the model: python scripts/inference.py --model_path {args.repo_id}")
        print(f"  3. Share with the community!")
    else:
        print("\n❌ Deployment failed. Please check the error messages above.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
