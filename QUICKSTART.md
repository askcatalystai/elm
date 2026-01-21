# 🚀 Quick Start Guide - Ecommerce Language Model (ELM)

This guide will get you up and running with your fine-tuned ecommerce language model in minutes!

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM) or A100 (40GB VRAM)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space (SSD recommended)

### Software Requirements
- Python 3.8+
- CUDA 11.8+ / PyTorch 2.0+
- HuggingFace account (for model hosting)

## Installation

### 1. Clone and Setup

```bash
# Clone the repository (or use your existing directory)
cd ecommerce-llm

# Create virtual environment
python -m venv elm-env
source elm-env/bin/activate  # On Windows: elm-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Access

```bash
# Login to HuggingFace (required for Llama access)
huggingface-cli login

# Verify login
huggingface-cli whoami
```

### 3. Verify Installation

```python
# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}')"
```

Expected output:
```
PyTorch: 2.0.0+
CUDA available: True
GPU: NVIDIA RTX 4090
```

## Training Your Model

### Step 1: Generate Training Data (10 minutes)

```bash
# Generate 4,000 training samples
python scripts/prepare_data.py
```

This creates:
- `data/processed/ecommerce_train.json` - 80% for training
- `data/processed/ecommerce_eval.json` - 20% for evaluation

**What it generates:**
- Product descriptions (40%)
- Conversational commerce (30%)
- Product categorization (20%)
- Tool calling examples (10%)

### Step 2: Start Training (2-4 hours)

```bash
# Start training on Llama-3.1-8B-Instruct
python scripts/train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/elm-llama-8b \
    --config config/training_config.yaml
```

**Training Configuration:**
- **Base Model**: Llama-3.1-8B-Instruct
- **Method**: LoRA (rank=64, alpha=128)
- **Quantization**: 4-bit (QLoRA)
- **Epochs**: 3
- **Batch Size**: 4 (effective: 16 with gradient accumulation)
- **Learning Rate**: 2e-4
- **Training Time**: ~3 hours on RTX 4090

### Step 3: Monitor Training

```bash
# View training logs in real-time
tensorboard --logdir outputs/elm-llama-8b/logs

# Or check the latest training summary
cat outputs/elm-llama-8b/training_summary.json
```

## Testing Your Model

### Quick Test

```bash
# Test all features
python scripts/inference.py \
    --model_path outputs/elm-llama-8b \
    --task all
```

**Output includes:**
- ✓ Product description generation
- ✓ Conversational commerce examples
- ✓ Product categorization
- ✓ Tool calling demonstrations

### Interactive Chat

```bash
# Start interactive chat
python scripts/inference.py \
    --model_path outputs/elm-llama-8b \
    --task interactive
```

Example interactions:
```
You: I need wireless headphones under $150
ELM: I'd be happy to help! Let me search for wireless headphones with noise canceling under $150.
<tool_call>search_products(query='wireless headphone noise canceling under $150')</tool_call>

You: Classify this product: LED Desk Lamp
ELM: The LED Desk Lamp belongs to the Electronics category.
```

### Evaluation

```bash
# Run comprehensive evaluation
python scripts/evaluate.py \
    --model_path outputs/elm-llama-8b \
    --output_dir evaluation_results
```

**Evaluation metrics:**
- Product descriptions (relevance, completeness, creativity, readability)
- Categorization accuracy
- Conversational quality
- Tool calling accuracy

## Deploying Your Model

### Option 1: Push to HuggingFace Hub

```bash
# Make your model publicly available
python scripts/push_to_hub.py \
    --model_path outputs/elm-llama-8b \
    --repo_id your-username/elm-llama-8b

# Or private repository
python scripts/push_to_hub.py \
    --model_path outputs/elm-llama-8b \
    --repo_id your-username/elm-llama-8b-private \
    --private
```

**Access your model:**
- Public: https://huggingface.co/your-username/elm-llama-8b
- API: `https://api-inference.huggingface.co/models/your-username/elm-llama-8b`

### Option 2: Local Inference Server

```python
# Create a simple FastAPI server
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

# Load model
tokenizer = AutoTokenizer.from_pretrained("outputs/elm-llama-8b")
model = AutoModelForCausalLM.from_pretrained(
    "outputs/elm-llama-8b",
    torch_dtype=torch.float16,
    device_map="auto"
)

@app.post("/chat")
async def chat(messages: list):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

# Run with: uvicorn server:app --reload
```

## Using Your Model

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
model_name = "outputs/elm-llama-8b"  # or your HF repo ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# If using LoRA adapter
model = PeftModel.from_pretrained(model, model_name)

# Chat completion
messages = [
    {"role": "system", "content": "You are ELM, an ecommerce assistant."},
    {"role": "user", "content": "Recommend a laptop for programming"}
]

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
print(response)
```

### Product Description Generator

```python
def generate_product_description(name, price, features):
    messages = [
        {"role": "system", "content": "Write compelling product descriptions."},
        {"role": "user", "content": f"Product: {name}\nPrice: ${price}\nFeatures: {features}"}
    ]

    # ... (same as above) ...

    return response

# Example
desc = generate_product_description(
    name="Wireless Headphones",
    price=99,
    features="noise canceling, 30hr battery, wireless"
)
print(desc)
```

### Tool Calling

```python
messages = [
    {"role": "system", "content": "Use tools to help customers."},
    {"role": "user", "content": "Check if iPhone 15 is in stock"}
]

# Generate response (will include tool calls)
response = generate(messages)
print(response)
# Output: "... <tool_call>get_inventory(product_id='iPhone 15')</tool_call> ..."
```

## Customization

### Modify Training Data

Edit `scripts/prepare_data.py` to add your own data:

```python
# Add your products
products = [
    {
        "category": "Your Category",
        "items": [
            {"name": "Your Product", "price": 100, "features": ["feature1", "feature2"]}
        ]
    }
]
```

### Adjust Training Parameters

Edit `config/training_config.yaml`:

```yaml
training:
  num_train_epochs: 5  # Increase for more training
  learning_rate: 0.0001  # Adjust learning rate
  per_device_train_batch_size: 2  # Reduce if OOM errors

lora:
  r: 128  # Increase LoRA rank (more parameters)
```

### Add New Tasks

1. Generate data in `prepare_data.py`
2. Update training config if needed
3. Retrain model
4. Update evaluation in `evaluate.py`

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
per_device_train_batch_size: 2

# Enable gradient checkpointing (already enabled)
# Use deeper quantization (8-bit instead of 4-bit)
```

### Slow Training

```bash
# Enable Flash Attention 2 (requires CUDA 12.1+)
pip install flash-attn --no-build-isolation

# Or use gradient checkpointing and larger batch size
```

### Model Not Generating Tool Calls

```bash
# Check training data includes tool calls
grep -r "tool_call" data/processed/ecommerce_train.json | head -5

# Verify system prompt includes tool instructions
```

### Poor Quality Output

```bash
# Increase training epochs
num_train_epochs: 5

# Or improve data quality
# - Add more examples
# - Remove duplicates
# - Ensure consistent formatting
```

## Performance Tips

### Faster Inference

```python
# Use smaller context
max_length=512  # instead of 2048

# Use 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
)

# Enable tensorrt (NVIDIA GPUs only)
# Requires: tensorrt, onnx, onnxruntime
```

### Better Quality

```python
# Adjust generation parameters
outputs = model.generate(
    **inputs,
    temperature=0.7,  # Lower = more focused, higher = more creative
    top_p=0.9,        # Nucleus sampling
    repetition_penalty=1.1,  # Avoid repetition
    do_sample=True    # Enable sampling
)
```

## Cloud Training

### RunPod

```bash
# Start RunPod instance (A100 40GB)
# SSH into instance
git clone <your-repo>
cd ecommerce-llm
pip install -r requirements.txt

# Run training
python scripts/train.py --model_name meta-llama/Llama-3.1-8B-Instruct
```

### Google Colab Pro

```python
# Upload notebook
# Enable GPU: Runtime -> Change runtime type -> GPU
!pip install -r requirements.txt
!python scripts/prepare_data.py
!python scripts/train.py --model_name meta-llama/Llama-3.1-8B-Instruct
```

### AWS EC2

```bash
# Launch p4d.24xlarge or g5.12xlarge instance
# Install drivers and PyTorch
# Run training script
```

## Resources

- **HuggingFace Hub**: https://huggingface.co/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **Llama Documentation**: https://llama.meta.com/

## Need Help?

- Check the main README.md
- Review scripts/evaluate.py for examples
- Test with smaller models (Phi-3-Mini) first
- Start with 100 training samples for quick tests

## Next Steps

1. ✅ Train your model
2. ✅ Evaluate performance
3. ✅ Deploy to HuggingFace Hub
4. ✅ Build applications
5. ✅ Share with community!

---

**Happy fine-tuning! 🎉**
