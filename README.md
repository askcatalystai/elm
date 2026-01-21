# Ecommerce Language Model (ELM) 🚀

A fine-tuned **Llama-3.1-8B-Instruct** model optimized for ecommerce applications with exceptional **tool-calling** capabilities.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## ✨ Key Features

### 🎯 Core Capabilities

1. **📝 Product Description Generation**
   - Creates compelling, SEO-optimized product copy
   - Highlights key features and benefits
   - Adapts to different brand voices

2. **💬 Conversational Commerce**
   - Handles customer inquiries professionally
   - Provides personalized product recommendations
   - Manages sales conversations with context

3. **🏷️ Product Categorization**
   - Automatically classifies products into categories
   - Generates relevant tags and attributes
   - Maintains taxonomy consistency

4. **🔧 Advanced Tool Calling** ⭐
   - **Primary Feature**: Seamless tool integration
   - Search products, check inventory, calculate shipping
   - Access order history, process returns
   - Structured tool calls: `<tool_call>search_products(query='...')</tool_call>`

## 🏗️ Architecture

### Base Model: Llama-3.1-8B-Instruct

**Why Llama-3.1-8B-Instruct?**
- ✅ **Excellent tool-calling** - Best-in-class for function calling
- ✅ **Strong reasoning** - Superior problem-solving capabilities
- ✅ **Large context** - 128K token window
- ✅ **Open source** - Freely available weights
- ✅ **Well-documented** - Extensive community support
- ✅ **Production-ready** - Battle-tested in many applications

**Model Specs:**
- Parameters: 8B (perfect balance of capability and efficiency)
- Context Window: 128,000 tokens
- Training: Supervised fine-tuning + RLHF
- Strengths: Instruction following, reasoning, tool use

### Fine-tuning: LoRA + QLoRA

- **Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (QLoRA)
- **Memory Usage**: ~16GB VRAM (vs 64GB for full fine-tuning)
- **Trainable Parameters**: ~0.1% (8M parameters)
- **Training Speed**: 3-4 hours on RTX 4090
- **Quality**: Comparable to full fine-tuning

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd ecommerce-llm

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for Llama access)
huggingface-cli login
```

### One-Command Training

```bash
# 1. Generate training data (10 min)
python scripts/prepare_data.py

# 2. Train model (3 hours on RTX 4090)
python scripts/train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/elm-llama-8b

# 3. Test model (instant)
python scripts/inference.py \
    --model_path outputs/elm-llama-8b \
    --task interactive
```

**That's it! 🎉 Your ecommerce LLM is ready!**

## 📚 Project Structure

```
ecommerce-llm/
├── README.md                    # This file
├── QUICKSTART.md               # Detailed quick start guide
├── requirements.txt            # Python dependencies
│
├── scripts/                    # Training & evaluation scripts
│   ├── prepare_data.py         # Generate 4K training samples
│   ├── train.py               # LoRA fine-tuning script
│   ├── inference.py           # Testing & examples
│   ├── evaluate.py            # Comprehensive evaluation
│   └── push_to_hub.py         # Deploy to HuggingFace Hub
│
├── config/
│   └── training_config.yaml   # Training hyperparameters
│
├── data/
│   ├── raw/                   # Original datasets
│   └── processed/             # Training-ready datasets
│       ├── ecommerce_train.json
│       └── ecommerce_eval.json
│
└── outputs/
    ├── elm-llama-8b/          # Fine-tuned model
    │   ├── adapter_model.safetensors
    │   ├── adapter_config.json
    │   └── tokenizer files
    └── logs/                   # Training logs
```

## 📊 Training Data

### Dataset: 4,000 Samples

| Category | Percentage | Samples | Description |
|----------|-----------|---------|-------------|
| **Product Descriptions** | 40% | 1,600 | Compelling product copy with features, benefits |
| **Conversational Commerce** | 30% | 1,200 | Customer service, Q&A, recommendations |
| **Product Categorization** | 20% | 800 | Auto-classification, tagging, taxonomy |
| **Tool Calling Examples** | 10% | 400 | Structured tool usage for ecommerce tasks |

### Sample Data Format

```json
{
  "instruction": "You are a helpful ecommerce assistant. A customer is asking about a product.",
  "input": "Customer: Hi, I'm looking for wireless headphones under $150 with noise canceling",
  "tools": ["search_products(query='wireless headphone noise canceling under $150')"],
  "output": "I'd be happy to help you find the perfect wireless headphones! Let me search for options with noise canceling under $150.\n\n<tool_call>search_products(query='wireless headphone noise canceling under $150')</tool_call>\n\nGreat news! I found several excellent options for you..."
}
```

## 🛠️ Usage Examples

### 1. Product Description Generator

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("outputs/elm-llama-8b")
model = AutoModelForCausalLM.from_pretrained("outputs/elm-llama-8b", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "outputs/elm-llama-8b")

# Generate product description
messages = [
    {"role": "system", "content": "You are ELM, an expert ecommerce copywriter."},
    {"role": "user", "content": "Write a product description:\n\nProduct: Wireless Headphones\nPrice: $99\nFeatures: noise canceling, 30hr battery, wireless, premium sound"}
]

# ... (chat completion code) ...

# Output:
# "🎧 Experience Audio Like Never Before!
#  Transform your listening experience with these premium wireless headphones featuring..."
```

### 2. Conversational Commerce

```python
messages = [
    {"role": "system", "content": "You are ELM, a helpful ecommerce assistant."},
    {"role": "user", "content": "I need a laptop for video editing under $1500"}
]

# ... (chat completion) ...

# Output:
# "I'd be happy to help you find the perfect laptop for video editing! Let me search for options under $1500 with the specs you'll need..."
```

### 3. Product Categorization

```python
messages = [
    {"role": "system", "content": "You are an expert at categorizing products."},
    {"role": "user", "content": "Classify: LED Desk Lamp with USB charging port"}
]

# Output:
# "The LED Desk Lamp with USB charging port belongs to the Electronics category."
```

### 4. Tool Calling ⭐

```python
messages = [
    {"role": "system", "content": "You are an ecommerce assistant. Use tools to help customers."},
    {"role": "user", "content": "Check if the Sony WH-1000XM5 is in stock"}
]

# Output:
# "I'll check the inventory for you!\n\n<tool_call>get_inventory(product_id='Sony WH-1000XM5')</tool_call>\n\nLet me look that up for you..."
```

## 🔧 Available Tools

The model is trained to use these ecommerce tools:

| Tool | Purpose | Parameters |
|------|---------|------------|
| `search_products` | Find products by query/price/category | query, min_price, max_price, category |
| `get_product_info` | Get detailed product information | product_id |
| `get_inventory` | Check stock levels | product_id, location |
| `calculate_shipping` | Calculate shipping costs | weight, destination, shipping_method |
| `get_customer_orders` | Retrieve order history | customer_id, status |
| `initiate_return` | Process returns | order_id, product_id, reason |
| `get_discount_eligibility` | Check promo codes | customer_id, promo_code |
| `get_similar_products` | Find similar items | product_id, limit |

## 📈 Performance

### Hardware Requirements

#### Training (QLoRA)
- **GPU**: NVIDIA RTX 4090 (24GB) or A100 (40GB)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ SSD
- **Time**: ~3 hours for 3 epochs

#### Inference
- **GPU**: RTX 4090 (recommended) or RTX 3080+ (minimum)
- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 16GB+ system memory
- **Storage**: 10GB for model

### Evaluation Metrics

| Task | Score* |
|------|--------|
| Product Description Quality | TBD |
| Categorization Accuracy | TBD |
| Conversational Helpfulness | TBD |
| Tool Calling Precision | TBD |

*Run `python scripts/evaluate.py` to generate metrics for your instance*

## 🚀 Deployment

### Deploy to HuggingFace Hub

```bash
# Make your model publicly available
python scripts/push_to_hub.py \
    --model_path outputs/elm-llama-8b \
    --repo_id your-username/elm-llama-8b

# Access at: https://huggingface.co/your-username/elm-llama-8b
```

### Local Inference Server

```python
# FastAPI example (see scripts/inference.py for full code)
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

@app.post("/chat")
async def chat(messages: list):
    # Your chat completion code here
    return {"response": generated_text}
```

## 🧪 Testing & Evaluation

### Quick Test

```bash
# Test all capabilities
python scripts/inference.py --model_path outputs/elm-llama-8b --task all
```

### Interactive Chat

```bash
# Start chat session
python scripts/inference.py --model_path outputs/elm-llama-8b --task interactive
```

### Comprehensive Evaluation

```bash
# Run full evaluation suite
python scripts/evaluate.py \
    --model_path outputs/elm-llama-8b \
    --output_dir evaluation_results
```

**Generates:**
- Product description quality scores
- Categorization accuracy
- Conversational metrics
- Tool calling precision

## ⚙️ Customization

### Modify Training Data

Edit `scripts/prepare_data.py`:

```python
# Add your own products
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
  num_train_epochs: 5          # More training
  learning_rate: 0.0001        # Lower learning rate
  per_device_train_batch_size: 2  # Smaller batch

lora:
  r: 128                       # Higher rank (more parameters)
  lora_alpha: 256              # Higher alpha
```

### Add New Tasks

1. Generate data in `scripts/prepare_data.py`
2. Re-run training: `python scripts/train.py`
3. Update evaluation in `scripts/evaluate.py`

## 🛠️ Troubleshooting

### Out of Memory (OOM)

```yaml
# Reduce batch size
per_device_train_batch_size: 2

# Use 8-bit quantization (less memory)
load_in_8bit: true

# Enable gradient checkpointing (already enabled)
gradient_checkpointing: true
```

### Poor Quality Output

```bash
# Increase training epochs
num_train_epochs: 5

# Improve data quality
# - Add more examples
# - Remove duplicates
# - Ensure formatting consistency
```

### Slow Training

```bash
# Enable Flash Attention 2 (CUDA 12.1+)
pip install flash-attn --no-build-isolation

# Or use gradient checkpointing
# Already configured in training_config.yaml
```

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Detailed quick start guide with examples
- **[scripts/inference.py](scripts/inference.py)** - Comprehensive usage examples
- **[scripts/evaluate.py](scripts/evaluate.py)** - Evaluation metrics and tests

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- More training data (real ecommerce datasets)
- Additional tool integrations
- Multi-language support
- Better evaluation metrics
- Performance optimizations

## 📝 License

- **Base Model**: Llama-3.1-8B-Instruct (Custom License)
- **LoRA Adapters**: Apache 2.0
- **Training Scripts**: MIT License
- **This Project**: Apache 2.0

## 🙏 Acknowledgments

- **Meta** for Llama models
- **HuggingFace** for transformers, PEFT, and training infrastructure
- **Microsoft** for BitsAndBytes quantization
- **Open source ML community**

## 🔗 Resources

- **HuggingFace Model Hub**: https://huggingface.co/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **Llama Documentation**: https://llama.meta.com/

## 📞 Support

- Check [QUICKSTART.md](QUICKSTART.md) for detailed guides
- Review evaluation results in `evaluation_results/`
- Test with smaller models (Phi-3-Mini) for quick iterations

---

<div align="center">

**Built with ❤️ for the ecommerce community**

[⭐ Star this repo](https://github.com/yourusername/ecommerce-llm) if you find it helpful!

</div>
