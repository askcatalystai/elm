# Deploy ELM on Hugging Face Spaces

This guide shows you how to deploy your Ecommerce Language Model (ELM) on **Hugging Face Spaces** for public demo and API access.

## 🚀 Option 1: Deploy Trained Model to Spaces

### Step 1: Push Your Trained Model to Hub

```bash
# After training locally
python scripts/push_to_hub.py \
    --model_path outputs/elm-llama-8b \
    --repo_id askcatalystai/elm

# Or with real data
python scripts/push_to_hub.py \
    --model_path outputs/elm-real-data \
    --repo_id askcatalystai/elm-real-data
```

### Step 2: Create Spaces App

1. **Go to Spaces**: https://huggingface.co/spaces

2. **Create New Space**:
   - Name: `elm-demo`
   - License: `apache-2.0`
   - Hardware: `CPU basic` (or `GPU upgrade` for faster inference)
   - SDK: `Gradio`
   - Template: `Blank`

3. **Clone the Space**:
```bash
git clone https://huggingface.co/spaces/askcatalystai/elm-demo
cd elm-demo
```

4. **Copy Files**:
```bash
# Copy our app
cp /path/to/ecommerce-llm/app.py ./
cp /path/to/ecommerce-llm/hf_requirements.txt ./requirements.txt

# Update app.py to use your model
sed -i 's/askcatalystai\/elm/YOUR_USERNAME\/YOUR_MODEL/' app.py
```

5. **Push to Spaces**:
```bash
git add .
git commit -m "Add ELM demo app"
git push
```

6. **Your app is live!** 🎉
   - URL: https://huggingface.co/spaces/askcatalystai/elm-demo

### Step 3: Enable API Access

Once deployed, your model is automatically available via API:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/askcatalystai/elm"

response = requests.post(
    API_URL,
    headers={"Authorization": "Bearer YOUR_HF_TOKEN"},
    json={
        "inputs": "Find wireless headphones under $100",
        "parameters": {"temperature": 0.7}
    }
)

result = response.json()
print(result)
```

## 🏗️ Option 2: Train Directly on Hugging Face

### Step 1: Create Training Script

We've created `huggingface_train.py` optimized for HF infrastructure.

### Step 2: Upload to Hub

```bash
# Create training directory
mkdir hf-training
cp huggingface_train.py hf-training/
cp -r data/processed hf-training/  # Your training data
cp config/training_config.yaml hf-training/
cd hf-training

# Initialize git
git init
git add .
git commit -m "Initial training setup"

# Push to HF
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/elm-training
git push -u origin main
```

### Step 3: Create Training Space

1. Create new Space with:
   - Name: `elm-training`
   - Hardware: `GPU upgrade` (required for training)
   - SDK: `Docker`
   - Dockerfile: Use HF's PyTorch template

2. Add Dockerfile:
```dockerfile
FROM pytorch/pytorch:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "huggingface_train.py"]
```

### Step 4: Start Training

Once pushed, HF will automatically:
- Install dependencies
- Start training
- Save checkpoints
- Upload final model to Hub

## 📋 Files for HF Deployment

### Required Files:

```
hf-deployment/
├── app.py                      # Gradio demo app
├── requirements.txt           # Dependencies (use hf_requirements.txt)
├── huggingface_train.py       # Training script
├── README.md                  # Space documentation
├── config.yaml               # Model config (optional)
└── Dockerfile                 # For Docker spaces (optional)
```

### Optional Files:

```
├── examples/                  # Example conversations
│   ├── product_description.md
│   ├── conversational_commerce.md
│   └── tool_calling.md
├── static/                   # Static files (CSS, images)
└── utils.py                  # Helper functions
```

## 🎨 Customize Your Space

### Update README.md

```markdown
# ELM - Ecommerce Language Model

## About
Fine-tuned Llama-3.1-8B-Instruct for ecommerce tasks.

## Features
- Product description generation
- Conversational commerce
- Product categorization
- Tool calling

## Examples
Try these prompts:
- "Find wireless headphones under $150"
- "Write a product description for a laptop"
- "Classify: LED Desk Lamp"

## API
```python
import requests
response = requests.post(
    "https://api-inference.huggingface.co/models/YOUR_USERNAME/YOUR_MODEL",
    json={"inputs": "Your prompt"}
)
```
```

### Add Custom CSS

Create `static/style.css`:

```css
.gradio-container {
    font-family: 'Inter', sans-serif;
}

h1 {
    color: #6366f1;
}

.chatbot {
    border-radius: 12px;
}
```

Update `app.py`:

```python
demo.load_css("static/style.css")
```

## 🔧 Hardware Recommendations

| Use Case | Hardware | Cost/Month | Description |
|----------|----------|------------|-------------|
| **Demo** | CPU basic | Free | Public demos, light usage |
| **API** | CPU upgrade | $0.06/hour | Production API |
| **Training** | GPU upgrade | $3.00/hour | Fine-tuning |
| **Production** | GPU large | $6.00/hour | High-traffic production |

## 📊 Monitoring

HF Spaces provides:
- **Usage metrics**: Requests, users, latency
- **Logs**: Console output, errors
- **Scaling**: Auto-sleep, CPU/GPU switching

## 💰 Cost Estimation

| Scenario | Hardware | Usage | Monthly Cost |
|----------|----------|-------|-------------|
| **Demo** | CPU basic | 24/7 | Free |
| **Light API** | CPU upgrade | 100 req/day | ~$20 |
| **Production API** | GPU upgrade | 1000 req/day | ~$200 |
| **Training** | GPU upgrade | 20 hours | ~$60 |

## 🛠️ Troubleshooting

### Model Doesn't Load

Check:
1. Model is pushed to Hub
2. `adapter_config.json` exists
3. Tokenizer files present
4. Model size < hardware limit

### Out of Memory

Solutions:
1. Use smaller batch size
2. Enable gradient checkpointing
3. Use 4-bit quantization
4. Switch to larger hardware

### Slow Inference

Optimizations:
1. Enable Flash Attention 2
2. Use `torch.compile()`
3. Batch requests
4. Use GPU hardware

## 🎉 Success!

Your ELM model will be:
- ✅ Publicly accessible
- ✅ API enabled
- ✅ Scalable
- ✅ Monitored
- ✅ Easy to share

**URL format:**
- Demo: https://huggingface.co/spaces/YOUR_USERNAME/elm
- API: https://api-inference.huggingface.co/models/YOUR_USERNAME/elm

## 📚 Resources

- [HF Spaces Documentation](https://huggingface.co/docs/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [HF Hub API](https://huggingface.co/docs/hub/main)
- [Deploy Models Guide](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#usage-load)

## 🚀 Quick Start Checklist

- [ ] Train model locally
- [ ] Push to Hub: `python scripts/push_to_hub.py`
- [ ] Create Space on HF
- [ ] Copy `app.py` and `requirements.txt`
- [ ] Update `MODEL_PATH` in `app.py`
- [ ] Commit and push
- [ ] Test at Space URL
- [ ] Share with community!

---

**Ready to deploy?** Run:

```bash
# 1. Push model to Hub
python scripts/push_to_hub.py --model_path outputs/elm-llama-8b --repo_id askcatalystai/elm

# 2. Create Space
# Go to https://huggingface.co/spaces and create new Space

# 3. Deploy
git clone https://huggingface.co/spaces/YOUR_USERNAME/elm-demo
cp app.py hf_requirements.txt elm-demo/
cd elm-demo
git add .
git commit -m "Deploy ELM"
git push
```

**Your ELM will be live in minutes!** 🎊
