#!/bin/bash

# Quick Start Script: Train ELM with Real Ecommerce Data
# This script downloads, processes, and trains on real ecommerce datasets

set -e  # Exit on error

echo "================================================================"
echo "🚀 ELM - TRAIN WITH REAL ECOMMERCE DATA"
echo "================================================================"
echo ""
echo "This script will:"
echo "1. Download real ecommerce datasets (~8GB)"
echo "2. Process and combine them into training format"
echo "3. Train the Llama-3.1-8B-Instruct model"
echo "4. Deploy to HuggingFace Hub"
echo ""
echo "Total time: ~4 hours (including downloads)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "================================================================"
echo "STEP 1: INSTALLING DEPENDENCIES"
echo "================================================================"

# Check if Kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
    echo "✓ Kaggle CLI installed"
else
    echo "✓ Kaggle CLI already installed"
fi

# Check if other dependencies are installed
echo "Checking Python dependencies..."
pip install -q pandas pyyaml datasets transformers peft bitsandbytes accelerate

echo ""
echo "================================================================"
echo "STEP 2: SETUP KAGGLE"
echo "================================================================"

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  Kaggle setup required!"
    echo ""
    echo "Please:"
    echo "1. Go to https://kaggle.com/account"
    echo "2. Click 'Create API Token'"
    echo "3. Download kaggle.json"
    echo "4. Run:"
    echo "   mkdir -p ~/.kaggle"
    echo "   cp /path/to/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    read -p "Press Enter when Kaggle is configured..."
else
    echo "✓ Kaggle configured"
fi

echo ""
echo "================================================================"
echo "STEP 3: DOWNLOAD DATASETS"
echo "================================================================"

echo "Downloading real ecommerce datasets..."
echo "This will download ~8GB of data..."
python scripts/download_datasets.py --dataset all

echo ""
echo "================================================================"
echo "STEP 4: PROCESS DATASETS"
echo "================================================================"

echo "Processing datasets into training format..."
python scripts/prepare_real_data.py

echo ""
echo "================================================================"
echo "STEP 5: LOGIN TO HUGGINGFACE"
echo "================================================================"

# Check if logged in to HF
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to HuggingFace..."
    huggingface-cli login
else
    echo "✓ Already logged in to HuggingFace"
fi

echo ""
echo "================================================================"
echo "STEP 6: TRAIN MODEL"
echo "================================================================"

echo ""
echo "Training configuration:"
echo "  Model: Llama-3.1-8B-Instruct"
echo "  Method: LoRA + QLoRA (4-bit)"
echo "  Data: Real ecommerce datasets"
echo "  Epochs: 3"
echo "  Batch size: 4"
echo "  Estimated time: 3 hours"
echo ""

read -p "Start training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 1
fi

echo ""
echo "Starting training..."
python scripts/train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/elm-real-data \
    --config config/training_config.yaml

echo ""
echo "================================================================"
echo "STEP 7: TEST MODEL"
echo "================================================================"

echo "Testing the trained model..."
python scripts/inference.py \
    --model_path outputs/elm-real-data \
    --task interactive

echo ""
echo "================================================================"
echo "STEP 8: DEPLOY TO HUGGINGFACE"
echo "================================================================"

echo ""
read -p "Deploy to HuggingFace Hub? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    REPO_ID="askcatalystai/elm-real-data"
    echo "Deploying to https://huggingface.co/$REPO_ID"
    python scripts/push_to_hub.py \
        --model_path outputs/elm-real-data \
        --repo_id "$REPO_ID"
else
    echo "Skipping deployment."
fi

echo ""
echo "================================================================"
echo "✅ TRAINING COMPLETE!"
echo "================================================================"
echo ""
echo "Your model is ready!"
echo ""
echo "Model location: outputs/elm-real-data"
echo "Deployment: https://huggingface.co/$REPO_ID"
echo ""
echo "Next steps:"
echo "1. Test with: python scripts/inference.py --model_path outputs/elm-real-data"
echo "2. Evaluate: python scripts/evaluate.py --model_path outputs/elm-real-data"
echo "3. Use in your agent: Call HF API at https://api-inference.huggingface.co/models/$REPO_ID"
echo ""
echo "🎉 Happy using your ecommerce LLM!"
echo "================================================================"
