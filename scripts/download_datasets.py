#!/usr/bin/env python3
"""
Download real ecommerce datasets for training
Automatically downloads and extracts datasets from Kaggle and other sources
"""

import os
import subprocess
import argparse
import zipfile
import gzip
import shutil
from pathlib import Path

# Dataset configurations
DATASETS = {
    "amazon_products": {
        "kaggle_id": "saurabhbagchi/amazon-product-dataset",
        "description": "Amazon product metadata with descriptions, categories, and reviews",
        "expected_files": ["products.csv", "ratings_Electronics.csv", "ratings_Clothing.csv"],
        "output_dir": "data/raw/amazon",
        "size_mb": 5000
    },

    "ecommerce_transactions": {
        "kaggle_id": "carrie1/ecommerce-data",
        "description": "UK retailer transaction data (500K+ transactions)",
        "expected_files": ["data.csv"],
        "output_dir": "data/raw/transactions",
        "size_mb": 50
    },

    "retailrocket": {
        "kaggle_id": "retailrocket/ecommerce-dataset",
        "description": "Product catalog with events and recommendations",
        "expected_files": ["category.csv", "item_properties_part1.csv", "events.csv"],
        "output_dir": "data/raw/retailrocket",
        "size_mb": 1500
    },

    "google_reviews": {
        "kaggle_id": "thedevastator/google-local-reviews",
        "description": "Local business reviews (10M+ reviews)",
        "expected_files": ["data.csv"],
        "output_dir": "data/raw/reviews",
        "size_mb": 2000
    }
}

def check_kaggle():
    """Check if Kaggle CLI is installed"""
    try:
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        print(f"✓ Kaggle CLI installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ Kaggle CLI not found")
        print("\n📦 Installing Kaggle CLI...")
        subprocess.run(['pip', 'install', 'kaggle'], check=True)
        print("✓ Kaggle CLI installed")
        print("\n⚠️  IMPORTANT: Setup required!")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Create API token (Download kaggle.json)")
        print("3. Run: mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("5. Then re-run this script")
        return False

def download_dataset(dataset_name: str, force: bool = False):
    """Download a specific dataset"""
    if dataset_name not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False

    config = DATASETS[dataset_name]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if not force and all((output_dir / f).exists() for f in config["expected_files"]):
        print(f"✓ {dataset_name} already downloaded")
        return True

    print(f"\n📥 Downloading {dataset_name}...")
    print(f"   Description: {config['description']}")
    print(f"   Expected size: ~{config['size_mb']}MB")
    print(f"   Output directory: {output_dir}")

    # Download with Kaggle
    try:
        cmd = [
            'kaggle', 'datasets', 'download',
            '-d', config['kaggle_id'],
            '-p', str(output_dir),
            '--unzip'
        ]
        subprocess.run(cmd, check=True)
        print(f"✓ {dataset_name} downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {dataset_name}")
        print(f"   Error: {e}")
        print(f"\n💡 Alternative: Download manually from https://www.kaggle.com/datasets/{config['kaggle_id']}")
        return False

def download_all(force: bool = False):
    """Download all datasets"""
    print("=" * 60)
    print("📦 DOWNLOADING ECOMMERCE DATASETS")
    print("=" * 60)

    # Check Kaggle CLI
    if not check_kaggle():
        return False

    success_count = 0
    for dataset_name in DATASETS.keys():
        if download_dataset(dataset_name, force):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"✅ Download Summary: {success_count}/{len(DATASETS)} datasets downloaded")
    print("=" * 60)

    return success_count > 0

def verify_downloads():
    """Verify that all datasets were downloaded correctly"""
    print("\n🔍 Verifying downloads...")

    all_verified = True
    for dataset_name, config in DATASETS.items():
        output_dir = Path(config["output_dir"])
        missing_files = []

        for expected_file in config["expected_files"]:
            if not (output_dir / expected_file).exists():
                missing_files.append(expected_file)

        if missing_files:
            print(f"⚠️  {dataset_name}: Missing files - {', '.join(missing_files)}")
            all_verified = False
        else:
            # Get file size
            total_size = sum(
                (output_dir / f).stat().st_size
                for f in config["expected_files"]
                if (output_dir / f).exists()
            ) / (1024 * 1024)  # MB

            print(f"✓ {dataset_name}: {total_size:.1f}MB")

    return all_verified

def create_alternative_sources():
    """Create file with alternative data sources"""
    print("\n📝 Creating alternative sources guide...")

    guide = """# Alternative Data Sources

If Kaggle downloads fail, use these sources:

## 1. Amazon Product Data (Direct)

```bash
# Download Electronics category
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv.gz
gunzip ratings_Electronics.csv.gz

# Download Books category
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv.gz
gunzip ratings_Books.csv.gz
```

## 2. Shopify Store API

```python
import requests

# Get products from a Shopify store
store_url = "https://yourstore.myshopify.com"
api_url = f"{store_url}/admin/api/2024-01/products.json"
headers = {"X-Shopify-Access-Token": "YOUR_TOKEN"}

response = requests.get(api_url, headers=headers)
products = response.json()["products"]

# Export to CSV
import pandas as pd
df = pd.DataFrame(products)
df.to_csv("data/raw/shopify_products.csv", index=False)
```

## 3. Mock Data Generation

```python
# Use Mockaroo for realistic synthetic data
# Go to https://www.mockaroo.com/
# Create schema for products, customers, orders
# Export as CSV/JSON
```

## 4. Web Scraping (Check ToS!)

```python
import requests
from bs4 import BeautifulSoup

# Scrape a product listing page
url = "https://example-store.com/products"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract product info
products = []
for item in soup.select('.product-item'):
    products.append({
        'name': item.select('.product-title')[0].text,
        'price': item.select('.price')[0].text,
        'description': item.select('.description')[0].text
    })
```
"""

    with open("data/raw/ALTERNATIVE_SOURCES.md", 'w') as f:
        f.write(guide)

    print("   ✓ Created data/raw/ALTERNATIVE_SOURCES.md")

def main():
    parser = argparse.ArgumentParser(description="Download ecommerce datasets for training")
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()) + ['all'],
                       help='Specific dataset to download')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if exists')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing downloads')

    args = parser.parse_args()

    if args.verify:
        verify_downloads()
        return

    if args.dataset and args.dataset != 'all':
        if check_kaggle():
            download_dataset(args.dataset, args.force)
    else:
        if download_all(args.force):
            verify_downloads()
            create_alternative_sources()

    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("\nNext steps:")
    print("1. Run: python scripts/prepare_real_data.py")
    print("2. This will process and combine all datasets")
    print("3. Then: python scripts/train.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
