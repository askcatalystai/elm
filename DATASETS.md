# Real Ecommerce Datasets for Training

A comprehensive guide to finding and using real ecommerce datasets for training your language model.

## 🎯 Top Ecommerce Datasets

### 1. **Amazon Product Dataset** ⭐ (Most Popular)

**Source**: Kaggle
**URL**: https://www.kaggle.com/datasets/saurabhbagchi/amazon-product-dataset
**Size**: 500MB - 10GB (multiple versions)
**Format**: JSON, CSV
**Contents**:
- Product metadata (title, description, price, brand, category)
- Product images URLs
- Sales rank and ratings
- Product reviews (hundreds of thousands)
- Categories: Electronics, Books, Clothing, Home, etc.

**Download**:
```bash
# Option 1: Kaggle CLI
kaggle datasets download -d saurabhbagchi/amazon-product-dataset

# Option 2: Direct download
wget https://example.com/amazon-product-dataset.zip
```

**Key Fields**:
```csv
product_id,title,description,price,brand,category,rating,review_count
B00008OE6,Apple iPhone 15 Pro,Latest iPhone with A17 chip,1199,Apple,Electronics,4.5,1523
```

### 2. **eCommerce Transaction Dataset**

**Source**: Kaggle (UCI ML Repository)
**URL**: https://www.kaggle.com/datasets/carrie1/ecommerce-data
**Size**: 500MB
**Format**: CSV
**Contents**:
- 500K+ transactions from UK retailer (2010-2011)
- Customer IDs, product IDs, quantities, prices
- Invoice details
- Customer demographics

**Download**:
```bash
kaggle datasets download -d carrie1/ecommerce-data
unzip ecommerce-data.zip
```

**Key Fields**:
```csv
InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,2010-12-01 08:26:00,2.55,17850,United Kingdom
```

### 3. **RetailRocket Product Catalog**

**Source**: Kaggle
**URL**: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
**Size**: 1.5GB
**Format**: CSV
**Contents**:
- Product catalog (120K products)
- Categories (categories.csv)
- Item properties (item_properties.csv)
- Events (clicks, add-to-cart, transactions)
- Recommendation data

**Download**:
```bash
kaggle datasets download -d retailrocket/ecommerce-dataset
```

**Key Fields**:
```csv
productid,name,categorycode,categorypath,price,availablefrom
123456,iPhone 15 Pro,/electronics/phones/smartphones,1199.99,2023-09-15
```

### 4. **Google Local Reviews Dataset**

**Source**: Kaggle
**URL**: https://www.kaggle.com/datasets/thedevastator/google-local-reviews
**Size**: 2GB
**Format**: CSV
**Contents**:
- 10M+ local business reviews
- Business categories (including ecommerce/shopping)
- Star ratings, review text, timestamps
- User information

**Download**:
```bash
kaggle datasets download -d thedevastator/google-local-reviews
```

**Key Fields**:
```csv
business_id,name,category,review_text,rating,date
1,Best Buy,Electronics Store,Great selection of products...,5,2023-06-15
```

### 5. **Amazon Product Reviews (Multichannel)**

**Source**: Stanford SNAP
**URL**: https://snap.stanford.edu/data/web-Amazon-links.html
**Size**: 20GB+
**Format**: JSON
**Contents**:
- 233M reviews, ratings, helpfulness
- 24 categories (Electronics, Books, Movies, etc.)
- Review text, timestamps, user IDs
- Product metadata

**Download**:
```bash
# Download specific category
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Electronics.csv.gz
gunzip ratings_Electronics.csv.gz
```

**Key Fields**:
```csv
reviewerID,asin,overall,reviewText,summary,unixReviewTime
A1M7CU5SVRW28O,B00008OE6I,5.0,Great product!,Love it,1234567890
```

### 6. **Shopify Store Dataset** (Simulated Real Data)

**Source**: Mockaroo / Generated from Real Shopify Stores
**URL**: https://www.mockaroo.com/
**Size**: Customizable
**Format**: CSV, JSON
**Contents**:
- Product catalogs
- Customer data
- Orders and transactions
- Custom fields

**Generate**:
```python
# Use Mockaroo to create realistic ecommerce data
# 1. Go to mockaroo.com
# 2. Create schema for products, customers, orders
# 3. Export as CSV/JSON
```

## 📊 Dataset Comparison

| Dataset | Size | Products | Reviews | Transactions | Best For |
|---------|------|----------|---------|--------------|----------|
| Amazon Product | 10GB | 2M+ | 200M+ | ❌ | Product descriptions |
| eCommerce Transactions | 500MB | 10K+ | ❌ | 500K+ | Customer behavior |
| RetailRocket | 1.5GB | 120K+ | ❌ | 1M+ | Recommendations |
| Google Reviews | 2GB | 1M+ | 10M+ | ❌ | Sentiment analysis |
| Amazon Reviews | 20GB | 10M+ | 233M+ | ❌ | Reviews & ratings |

## 🔧 Data Preparation Scripts

### Script 1: Process Amazon Product Data

```python
import pandas as pd
import json

def process_amazon_products(csv_file: str) -> list:
    """
    Convert Amazon product CSV to training format
    """
    print(f"📂 Processing {csv_file}...")

    df = pd.read_csv(csv_file)

    training_data = []
    for _, row in df.iterrows():
        if pd.isna(row['title']) or pd.isna(row['description']):
            continue

        data_point = {
            "instruction": "Write a compelling product description.",
            "input": f"Product: {row['title']}\nPrice: ${row.get('price', 'N/A')}\nBrand: {row.get('brand', 'N/A')}\nCategory: {row.get('category', 'N/A')}",
            "output": row['description'][:500]  # Use actual description
        }
        training_data.append(data_point)

    print(f"   ✓ Created {len(training_data)} samples")
    return training_data

# Usage
data = process_amazon_products('amazon_products.csv')
```

### Script 2: Process eCommerce Transactions

```python
import pandas as pd

def process_transactions(csv_file: str) -> list:
    """
    Convert transaction data to recommendation training data
    """
    df = pd.read_csv(csv_file)

    # Group by customer
    customer_purchases = df.groupby('CustomerID')['Description'].apply(list).to_dict()

    training_data = []
    for customer_id, products in customer_purchases.items():
        data_point = {
            "instruction": "Recommend products based on purchase history.",
            "input": f"Customer {customer_id} purchased: {', '.join(products[:10])}",
            "output": "Based on your purchase history, I recommend..."
        }
        training_data.append(data_point)

    return training_data

# Usage
data = process_transactions('ecommerce_transactions.csv')
```

### Script 3: Process Reviews for Sentiment Analysis

```python
import pandas as pd

def process_reviews(csv_file: str) -> list:
    """
    Convert review data to sentiment training data
    """
    df = pd.read_csv(csv_file)

    training_data = []
    for _, row in df.iterrows():
        if pd.isna(row['reviewText']) or pd.isna(row['overall']):
            continue

        sentiment = "positive" if row['overall'] >= 4 else "negative" if row['overall'] <= 2 else "neutral"

        data_point = {
            "instruction": "Analyze this product review and provide insights.",
            "input": f"Product: {row.get('asin', 'N/A')}\nReview: {row['reviewText'][:500]}\nRating: {row['overall']}/5",
            "output": f"Sentiment: {sentiment}\nKey points: Extract insights from review"
        }
        training_data.append(data_point)

    return training_data

# Usage
data = process_reviews('amazon_reviews.csv')
```

## 📥 Step-by-Step Download Guide

### Step 1: Install Kaggle CLI

```bash
# Install kaggle
pip install kaggle

# Setup credentials
mkdir -p ~/.kaggle
# Download your API key from kaggle.com/account
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Download Datasets

```bash
# Amazon Product Dataset
kaggle datasets download -d saurabhbagchi/amazon-product-dataset -p data/raw/
unzip data/raw/amazon-product-dataset.zip -d data/raw/

# eCommerce Transactions
kaggle datasets download -d carrie1/ecommerce-data -p data/raw/
unzip data/raw/ecommerce-data.zip -d data/raw/

# RetailRocket
kaggle datasets download -d retailrocket/ecommerce-dataset -p data/raw/
unzip data/raw/ecommerce-dataset.zip -d data/raw/
```

### Step 3: Process and Combine

```python
import json
import os

# Process all datasets
amazon_products = process_amazon_products('data/raw/amazon_products.csv')
transactions = process_transactions('data/raw/ecommerce_data.csv')
reviews = process_reviews('data/raw/amazon_reviews.csv')

# Combine all data
all_data = amazon_products + transactions + reviews

# Save
os.makedirs('data/processed', exist_ok=True)

with open('data/processed/ecommerce_train_real.json', 'w') as f:
    for item in all_data[:int(0.8 * len(all_data))]:
        json.dump(item, f)
        f.write('\n')

with open('data/processed/ecommerce_eval_real.json', 'w') as f:
    for item in all_data[int(0.8 * len(all_data)):]:
        json.dump(item, f)
        f.write('\n')

print(f"✅ Created {len(all_data)} real training samples!")
```

## 🎯 Recommended Dataset Mix

For best results, use this combination:

```
Total: 50,000 samples

1. Amazon Product Dataset (30,000 samples)
   - Product descriptions (15,000)
   - Product Q&A (10,000)
   - Category classification (5,000)

2. eCommerce Transactions (10,000 samples)
   - Purchase recommendations (5,000)
   - Customer behavior (3,000)
   - Cart abandonment (2,000)

3. Amazon Reviews (10,000 samples)
   - Sentiment analysis (5,000)
   - Review summarization (3,000)
   - Feature extraction (2,000)
```

## 📋 Data Quality Checklist

Before training, ensure your data:

- ✅ Has no missing critical fields (title, description, category)
- ✅ Is in the correct format (CSV/JSON)
- ✅ Contains diverse products (not just one category)
- ✅ Has been cleaned (remove duplicates, invalid entries)
- ✅ Follows the training format (instruction/input/output)
- ✅ Includes tool-calling examples (conversational data)

## 🚀 Quick Start

```bash
# 1. Download datasets
python scripts/download_datasets.py

# 2. Process and combine
python scripts/prepare_real_data.py

# 3. Train model
python scripts/train.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/elm-real-data

# 4. Test
python scripts/inference.py --model_path outputs/elm-real-data --task interactive
```

## 💡 Pro Tips

1. **Start Small**: Test with 10K samples before using all 50K
2. **Mix Synthetic + Real**: Use 70% real data, 30% synthetic for coverage
3. **Domain-Specific**: Focus on your product category (electronics, fashion, etc.)
4. **Clean Data**: Remove HTML tags, fix encoding issues
5. **Augment Data**: Use back-translation, paraphrasing for variety

## 🔗 More Resources

- **Kaggle Datasets**: https://www.kaggle.com/datasets?search=ecommerce
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/datasets.php?format=all&task=all&area=&numAtt=&numInst=&type=&sort=nameUp&view=table
- **Google Dataset Search**: https://datasetsearch.research.google.com/
- **AWS Open Data**: https://aws.amazon.com/opendata/
- **UCL Datasets**: https://github.com/ucl/ecco (Ecommerce Customer Conversations)

---

**Happy training with real data! 🎉**
