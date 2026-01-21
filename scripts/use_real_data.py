#!/usr/bin/env python3
"""
Example: How to use REAL ecommerce data instead of synthetic data
This script shows how to load and format real datasets for training
"""

import json
import pandas as pd
from typing import List, Dict

def load_real_product_catalog(csv_file: str) -> List[Dict]:
    """
    Load real product catalog from CSV/JSON

    Expected CSV columns:
    - name: Product name
    - price: Product price
    - category: Product category
    - description: Existing description (optional)
    - features: Comma-separated features
    - brand: Product brand
    """
    print(f"📂 Loading product catalog from {csv_file}...")

    if csv_file.endswith('.csv'):
        df = pd.read_csv(csv_file)
    else:
        with open(csv_file, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    products = []
    for _, row in df.iterrows():
        product = {
            "name": row['name'],
            "price": float(row['price']),
            "category": row['category'],
            "features": row['features'].split(',') if isinstance(row['features'], str) else [],
            "brand": row.get('brand', ''),
            "description": row.get('description', '')
        }
        products.append(product)

    print(f"   ✓ Loaded {len(products)} products")
    return products

def load_conversational_data(jsonl_file: str) -> List[Dict]:
    """
    Load real customer conversations

    Expected JSONL format:
    {"customer_query": "...", "agent_response": "...", "intent": "product_inquiry"}
    """
    print(f"📂 Loading conversations from {jsonl_file}...")

    conversations = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            conv = json.loads(line)
            conversations.append({
                "instruction": "You are a helpful ecommerce assistant.",
                "input": f"Customer: {conv['customer_query']}",
                "output": conv['agent_response'],
                "intent": conv.get('intent', '')
            })

    print(f"   ✓ Loaded {len(conversations)} conversations")
    return conversations

def load_review_data(csv_file: str) -> List[Dict]:
    """
    Load customer reviews for sentiment analysis training

    Expected CSV columns:
    - product_id
    - rating (1-5)
    - review_text
    - sentiment (positive/negative/neutral)
    """
    print(f"📂 Loading reviews from {csv_file}...")

    df = pd.read_csv(csv_file)

    reviews = []
    for _, row in df.iterrows():
        reviews.append({
            "instruction": "Analyze this customer review and provide insights.",
            "input": f"Product: {row['product_id']}\nReview: {row['review_text']}\nRating: {row['rating']}/5",
            "output": f"Sentiment: {row['sentiment']}\nKey points: [Extract important points from review]",
            "rating": row['rating'],
            "sentiment": row['sentiment']
        })

    print(f"   ✓ Loaded {len(reviews)} reviews")
    return reviews

def load_order_data(csv_file: str) -> List[Dict]:
    """
    Load order history for recommendation training

    Expected CSV columns:
    - customer_id
    - product_id
    - category
    - order_date
    - order_value
    """
    print(f"📂 Loading orders from {csv_file}...")

    df = pd.read_csv(csv_file)

    # Group by customer to create recommendation scenarios
    recommendations = []
    for customer_id, group in df.groupby('customer_id'):
        products = group['product_id'].tolist()
        categories = group['category'].unique().tolist()

        recommendations.append({
            "instruction": "Recommend products based on purchase history.",
            "input": f"Customer ID: {customer_id}\nPreviously purchased: {', '.join(products)}\nPreferred categories: {', '.join(categories)}",
            "output": f"Based on {customer_id}'s purchase history, I recommend products in {', '.join(categories)} categories...",
            "customer_id": customer_id,
            "purchase_history": products
        })

    print(f"   ✓ Created {len(recommendations)} recommendation scenarios")
    return recommendations

def create_training_dataset(
    product_catalog: List[Dict],
    conversations: List[Dict] = None,
    reviews: List[Dict] = None,
    orders: List[Dict] = None
) -> List[Dict]:
    """
    Combine all real data sources into training format
    """
    dataset = []

    # 1. Product descriptions (from real catalog)
    print("\n📝 Creating product description samples...")
    for product in product_catalog:
        data_point = {
            "instruction": "Write a compelling product description for this item.",
            "input": f"Product: {product['name']}\nPrice: ${product['price']}\nFeatures: {', '.join(product['features'])}\nCategory: {product['category']}",
            "output": f"Description for {product['name']}..."  # Use existing description or generate
        }
        dataset.append(data_point)

    # 2. Conversational data (from real conversations)
    if conversations:
        print("💬 Adding conversational samples...")
        dataset.extend(conversations)

    # 3. Review analysis (from real reviews)
    if reviews:
        print("⭐ Adding review analysis samples...")
        dataset.extend(reviews)

    # 4. Recommendations (from order history)
    if orders:
        print("🎯 Adding recommendation samples...")
        dataset.extend(orders)

    # Shuffle dataset
    import random
    random.shuffle(dataset)

    print(f"\n✅ Created dataset with {len(dataset)} samples")
    return dataset

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """
    Example: How to use real data instead of synthetic data
    """

    # Option 1: Load from CSV files
    """
    products = load_real_product_catalog("data/real_products.csv")
    conversations = load_conversational_data("data/real_conversations.jsonl")
    reviews = load_review_data("data/real_reviews.csv")
    orders = load_order_data("data/real_orders.csv")

    # Create training dataset
    dataset = create_training_dataset(
        products,
        conversations=conversations,
        reviews=reviews,
        orders=orders
    )

    # Save in the same format as synthetic data
    with open("../data/processed/ecommerce_train_real.json", 'w') as f:
        for item in dataset[:int(0.8 * len(dataset))]:
            json.dump(item, f)
            f.write('\n')

    with open("../data/processed/ecommerce_eval_real.json", 'w') as f:
        for item in dataset[int(0.8 * len(dataset)):]:
            json.dump(item, f)
            f.write('\n')

    print("✅ Real data prepared for training!")
    """

    print(__doc__)

if __name__ == "__main__":
    main()
