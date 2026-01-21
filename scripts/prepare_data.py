#!/usr/bin/env python3
"""
Generate and prepare ecommerce training data for fine-tuning Llama-3.1-8B-Instruct
Focuses on tool-calling scenarios for ecommerce applications
"""

import json
import random
import os
from typing import List, Dict

# Create output directories
os.makedirs("../data/raw", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)

# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def generate_product_description_data(num_samples: int = 1000) -> List[Dict]:
    """Generate product description training data"""
    products = [
        {
            "category": "Electronics",
            "items": [
                {"name": "Wireless Bluetooth Headphones", "price": 99, "features": ["noise canceling", "30hr battery", "wireless", "premium sound"]},
                {"name": "Smart Watch", "price": 249, "features": ["heart rate monitor", "GPS", "waterproof", "7-day battery"]},
                {"name": "Laptop Stand", "price": 45, "features": ["adjustable height", "ergonomic", "foldable", "aluminum"]},
                {"name": "USB-C Hub", "price": 39, "features": ["7 ports", "4K HDMI", "fast charging", "compact"]}
            ]
        },
        {
            "category": "Home & Kitchen",
            "items": [
                {"name": "Coffee Maker", "price": 89, "features": ["12-cup", "programmable", "thermal carafe", "auto shut-off"]},
                {"name": "Air Fryer", "price": 129, "features": ["6-quart", "digital display", "pre-set programs", "easy clean"]},
                {"name": "Memory Foam Pillow", "price": 35, "features": ["cooling gel", "ergonomic support", "hypoallergenic", "washable"]},
                {"name": "LED Desk Lamp", "price": 55, "features": ["adjustable arm", "USB charging port", "dimmable", "eye-caring"]}
            ]
        },
        {
            "category": "Fashion",
            "items": [
                {"name": "Running Shoes", "price": 120, "features": ["lightweight", "breathable mesh", "cushioned sole", "durable"]},
                {"name": "Denim Jacket", "price": 75, "features": ["classic fit", "distressed", "premium denim", "versatile"]},
                {"name": "Yoga Pants", "price": 45, "features": ["stretchy fabric", "high-waisted", "moisture-wicking", "squat-proof"]},
                {"name": "Leather Wallet", "price": 65, "features": ["genuine leather", "RFID blocking", "slim design", "multiple slots"]}
            ]
        }
    ]

    templates = [
        {
            "instruction": "Write a compelling product description for this item.",
            "input_template": "Product: {name}\nPrice: ${price}\nFeatures: {features}\nTarget audience: {audience}\nBrand tone: {tone}",
            "output_template": "{hook}\n\n{product_name} - {selling_point}\n\n{description}\n\n{features_formatted}\n\n{benefits}\n\n{call_to_action}"
        }
    ]

    hooks = [
        "Experience the difference with",
        "Transform your daily routine with",
        "Discover the power of",
        "Upgrade to",
        "Meet your new favorite"
    ]

    selling_points = [
        "premium quality meets innovative design",
        "the perfect blend of style and functionality",
        "engineered for performance and built to last",
        "where comfort meets innovation",
        "designed for those who demand excellence"
    ]

    tones = ["Professional and authoritative", "Friendly and approachable", "Luxurious and premium", "Casual and modern"]
    audiences = ["busy professionals", "tech enthusiasts", "fitness lovers", "home decorators", "fashion-forward individuals"]

    data = []
    for _ in range(num_samples):
        category = random.choice(products)
        product = random.choice(category["items"])
        template = random.choice(templates)

        features_text = ", ".join(product["features"])
        audience = random.choice(audiences)
        tone = random.choice(tones)

        input_text = template["input_template"].format(
            name=product["name"],
            price=product["price"],
            features=features_text,
            audience=audience,
            tone=tone
        )

        hook = random.choice(hooks)
        features_formatted = "✓ " + "\n✓ ".join(product["features"])

        description = f"This {product['name'].lower()} delivers exceptional {product['category'].lower()} experience with cutting-edge features."
        benefits = f"Perfect for {audience} who value quality and performance. Backed by our satisfaction guarantee."

        call_to_action = "Order now and enjoy fast shipping and easy returns!"

        output_text = f"{hook} {product['name']}!\n\n{product['name']} - {random.choice(selling_points)}\n\n{description}\n\n**Key Features:**\n{features_formatted}\n\n**Why Choose This Product?**\n{benefits}\n\n**Special Offer:** {call_to_action}"

        data.append({
            "instruction": template["instruction"],
            "input": input_text,
            "output": output_text
        })

    return data

def generate_conversational_commerce_data(num_samples: int = 1500) -> List[Dict]:
    """Generate conversational commerce training data with tool-calling scenarios"""

    scenarios = [
        {
            "name": "Product Inquiry",
            "tools": ["get_product_info", "search_products", "get_product_reviews"],
            "conversations": [
                {
                    "instruction": "You are a helpful ecommerce assistant. A customer is asking about a product.",
                    "input": "Customer: Hi, I'm looking for a good wireless headphone under $150 with noise canceling",
                    "tools": ["search_products(query='wireless headphone noise canceling under $150')"],
                    "output": "I'd be happy to help you find the perfect wireless headphones! Let me search for options with noise canceling under $150.\n\n<tool_call>search_products(query='wireless headphone noise canceling under $150')</tool_call>\n\nGreat news! I found several excellent options for you. Here are the top recommendations:\n\n1. **Sony WH-CH720N** - $149\n   - Active noise canceling\n   - 35-hour battery life\n   - Comfortable fit for long listening sessions\n   - Quick Charge: 3 min charge = 1 hour playback\n\n2. **JBL Live 660NC** - $129\n   - Adaptive noise canceling\n   - 50-hour battery (with ANC off)\n   - JBL Signature Sound\n   - Multi-point connection\n\nWould you like me to show you more details about any of these, or do you have specific preferences (brand, color, special features)?"
                }
            ]
        },
        {
            "name": "Order Status",
            "tools": ["get_order_status", "track_shipment"],
            "conversations": [
                {
                    "instruction": "You are a customer service assistant. Help customers check their order status.",
                    "input": "Customer: Can you tell me where my order #12345 is?",
                    "tools": ["get_order_status(order_id='12345')"],
                    "output": "I'd be happy to check on your order for you! Let me look up the status of order #12345.\n\n<tool_call>get_order_status(order_id='12345')</tool_call>\n\nGood news! Your order #12345 is currently **in transit**. Here's the latest update:\n\n📦 **Current Status:** Out for Delivery\n📅 **Estimated Delivery:** Today by 8 PM\n🚚 **Tracking Number:** 1Z999AA10123456789\n\n**Delivery Details:**\n- Carrier: UPS Ground\n- Shipped from: Los Angeles, CA\n- Last Scan: Local facility at 9:30 AM\n\nYou'll receive a text/email notification once it's out for delivery. You can also track it in real-time using the tracking number above.\n\nIs there anything else I can help you with regarding this order?"
                }
            ]
        },
        {
            "name": "Product Recommendation",
            "tools": ["get_customer_history", "search_products", "get_product_compatibility"],
            "conversations": [
                {
                    "instruction": "You are a personal shopping assistant. Provide personalized recommendations.",
                    "input": "Customer: I bought a MacBook Pro last month. What accessories would you recommend?",
                    "tools": ["get_customer_history(customer_id='{customer_id}')", "get_product_compatibility(product_id='macbook_pro')"],
                    "output": "Great choice on the MacBook Pro! Let me check your purchase history and find the perfect accessories for you.\n\n<tool_call>get_customer_history(customer_id='{customer_id}')</tool_call>\n<tool_call>get_product_compatibility(product_id='macbook_pro')</tool_call>\n\nBased on your recent MacBook Pro purchase, here are my top accessory recommendations:\n\n**Must-Haves:**\n1. **USB-C Hub** ($39) - Essential for connecting devices, HDMI, SD cards\n2. **Laptop Stand** ($45) - Ergonomic setup, better posture\n3. **Protective Case** ($35) - Keep your laptop safe\n\n**Nice-to-Have:**\n4. **Wireless Mouse** ($29) - More comfortable than trackpad\n5. **Screen Cleaner Kit** ($12) - Keep your display pristine\n\n**Productivity:**\n6. **External SSD 1TB** ($89) - Extra storage and backup\n7. **Laptop Sleeve** ($25) - For travel protection\n\nWould you like details on any of these? I can also check current pricing and availability!"
                }
            ]
        },
        {
            "name": "Return/Exchange",
            "tools": ["get_return_policy", "initiate_return"],
            "conversations": [
                {
                    "instruction": "You are a customer service representative. Handle return and exchange requests.",
                    "input": "Customer: I want to return my headphones - they're too big for my head",
                    "tools": ["get_return_policy(product_id='{product_id}')"],
                    "output": "I understand you'd like to return the headphones. I'm sorry they didn't fit well!\n\n<tool_call>get_return_policy(product_id='{product_id}')</tool_call>\n\nNo problem at all! Here are your return options:\n\n**Return Details:**\n✅ **Return Window:** 30 days (purchased 15 days ago - you're within timeframe)\n✅ **Condition:** Original packaging and accessories required\n✅ **Refund Method:** Original payment method (Credit Card ending in 4242)\n\n**Return Options:**\n\n1. **Return & Refund**\n   - Full refund: $99.00\n   - Refund processed: 5-7 business days\n   - Return shipping: Free\n\n2. **Exchange for Different Size**\n   - I can check if we have a smaller size available\n   - Free exchange shipping\n   - Same product, better fit\n\nWhich option would you prefer? If you'd like to exchange, what color/style were you looking for?"
                }
            ]
        }
    ]

    data = []
    for scenario in scenarios:
        for conv in scenario["conversations"]:
            # Generate multiple variations of each conversation
            for _ in range(5):
                data.append({
                    "instruction": conv["instruction"],
                    "input": conv["input"],
                    "tools": conv["tools"],
                    "output": conv["output"]
                })

    return data

def generate_categorization_data(num_samples: int = 800) -> List[Dict]:
    """Generate product categorization training data"""

    categories = {
        "Electronics": ["smartphone", "laptop", "tablet", "headphones", "smartwatch", "camera", "speaker", "charger"],
        "Clothing": ["t-shirt", "jeans", "dress", "jacket", "shoes", "hat", "sweater", "pants"],
        "Home & Kitchen": ["blender", "coffee maker", "toaster", "pan", "knife", "cutting board", "mug", "towel"],
        "Books": ["fiction", "non-fiction", "textbook", "magazine", "comic", "biography", "cookbook", "journal"],
        "Sports": ["bicycle", "treadmill", "dumbbells", "yoga mat", "basketball", "tennis racket", "running shoes", "swimsuit"]
    }

    data = []
    for _ in range(num_samples):
        category = random.choice(list(categories.keys()))
        product = random.choice(categories[category])

        # Generate variations
        variations = [
            f"product category for {product}",
            f"what category is {product}",
            f"classify this item: {product}",
            f"category of {product}?",
            f"what type of product is {product}"
        ]

        instruction = random.choice(variations)
        input_text = f"Item: {product}"

        output = f"The {product} belongs to the **{category}** category."

        data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })

    return data

def generate_tool_calling_examples(num_samples: int = 1000) -> List[Dict]:
    """Generate specific tool-calling examples for ecommerce"""

    tool_definitions = {
        "search_products": {
            "description": "Search for products by query, price range, category, or features",
            "parameters": {"query": "search terms", "min_price": "minimum price", "max_price": "maximum price", "category": "product category"}
        },
        "get_product_info": {
            "description": "Get detailed information about a specific product",
            "parameters": {"product_id": "unique product identifier"}
        },
        "get_inventory": {
            "description": "Check stock levels for a product",
            "parameters": {"product_id": "unique product identifier", "location": "warehouse location (optional)"}
        },
        "calculate_shipping": {
            "description": "Calculate shipping cost and delivery time",
            "parameters": {"weight": "package weight in lbs", "destination": "shipping address or zip code", "shipping_method": "standard, express, overnight"}
        },
        "get_customer_orders": {
            "description": "Retrieve a customer's order history",
            "parameters": {"customer_id": "unique customer identifier", "status": "filter by status (optional)"}
        },
        "initiate_return": {
            "description": "Start a return process for an item",
            "parameters": {"order_id": "order number", "product_id": "item to return", "reason": "return reason"}
        },
        "get_discount_eligibility": {
            "description": "Check if customer qualifies for discounts or promotions",
            "parameters": {"customer_id": "unique customer identifier", "promo_code": "promotional code (optional)"}
        },
        "get_similar_products": {
            "description": "Find products similar to a given item",
            "parameters": {"product_id": "base product to compare", "limit": "number of similar products to return"}
        }
    }

    examples = []

    # Generate examples for each tool
    for tool_name, tool_info in tool_definitions.items():
        for _ in range(num_samples // len(tool_definitions)):
            # Customer query
            customer_query = generate_customer_query(tool_name)
            instruction = f"Use the appropriate tool to answer this customer question: {customer_query}"
            input_text = f"Customer Query: {customer_query}"

            # Generate tool call
            tool_call = f"<tool_call>{tool_name}({generate_tool_parameters(tool_name)})</tool_call>"

            # Generate response
            response = f"I'll help you with that! Let me use the {tool_name} tool to get the information you need.\n\n{tool_call}\n\n[Tool result would appear here]\n\nBased on the tool response, here's what I found for you..."

            examples.append({
                "instruction": instruction,
                "input": input_text,
                "tools": [tool_call],
                "output": response
            })

    return examples

def generate_customer_query(tool_name: str) -> str:
    """Generate realistic customer queries for different tools"""
    queries = {
        "search_products": [
            "Find wireless headphones under $100",
            "Show me laptops with at least 16GB RAM",
            "I'm looking for running shoes in size 10",
            "What wireless earbuds are available?"
        ],
        "get_product_info": [
            "Tell me more about product ABC123",
            "What are the specifications of the MacBook Pro?",
            "Show me details for the Sony WH-1000XM5"
        ],
        "get_inventory": [
            "Is the iPhone 15 Pro in stock?",
            "Check if size M blue shirt is available",
            "How many units of this product do you have?"
        ],
        "calculate_shipping": [
            "How much would shipping cost to New York?",
            "What's the shipping cost for this 5lb package to California?",
            "Can you calculate delivery time to Miami?"
        ],
        "get_customer_orders": [
            "Show me my order history",
            "What's the status of my recent orders?",
            "List all my pending orders"
        ],
        "initiate_return": [
            "I want to return item from order #12345",
            "Start a return for my recent purchase",
            "Process a return for a defective product"
        ]
    }

    return random.choice(queries.get(tool_name, ["Help me with my order"]))

def generate_tool_parameters(tool_name: str) -> str:
    """Generate realistic tool parameters"""
    param_templates = {
        "search_products": "query='wireless headphones', max_price=100, category='Electronics'",
        "get_product_info": "product_id='ABC123'",
        "get_inventory": "product_id='XYZ789', location='warehouse_1'",
        "calculate_shipping": "weight=2.5, destination='10001', shipping_method='standard'",
        "get_customer_orders": "customer_id='cust_12345'",
        "initiate_return": "order_id='ord_12345', product_id='prod_789', reason='defective'",
        "get_discount_eligibility": "customer_id='cust_12345', promo_code='SAVE20'",
        "get_similar_products": "product_id='ABC123', limit=5"
    }

    return param_templates.get(tool_name, "")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("🚀 Generating Ecommerce Training Data...")
    print("=" * 60)

    # Generate all datasets
    print("\n1. Generating Product Description Data...")
    product_desc_data = generate_product_description_data(1000)
    print(f"   ✓ Generated {len(product_desc_data)} samples")

    print("\n2. Generating Conversational Commerce Data...")
    conv_commerce_data = generate_conversational_commerce_data(1500)
    print(f"   ✓ Generated {len(conv_commerce_data)} samples")

    print("\n3. Generating Product Categorization Data...")
    categorization_data = generate_categorization_data(800)
    print(f"   ✓ Generated {len(categorization_data)} samples")

    print("\n4. Generating Tool-Calling Examples...")
    tool_calling_data = generate_tool_calling_examples(1000)
    print(f"   ✓ Generated {len(tool_calling_data)} samples")

    # Combine all data
    all_data = product_desc_data + conv_commerce_data + categorization_data + tool_calling_data
    print(f"\n📊 Total samples: {len(all_data)}")

    # Shuffle data
    random.shuffle(all_data)

    # Split into train/eval (80/20)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    print(f"   📚 Training set: {len(train_data)} samples")
    print(f"   📝 Evaluation set: {len(eval_data)} samples")

    # Save datasets
    train_file = "../data/processed/ecommerce_train.json"
    eval_file = "../data/processed/ecommerce_eval.json"

    print(f"\n💾 Saving to {train_file}...")
    with open(train_file, 'w') as f:
        for item in train_data:
            json.dump(item, f)
            f.write('\n')

    print(f"💾 Saving to {eval_file}...")
    with open(eval_file, 'w') as f:
        for item in eval_data:
            json.dump(item, f)
            f.write('\n')

    print("\n✅ Data preparation complete!")
    print("\nNext steps:")
    print("1. Run: python scripts/train.py")
    print("2. Monitor training with: tensorboard --logdir outputs/logs")
    print("3. Evaluate with: python scripts/evaluate.py")

if __name__ == "__main__":
    main()
