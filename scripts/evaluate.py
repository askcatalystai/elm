#!/usr/bin/env python3
"""
Evaluation script for Ecommerce Language Model (ELM)
Tests model performance on various ecommerce tasks
"""

import os
import json
import argparse
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

# Evaluation datasets
EVAL_DATASETS = {
    "product_descriptions": {
        "description": "Product description generation quality",
        "metrics": ["relevance", "completeness", "creativity", "readability"],
        "test_cases": [
            {
                "input": "Product: Wireless Headphones, Price: $99, Features: noise canceling, 30hr battery",
                "expected_elements": ["noise canceling", "battery", "headphones", "$99"]
            },
            {
                "input": "Product: Smart Watch, Price: $249, Features: GPS, heart rate, waterproof",
                "expected_elements": ["GPS", "heart rate", "watch", "$249"]
            }
        ]
    },
    "categorization": {
        "description": "Product categorization accuracy",
        "metrics": ["accuracy"],
        "test_cases": [
            {"input": "Wireless Bluetooth Speaker", "expected": "Electronics"},
            {"input": "Cotton T-Shirt", "expected": "Clothing"},
            {"input": "Python Programming Book", "expected": "Books"},
            {"input": "Stainless Steel Water Bottle", "expected": "Home & Kitchen"},
            {"input": "Yoga Mat", "expected": "Sports"},
            {"input": "LED Desk Lamp", "expected": "Electronics"}
        ]
    },
    "conversational_commerce": {
        "description": "Quality of customer service responses",
        "metrics": ["helpfulness", "accuracy", "friendliness", "completeness"],
        "test_cases": [
            {
                "input": "I need wireless headphones under $150",
                "expected_tool_call": "search_products"
            },
            {
                "input": "Where is my order #12345?",
                "expected_tool_call": "get_order_status"
            },
            {
                "input": "I want to return my headphones",
                "expected_tool_call": "initiate_return"
            }
        ]
    },
    "tool_calling": {
        "description": "Ability to use tools correctly",
        "metrics": ["correct_format", "correct_tool", "correct_params"],
        "test_cases": [
            {
                "input": "Find laptops with 16GB RAM",
                "expected_tool": "search_products",
                "expected_params": ["query", "16GB RAM"]
            },
            {
                "input": "Check stock for iPhone 15",
                "expected_tool": "get_inventory",
                "expected_params": ["product_id"]
            }
        ]
    }
}

def load_model(model_path: str):
    """Load model for evaluation"""
    print(f"🤖 Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate model response"""
    messages = [
        {"role": "system", "content": "You are ELM, an expert ecommerce assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    response = response.split("<|end_of_signal|>")[0].strip()

    return response

def evaluate_product_descriptions(model, tokenizer, test_cases: List[Dict]) -> Dict:
    """Evaluate product description generation"""
    print("\n📝 Evaluating Product Descriptions...")
    print("-" * 60)

    results = {
        "scores": {
            "relevance": [],
            "completeness": [],
            "creativity": [],
            "readability": []
        },
        "responses": []
    }

    for i, case in enumerate(test_cases, 1):
        prompt = f"Write a compelling product description.\n\n{case['input']}"
        response = generate_response(model, tokenizer, prompt, max_length=400)

        # Manual evaluation (in practice, would use human evaluators or specialized metrics)
        # Here we check for expected elements
        relevance = check_relevance(response, case["expected_elements"])
        completeness = check_completeness(response, case["expected_elements"])
        creativity = assess_creativity(response)
        readability = assess_readability(response)

        results["scores"]["relevance"].append(relevance)
        results["scores"]["completeness"].append(completeness)
        results["scores"]["creativity"].append(creativity)
        results["scores"]["readability"].append(readability)

        results["responses"].append({
            "input": case["input"],
            "response": response,
            "relevance": relevance,
            "completeness": completeness,
            "creativity": creativity,
            "readability": readability
        })

        print(f"Test {i}:")
        print(f"  Input: {case['input'][:60]}...")
        print(f"  Response: {response[:80]}...")
        print(f"  Scores - R:{relevance:.2f} C:{completeness:.2f} Cr:{creativity:.2f} Rd:{readability:.2f}\n")

    # Calculate averages
    avg_scores = {k: sum(v) / len(v) for k, v in results["scores"].items()}

    return results, avg_scores

def check_relevance(response: str, expected_elements: List[str]) -> float:
    """Check if response contains relevant information"""
    response_lower = response.lower()
    matches = sum(1 for elem in expected_elements if elem.lower() in response_lower)
    return matches / len(expected_elements)

def check_completeness(response: str, expected_elements: List[str]) -> float:
    """Check if response is complete"""
    # Simple heuristic: longer responses tend to be more complete
    word_count = len(response.split())
    # Normalize to 0-1 scale (assuming 50+ words is good)
    return min(1.0, word_count / 50)

def assess_creativity(response: str) -> float:
    """Assess creativity of response"""
    # Check for creative language patterns
    creative_indicators = ["experience", "discover", "transform", "premium", "innovative", "perfect"]
    response_lower = response.lower()
    matches = sum(1 for indicator in creative_indicators if indicator in response_lower)
    return min(1.0, matches / len(creative_indicators))

def assess_readability(response: str) -> float:
    """Assess readability"""
    # Check for proper formatting
    has_bullets = "-" in response or "•" in response
    has_structure = "\n" in response
    avg_sentence_length = len(response.split()) / max(1, response.count("."))

    # Simple scoring
    score = 0
    if has_bullets:
        score += 0.3
    if has_structure:
        score += 0.3
    if 10 <= avg_sentence_length <= 20:
        score += 0.4

    return score

def evaluate_categorization(model, tokenizer, test_cases: List[Dict]) -> Dict:
    """Evaluate product categorization"""
    print("\n🏷️  Evaluating Product Categorization...")
    print("-" * 60)

    predictions = []
    expected = []

    for case in test_cases:
        prompt = f"Categorize this product: {case['input']}\n\nProvide only the category name."
        response = generate_response(model, tokenizer, prompt, max_length=100)

        # Extract category from response
        predicted = extract_category(response)
        predictions.append(predicted)
        expected.append(case["expected"])

        print(f"Input: {case['input']}")
        print(f"  Predicted: {predicted}")
        print(f"  Expected: {case['expected']}")
        print(f"  {'✓' if predicted == case['expected'] else '✗'}\n")

    accuracy = accuracy_score(expected, predictions)

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "expected": expected
    }

def extract_category(response: str) -> str:
    """Extract category from model response"""
    # Known categories
    categories = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Fashion"]

    response_lower = response.lower()

    # Direct match
    for cat in categories:
        if cat.lower() in response_lower:
            return cat

    # Fuzzy matching
    if any(word in response_lower for word in ["shirt", "pants", "dress", "jacket"]):
        return "Clothing"
    elif any(word in response_lower for word in ["phone", "laptop", "headphone", "watch"]):
        return "Electronics"
    elif any(word in response_lower for word in ["book", "novel", "textbook"]):
        return "Books"
    elif any(word in response_lower for word in ["pan", "pot", "towel", "mug"]):
        return "Home & Kitchen"
    elif any(word in response_lower for word in ["ball", "mat", "racket", "shoes"]):
        return "Sports"

    return "Unknown"

def evaluate_conversational_commerce(model, tokenizer, test_cases: List[Dict]) -> Dict:
    """Evaluate conversational commerce quality"""
    print("\n💬 Evaluating Conversational Commerce...")
    print("-" * 60)

    results = {
        "scores": {
            "helpfulness": [],
            "accuracy": [],
            "friendliness": [],
            "completeness": []
        },
        "responses": []
    }

    for case in test_cases:
        messages = [
            {"role": "system", "content": "You are a helpful ecommerce assistant."},
            {"role": "user", "content": case["input"]}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=400,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        response = response.split("<|end_of_signal|>")[0].strip()

        # Check for tool call
        has_tool_call = case["expected_tool_call"].lower() in response.lower()

        # Simple heuristics for quality metrics
        helpfulness = 1.0 if has_tool_call else 0.5
        accuracy = 1.0 if has_tool_call else 0.6
        friendliness = check_friendliness(response)
        completeness = check_completeness(response, [])

        results["scores"]["helpfulness"].append(helpfulness)
        results["scores"]["accuracy"].append(accuracy)
        results["scores"]["friendliness"].append(friendliness)
        results["scores"]["completeness"].append(completeness)

        results["responses"].append({
            "input": case["input"],
            "response": response,
            "has_tool_call": has_tool_call
        })

        print(f"Query: {case['input']}")
        print(f"  Response: {response[:80]}...")
        print(f"  Tool call: {'✓' if has_tool_call else '✗'}\n")

    avg_scores = {k: sum(v) / len(v) for k, v in results["scores"].items()}

    return results, avg_scores

def check_friendliness(response: str) -> float:
    """Check if response is friendly"""
    friendly_indicators = ["i'd", "i'm happy", "please", "thank", "glad", "help"]
    response_lower = response.lower()
    matches = sum(1 for indicator in friendly_indicators if indicator in response_lower)
    return min(1.0, matches / 2)  # Need at least 2 friendly indicators

def evaluate_tool_calling(model, tokenizer, test_cases: List[Dict]) -> Dict:
    """Evaluate tool calling capabilities"""
    print("\n🔧 Evaluating Tool Calling...")
    print("-" * 60)

    results = {
        "format_score": 0,
        "correct_tool_score": 0,
        "param_score": 0,
        "total": len(test_cases)
    }

    for case in test_cases:
        messages = [
            {"role": "system", "content": "You are an ecommerce assistant. Use tools to help customers."},
            {"role": "user", "content": case["input"]}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=300,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check tool call format
        has_tool_call_format = "<tool_call>" in response and "</tool_call>" in response

        # Check if correct tool is mentioned
        tool_mentioned = case["expected_tool"] in response

        # Check parameters
        params_mentioned = all(param in response for param in case["expected_params"])

        results["format_score"] += 1 if has_tool_call_format else 0
        results["correct_tool_score"] += 1 if tool_mentioned else 0
        results["param_score"] += 1 if params_mentioned else 0

        print(f"Query: {case['input']}")
        print(f"  Expected tool: {case['expected_tool']}")
        print(f"  Format: {'✓' if has_tool_call_format else '✗'}")
        print(f"  Tool: {'✓' if tool_mentioned else '✗'}")
        print(f"  Params: {'✓' if params_mentioned else '✗'}\n")

    # Convert to percentages
    results["format_score"] = (results["format_score"] / results["total"]) * 100
    results["correct_tool_score"] = (results["correct_tool_score"] / results["total"]) * 100
    results["param_score"] = (results["param_score"] / results["total"]) * 100

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ecommerce Language Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="../evaluation_results",
                        help="Directory to save evaluation results")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path '{args.model_path}' not found")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model_path)

    print("=" * 60)
    print("🎯 EVALUATING ECOMMERCE LANGUAGE MODEL")
    print("=" * 60)

    overall_results = {}

    # Evaluate each task
    for task_name, task_data in EVAL_DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name.upper()}")
        print(f"Description: {task_data['description']}")
        print("=" * 60)

        if task_name == "product_descriptions":
            results, avg_scores = evaluate_product_descriptions(
                model, tokenizer, task_data["test_cases"]
            )
            overall_results[task_name] = {
                "average_scores": avg_scores,
                "detailed_results": results
            }
            print(f"\nAverage Scores:")
            for metric, score in avg_scores.items():
                print(f"  {metric}: {score:.2f}")

        elif task_name == "categorization":
            results = evaluate_categorization(
                model, tokenizer, task_data["test_cases"]
            )
            overall_results[task_name] = results
            print(f"\nAccuracy: {results['accuracy']:.2%}")

        elif task_name == "conversational_commerce":
            results, avg_scores = evaluate_conversational_commerce(
                model, tokenizer, task_data["test_cases"]
            )
            overall_results[task_name] = {
                "average_scores": avg_scores,
                "detailed_results": results
            }
            print(f"\nAverage Scores:")
            for metric, score in avg_scores.items():
                print(f"  {metric}: {score:.2f}")

        elif task_name == "tool_calling":
            results = evaluate_tool_calling(
                model, tokenizer, task_data["test_cases"]
            )
            overall_results[task_name] = results
            print(f"\nScores:")
            print(f"  Format: {results['format_score']:.1f}%")
            print(f"  Correct Tool: {results['correct_tool_score']:.1f}%")
            print(f"  Parameters: {results['param_score']:.1f}%")

    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(overall_results, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Results saved to: {results_file}")
    print("\n📈 Summary:")

    # Print overall summary
    if "product_descriptions" in overall_results:
        pd_scores = overall_results["product_descriptions"]["average_scores"]
        print(f"  Product Descriptions: {sum(pd_scores.values())/len(pd_scores):.2f}/1.00")

    if "categorization" in overall_results:
        print(f"  Categorization: {overall_results['categorization']['accuracy']:.2%}")

    if "conversational_commerce" in overall_results:
        cc_scores = overall_results["conversational_commerce"]["average_scores"]
        print(f"  Conversational Commerce: {sum(cc_scores.values())/len(cc_scores):.2f}/1.00")

    if "tool_calling" in overall_results:
        tc = overall_results["tool_calling"]
        avg_score = (tc["format_score"] + tc["correct_tool_score"] + tc["param_score"]) / 3
        print(f"  Tool Calling: {avg_score:.1f}%")

if __name__ == "__main__":
    main()
