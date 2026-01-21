# Ecommerce Language Model (ELM) - System Architecture

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ECOMMERCE LANGUAGE MODEL                            │
│                              (ELM) SYSTEM                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA GENERATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ prepare_data.py │
    └──────┬───────┘
           │
           ├─ Synthetic Data Generation (Current)
           │  ┌──────────────────────────────────┐
           │  │ • 4,000 training samples         │
           │  │ • Product descriptions (40%)     │
           │  │ • Conversations (30%)            │
           │  │ • Categorization (20%)          │
           │  │ • Tool calling (10%)           │
           │  └──────────────────────────────────┘
           │
           └─ Real Data Integration (Optional)
              ┌──────────────────────────────────┐
              │ • Shopify/WooCommerce API       │
              │ • Customer chat logs           │
              │ • Product reviews             │
              │ • Order history              │
              └──────────────────────────────────┘

           │
           ▼
    ┌─────────────────────────────────┐
    │  data/processed/                │
    │  ├── ecommerce_train.json       │
    │  └── ecommerce_eval.json        │
    └─────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          2. MODEL TRAINING                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   Base Model         │
                    │  Llama-3.1-8B-Instruct │
                    │  (8B parameters)     │
                    └──────────┬───────────┘
                               │
                               │ 4-bit Quantization
                               │ (QLoRA)
                               ▼
                    ┌──────────────────────┐
                    │  Quantized Model    │
                    │  (~16GB VRAM)      │
                    └──────────┬───────────┘
                               │
                               │ LoRA Training
                               │ (0.1% parameters)
                               ▼
                    ┌──────────────────────┐
                    │  LoRA Adapters      │
                    │  (8M parameters)   │
                    └──────────┬───────────┘
                               │
                               ▼
    ┌──────────────────────────────────────────────────┐
    │  outputs/elm-llama-8b/                          │
    │  ├── adapter_model.safetensors (LoRA weights)   │
    │  ├── adapter_config.json                         │
    │  ├── tokenizer.model                            │
    │  └── training_summary.json                      │
    └──────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          3. MODEL INFERENCE                                │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────┐
    │  Client (Python/JS/Any Language)                │
    │  ┌────────────────────────────────────────────┐  │
    │  │  import requests                          │  │
    │  │                                           │  │
    │  │  response = requests.post(               │  │
    │  │    "https://api-inference.huggingface.co │  │
    │  │    /askcatalystai/elm",                  │  │
    │  │    json={"inputs": "..."}                │  │
    │  │  )                                       │  │
    │  └────────────────────────────────────────────┘  │
    └────────────┬───────────────────────────────┬─────┘
                 │                               │
                 │ HTTPS Request                 │ HTTPS Response
                 │ (JSON payload)                │ (Generated text)
                 ▼                               ▼
    ┌──────────────────────────────────────────────────┐
    │             HuggingFace Inference API             │
    │  ┌────────────────────────────────────────────┐  │
    │  │  • Auto-scaling                          │  │
    │  │  • Load balancing                        │  │
    │  │  • Pay-per-use                           │  │
    │  │  • No infrastructure needed              │  │
    │  └────────────────────────────────────────────┘  │
    └────────────┬───────────────────────────────┬─────┘
                 │                               │
                 │ API Request                   │ API Response
                 ▼                               ▼
    ┌──────────────────────────────────────────────────┐
    │           HF Model Hub Deployment                 │
    │  ┌────────────────────────────────────────────┐  │
    │  │  askcatalystai/elm                       │  │
    │  │                                           │  │
    │  │  ┌──────────────┐  ┌──────────────┐     │  │
    │  │  │ Base Model   │  │ LoRA Adapter │     │  │
    │  │  │ Llama-3.1-8B │  │   (8M)      │     │  │
    │  │  └──────────────┘  └──────────────┘     │  │
    │  │                                           │  │
    │  │  ┌────────────────────────────────────┐  │  │
    │  │  │   Config Files                     │  │  │
    │  │  │  • tokenizer_config.json           │  │  │
    │  │  │  • generation_config.json          │  │  │
    │  │  │  • README.md                       │  │  │
    │  │  └────────────────────────────────────┘  │  │
    │  └────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          4. AGENT INTEGRATION                               │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │                    AI AGENT APPLICATION                  │
    │  ┌─────────────────────────────────────────────────────┐ │
    │  │  LangChain / AutoGen / Custom Agent Framework     │ │
    │  │                                                     │ │
    │  │  ┌──────────────────────────────────────────────┐ │ │
    │  │  │  1. User Query                             │ │ │
    │  │  │     "Find wireless headphones under $100"  │ │ │
    │  │  └──────────────────────────────────────────────┘ │ │
    │  │                    │                             │ │
    │  │                    ▼                             │ │
    │  │  ┌──────────────────────────────────────────────┐ │ │
    │  │  │  2. Agent calls ELM via API                │ │ │
    │  │  │     HF_API_URL = "..."                     │ │ │
    │  │  │     response = call_hf_model(query)        │ │ │
    │  │  └──────────────────────────────────────────────┘ │ │
    │  │                    │                             │ │
    │  │                    ▼                             │ │
    │  │  ┌──────────────────────────────────────────────┐ │ │
    │  │  │  3. ELM Returns Tool Call                   │ │ │
    │  │  │     <tool_call>search_products(             │ │ │
    │  │  │       query="wireless headphones",          │ │ │
    │  │  │       max_price=100                         │ │ │
    │  │  │     )</tool_call>                           │ │ │
    │  │  └──────────────────────────────────────────────┘ │ │
    │  │                    │                             │ │
    │  │                    ▼                             │ │
    │  │  ┌──────────────────────────────────────────────┐ │ │
    │  │  │  4. Agent Executes Tool                     │ │ │
    │  │  │     results = search_products(...)           │ │ │
    │  │  └──────────────────────────────────────────────┘ │ │
    │  │                    │                             │ │
    │  │                    ▼                             │ │
    │  │  ┌──────────────────────────────────────────────┐ │ │
    │  │  │  5. Agent Calls ELM Again                  │ │ │
    │  │  │     with tool results                       │ │ │
    │  │  └──────────────────────────────────────────────┘ │ │
    │  │                    │                             │ │
    │  │                    ▼                             │ │
    │  │  ┌──────────────────────────────────────────────┐ │ │
    │  │  │  6. ELM Generates Final Response           │ │ │
    │  │  │     "I found 5 great options..."           │ │ │
    │  │  └──────────────────────────────────────────────┘ │ │
    │  └─────────────────────────────────────────────────────┘ │
    └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────────────┐
    │              7. Final Response to User                   │
    │     "Here are the best wireless headphones under $100:   │
    │      1. Sony WH-CH720N - $99                           │
    │      2. JBL Live 660NC - $129                         │
    │      ..."                                                │
    └─────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          5. TOOL ECOSYSTEM                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                    Available Tools (Model Can Call):
                    ┌────────────────────────────────┐
                    │ 1. search_products()            │
                    │    - Query products             │
                    │    - Filter by price/category  │
                    │                                │
                    │ 2. get_product_info()          │
                    │    - Get product details      │
                    │    - Specifications            │
                    │                                │
                    │ 3. get_inventory()             │
                    │    - Check stock              │
                    │    - Availability              │
                    │                                │
                    │ 4. calculate_shipping()       │
                    │    - Shipping cost            │
                    │    - Delivery time            │
                    │                                │
                    │ 5. get_customer_orders()      │
                    │    - Order history            │
                    │    - Order status             │
                    │                                │
                    │ 6. initiate_return()           │
                    │    - Process returns          │
                    │    - Refund status            │
                    │                                │
                    │ 7. get_discount_eligibility() │
                    │    - Promo codes              │
                    │    - Loyalty benefits         │
                    │                                │
                    │ 8. get_similar_products()      │
                    │    - Recommendations          │
                    │    - Alternatives             │
                    └────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          6. DEPLOYMENT FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    Local Development              →           Cloud Deployment
    ┌─────────────────┐                         ┌──────────────────┐
    │  Train Model    │                         │  Push to HF Hub  │
    │  python train.py│                         │  python push.py  │
    └────────┬────────┘                         └────────┬─────────┘
             │                                           │
             ▼                                           ▼
    ┌─────────────────┐                         ┌──────────────────┐
    │  Test Locally   │                         │  HF Inference API│
    │  python infer.py│                         │  Auto-scaling   │
    └────────┬────────┘                         └────────┬─────────┘
             │                                           │
             └───────────────────┬───────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Agent Integration     │
                    │   Any Framework         │
                    │   - LangChain           │
                    │   - AutoGen             │
                    │   - Custom              │
                    └──────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          7. KEY BENEFITS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    ✓ Memory Efficient      → QLoRA: 16GB VRAM (vs 64GB full fine-tune)
    ✓ Fast Training         → 3 hours on RTX 4090
    ✓ No Infrastructure     → Hosted on HuggingFace
    ✓ Pay-per-Use          → Only pay for API calls
    ✓ Auto-scaling         → Handles any load
    ✓ Tool Integration     → 8 built-in ecommerce tools
    ✓ Easy API Access      → Standard REST API
    ✓ Language Agnostic    → Works with any agent framework


┌─────────────────────────────────────────────────────────────────────────────┐
│                          8. USAGE COST                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    Training (One-time):
    ┌──────────────────────────────────────┐
    │  RTX 4090 Cloud (3 hours)            │
    │  ≈ $5-10 (RunPod/AWS)                │
    └──────────────────────────────────────┘

    Inference (Per Use):
    ┌──────────────────────────────────────┐
    │  HuggingFace Inference API            │
    │  ≈ $0.001-0.01 per request           │
    │  (Depends on input/output length)     │
    └──────────────────────────────────────┘


```

## Component Details

### 1. **Data Generation Layer**
- **Synthetic**: Template-based generation (current)
- **Real Data**: CSV/JSON import from stores
- **Output**: JSONL format for training

### 2. **Training Layer**
- **Base Model**: Llama-3.1-8B-Instruct
- **Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (QLoRA)
- **Output**: LoRA adapters + base model

### 3. **Deployment Layer**
- **Platform**: HuggingFace Model Hub
- **API**: REST inference endpoint
- **Scaling**: Auto-scaling infrastructure
- **Cost**: Pay-per-use

### 4. **Integration Layer**
- **Protocol**: HTTPS/JSON
- **SDKs**: Python, JavaScript, cURL
- **Frameworks**: LangChain, AutoGen, Custom
- **Tools**: 8 ecommerce-specific functions

### 5. **Agent Layer**
- **Framework**: Any (LangChain, AutoGen, etc.)
- **Flow**: Query → ELM → Tool Call → Execute → Response
- **Tools**: Pre-defined ecommerce functions
- **Memory**: Conversation context maintained
