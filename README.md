# 🧠 LangChain Fundamentals — From Basics to Real AI Systems

A structured, hands-on journey through **LangChain**, covering core concepts step by step and evolving into real-world AI patterns like agents, tools, and memory.

---

## 🚀 What This Project Covers

This repository is a **progressive learning path**, not just random scripts.

### 🔹 1. LLM Basics

* Direct model invocation
* Token + cost tracking
* Deterministic vs creative outputs

### 🔹 2. Prompt Engineering

* Dynamic prompt templates
* Reusable prompt patterns

### 🔹 3. Chains (LCEL - Modern)

* Simple chains
* Sequential chains (manual + explicit control)
* Composition using `|` operator

### 🔹 4. Agents (Modern LangChain)

* Tool-based reasoning
* Wikipedia + Search APIs
* Calculator integration
* Multi-step reasoning flow

### 🔹 5. Memory Systems (Modern Approach)

| Type           | Description             | File Location                      |
| -------------- | ----------------------- | --------------------------------- |
| In-memory      | Basic chat history      | `memory/memory_seq_chain.py`      |
| File-based     | Persistent local memory | `memory/file/file_memory_chain.py` |
| Redis          | Multi-user, fast memory | `llm_call/redis_memory_chain.py`   |
| Vector DB      | Semantic retrieval      | `memory/vector_db/vector_memory_chain.py` |
| Sliding Window | Token-efficient memory  | `memory/sliding_window/sliding_window_chain.py` |

---

## 🧱 Project Structure

```text
langchain_basics/
│
├── agent/                  # Agents (tools + reasoning)
│   ├── agent_sliding_window/  # Agent with sliding memory
│   ├── simple_agent_google_search.py  # Google Search API agent
│   └── simple_agent_wiki.py  # Wikipedia agent
│
├── chain/                  # Chains (simple + sequential)
│   ├── sequence/            # Sequential chain implementations
│   │   ├── base_seq_chain.py    # Modern sequential chain
│   │   └── simple_seq_chain.py  # Simple sequence chain
│   └── simple_chain.py     # Basic LCEL chain
│
├── memory/                 # Memory implementations
│   ├── file/               # File-based persistent memory
│   ├── redis/              # Redis memory (in llm_call/)
│   ├── sliding_window/     # Token-efficient sliding window
│   ├── vector_db/          # Semantic vector memory
│   └── memory_seq_chain.py # Basic memory sequence
│
├── prompt_template/        # Prompt examples
│   └── simple_prompt_template.py
│
├── llm_call/              # Basic LLM operations
│   ├── redis_memory_chain.py  # Redis memory chain
│   ├── simple_llm_call.py     # Direct LLM call
│   └── test_call.py           # OpenAI API test
│
├── create_model/           # LLM factory
│   └── create_model.py
│
├── utils/                  # Helpers + schemas
│   ├── langchain_helper.py    # Utility functions
│   └── RestaurantOutput.py     # Pydantic model
│
├── app_ui/                 # Streamlit app
│   └── restaurant_generator.py
│
├── global_examples/        # Comprehensive examples
│   └── langchain_basics.py
│
├── requirements.txt        # Clean dependencies
├── pyproject.toml          # Project config
└── uv.lock                 # Reproducible environment
```

---

## ⚙️ Setup

### 1. Clone repo

```bash
git clone <your-repo>
cd langchain_basics
```

### 2. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Using uv (recommended):

```bash
uv sync
```

### 3. Set environment variables

Create `.env`:

```env
OPENAI_API_KEY=your_api_key
SEARCHAPI_API_KEY=your_searchapi_key  # For Google Search agent
```

### 4. Optional: Redis setup

For Redis memory examples:

```bash
# Install Redis locally or use Docker
docker run -d -p 6379:6379 redis:latest
```

---

## ▶️ Run Examples

### 🔹 LLM Basics

```bash
# Direct LLM call
python llm_call/simple_llm_call.py

# OpenAI API test
python llm_call/test_call.py
```

### 🔹 Prompt Templates

```bash
python prompt_template/simple_prompt_template.py
```

### 🔹 Chains

```bash
# Simple LCEL chain
python chain/simple_chain.py

# Sequential chains
python chain/sequence/simple_seq_chain.py
python chain/sequence/base_seq_chain.py
```

### 🔹 Memory Systems

```bash
# Basic memory sequence
python memory/memory_seq_chain.py

# File-based memory
python memory/file/file_memory_chain.py

# Sliding window memory
python memory/sliding_window/sliding_window_chain.py

# Vector database memory
python memory/vector_db/vector_memory_chain.py

# Redis memory
python llm_call/redis_memory_chain.py
```

### 🔹 Agents

```bash
# Wikipedia agent
python agent/simple_agent_wiki.py

# Google Search agent
python agent/simple_agent_google_search.py

# Agent with sliding memory
python agent/agent_sliding_window/agent_with_sliding_memory.py
```

### 🔹 Comprehensive Examples

```bash
# All-in-one examples
python global_examples/langchain_basics.py
```

### 🔹 Streamlit App

```bash
streamlit run app_ui/restaurant_generator.py
```

---

## 🧠 Key Learnings

### Core Concepts
* **Modern LangChain** uses **LCEL (| operator)** instead of old chains
* **Memory** is just **message passing**, not magic
* **Agents** = **LLM + tools + reasoning loop**
* **Token control** is critical (sliding window pattern)

### Architecture Patterns
* Real systems combine:
  * short-term memory (window)
  * long-term memory (vector DB)
* **Factory pattern** for consistent LLM configuration
* **Helper utilities** for token tracking and reusable prompts

### Best Practices
* Use **structured outputs** (Pydantic models)
* Implement **token tracking** for cost control
* Choose **appropriate memory type** based on use case
* Leverage **LCEL** for composable chains

---

## 🔮 What’s Next

This project sets the foundation for:

* LangGraph (stateful AI workflows)
* Multi-agent systems
* Production-grade AI apps

---

## 💥 Final Thought

> This is not just a basic repo —
> it’s a blueprint for building real AI systems.

---
