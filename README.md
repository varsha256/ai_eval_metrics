# AI Evaluation Metrics Playground

A practical AI Quality Engineering project demonstrating how to evaluate LLM and RAG applications using industry-standard evaluation metrics including BLEU, ROUGE, BERTScore, and RAGAS.

## Overview

This project provides a unified framework to measure the quality of AI-generated responses against reference answers and retrieved context.

The framework helps answer key questions:

* Is the generated answer relevant to the user's question?
* Is the answer grounded in the retrieved context?
* Is the answer hallucinating information?
* How semantically similar is the generated answer to the expected answer?
* How effective is the retrieval layer in a RAG system?

## Metrics Implemented

### 1. BLEU (Bilingual Evaluation Understudy)

Measures n-gram overlap between generated and reference answers.

**Use Cases**

* Machine Translation
* Text Generation

**Interpretation**

* Higher score indicates greater lexical similarity.
* Sensitive to wording changes.

---

### 2. ROUGE

Measures overlap between generated and reference answers.

Implemented variants:

* ROUGE-1
* ROUGE-2
* ROUGE-L

**Use Cases**

* Summarization
* Question Answering

**Interpretation**

* Higher score indicates better content coverage.

---

### 3. BERTScore

Measures semantic similarity using transformer embeddings.

**Use Cases**

* LLM Evaluation
* Semantic Similarity
* Question Answering

**Interpretation**

* Captures meaning rather than exact wording.
* More robust than BLEU and ROUGE for LLM applications.

---

### 4. RAGAS Metrics

Designed specifically for RAG (Retrieval Augmented Generation) systems.

#### Answer Relevancy

Measures whether the generated answer addresses the user's question.

#### Faithfulness

Measures whether the generated answer is supported by the retrieved context.

#### Context Precision

Measures the relevance of retrieved context.

#### Context Recall

Measures whether the retrieved context contains all information required to answer the question.

---

### 5. Hallucination Score

Derived from Faithfulness.

```text
Hallucination Score = 1 - Faithfulness
```

Interpretation:

| Score     | Risk   |
| --------- | ------ |
| 0.0 - 0.1 | Low    |
| 0.1 - 0.3 | Medium |
| > 0.3     | High   |

---

## Evaluation Architecture

```text
User Question
      │
      ▼
Retrieved Context
      │
      ▼
Generated Answer
      │
      ├────────► BLEU
      ├────────► ROUGE
      ├────────► BERTScore
      └────────► RAGAS
                    │
                    ├─ Answer Relevancy
                    ├─ Faithfulness
                    ├─ Context Precision
                    └─ Context Recall
```

---

## Installation

```bash
git clone <repository-url>

cd ai-eval-metrics

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Running Evaluation

```bash
python evaluate.py
```

---

## Sample Output

```text
BLEU Score: 0.82

ROUGE-1: 0.91
ROUGE-2: 0.84
ROUGE-L: 0.88

BERTScore F1: 0.94

Answer Relevancy: 0.92
Faithfulness: 0.96
Context Precision: 0.94
Context Recall: 0.89

Hallucination Score: 0.04
```

---

## AI Quality Dashboard

The project can generate visualizations including:

* Overall Quality Score
* BLEU vs ROUGE Comparison
* BERTScore Trend
* Faithfulness Trend
* Hallucination Trend
* RAGAS Metrics Dashboard
* Retrieval Quality Analysis

---

## Recommended Quality Thresholds

| Metric              | Recommended |
| ------------------- | ----------- |
| BLEU                | > 0.75      |
| ROUGE-L             | > 0.80      |
| BERTScore F1        | > 0.90      |
| Faithfulness        | > 0.90      |
| Answer Relevancy    | > 0.85      |
| Context Precision   | > 0.85      |
| Context Recall      | > 0.80      |
| Hallucination Score | < 0.10      |

---

## Future Enhancements

* DeepEval Integration
* Bias Detection
* Toxicity Detection
* Drift Monitoring
* Prompt Regression Testing
* Agentic Workflow Evaluation
* Multi-Agent Quality Metrics
* Cost and Latency Tracking
* LLM Benchmark Comparison

---

## Target Audience

* AI Quality Engineers
* QA Engineers exploring AI Testing
* LLM Engineers
* RAG Developers
* GenAI Researchers
* AI Platform Teams

---

## Tech Stack

* Python
* RAGAS
* Hugging Face Evaluate
* BERTScore
* Sentence Transformers
* LangChain
* Gemini / Ollama
* Pandas
* Matplotlib

---


