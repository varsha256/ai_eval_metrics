import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_ollama import ChatOllama



from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0,
#     timeout=180,
#     max_retries=3,
#     google_api_key="AIzaSyAyezwFtrCvGM_RRuUFNUV7WPu_-IVv5LM"
# )
llm = ChatOllama(
    model="qwen3:4b",
    temperature=0,
    timeout=800
)
# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004",
#     google_api_key="AIzaSyAyezwFtrCvGM_RRuUFNUV7WPu_-IVv5LM"
# )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
data = {
    "user_input": [
        "What is LangChain?",
        "Who developed ChromaDB?"
    ],
    "retrieved_contexts": [
        ["LangChain is an open-source framework for developing applications powered by large language models."],
        ["ChromaDB is a vector database developed by Chroma, an open-source embedding database company."]
    ],
    "response": [
        "LangChain is a framework for developing applications using LLMs.",
        "ChromaDB was developed by Chroma, an open-source company."
    ],
    "reference": [
        "LangChain is an open-source framework for building applications using large language models.",
        "ChromaDB was built by the Chroma team as an open-source vector database."
    ]
}

dataset = Dataset.from_dict(data)
from ragas.run_config import RunConfig

run_config = RunConfig(
    timeout=600,
    max_retries=3,
    max_wait=60,
    max_workers=1
)
results = evaluate(
    dataset=dataset,
    metrics=[
        # answer_relevancy
        faithfulness
        # context_precision
        # context_recall
    ],
    llm=llm,
    embeddings=embeddings,
    run_config=run_config,
    raise_exceptions=True
)

print("\n=== RAGAS Evaluation Results ===")
print(results)

df = results.to_pandas()
print("\nDetailed Metric Breakdown:\n", df)