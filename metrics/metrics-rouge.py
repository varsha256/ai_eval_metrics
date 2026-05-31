import pandas as pd
import evaluate
from datasets import Dataset

data = {
    "user_input": [
        "What is LangChain?",
        "Who developed ChromaDB?"
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

rouge = evaluate.load("rouge")

results = rouge.compute(
    predictions=dataset["response"],
    references=dataset["reference"]
)

print("\n=== ROUGE Results ===")
print(results)