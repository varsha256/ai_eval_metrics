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

bleu = evaluate.load("bleu")

predictions = dataset["response"]
references = [[ref] for ref in dataset["reference"]]

bleu_result = bleu.compute(
    predictions=predictions,
    references=references
)

print("\n=== BLEU Evaluation Result ===")
print(bleu_result)

df = pd.DataFrame(data)
df["bleu_score"] = [
    bleu.compute(
        predictions=[pred],
        references=[[ref]]
    )["bleu"]
    for pred, ref in zip(df["response"], df["reference"])
]

print("\nDetailed BLEU Breakdown:")
print(df)