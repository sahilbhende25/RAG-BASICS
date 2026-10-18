import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY in .env")

client = OpenAI()

# 1. Our knowledge base
documents = [
    "Full-time employees receive 20 paid vacation days annually. "
    "Up to 5 unused vacation days may be carried over.",

    "Employees receive 10 paid sick days annually. "
    "Sick leave is separate from vacation leave.",

    "Employees may work remotely up to 3 days per week "
    "with manager approval."
]

# 2. Convert documents into embeddings
def embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])

document_vectors = embed(documents)


# 3. Retrieve the most relevant document chunks
def retrieve(question, top_k=2):
    query_vector = embed([question])[0]

    # Cosine similarity
    scores = (
        document_vectors @ query_vector
        / (
            np.linalg.norm(document_vectors, axis=1)
            * np.linalg.norm(query_vector)
        )
    )

    # Highest similarity first
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        (documents[i], float(scores[i]))
        for i in indices
    ]


# 4. Generate an answer using retrieved context
def answer_question(question):
    results = retrieve(question)

    context = "\n".join(
        f"- {document}" for document, score in results
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions=(
            "Answer using only the provided context. "
            "If the context does not support an answer, "
            "say you don't know."
        ),
        input=f"Context:\n{context}\n\nQuestion: {question}"
    )

    return response.output_text


# 5. Test the complete RAG pipeline
question = "Can I save unused vacation for next year?"

print("Answer:", answer_question(question))
