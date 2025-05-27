# 🧠 Retrieval-Augmented Generation (RAG) for Medical QA

This project implements a Retrieval-Augmented Generation (RAG) system for medical question answering using the [MedQuad](https://huggingface.co/datasets/lavita/MedQuAD) dataset and the `microsoft/Phi-3-mini-4k-instruct` large language model (LLM) from Hugging Face. The goal is to retrieve relevant medical context and generate accurate, grounded responses.


![image](https://github.com/user-attachments/assets/5bbdb646-0b52-46eb-bc4b-5083c7a54a15)






## 📦 Installation

```bash
pip install langchain langchain_huggingface langchain-community faiss-cpu accelerate einops transformers datasets sentence-transformers
```

## 🧪 Datasets & Embeddings

- **Dataset**: [LavitA/MedQuad](https://huggingface.co/datasets/lavita/MedQuAD)

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

- **Vector Store**: FAISS




## 🧰 Pipeline Overview

### 1. Load and Preprocess Dataset

```pyton
from datasets import load_dataset

dataset = load_dataset("LavitA/MedQuad", split="train[:20000]")
dataset = dataset.filter(lambda row: row["question"] and row["answer"])
```

### 2. Convert to LangChain Documents

```python
from langchain.docstore.document import Document

documents = [
    Document(page_content=f"Question: {row['question']}\nAnswer: {row['answer']}")
    for row in dataset
]
```

### 3. Build FAISS Vector Store

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.from_documents(documents, embedding_model)
faiss_db.save_local("faiss_medquad_index")
```


## 💡 RAG System Logic

### Retrieval Function

```python
def retriev_docs(query, k=5):
    return faiss_db.similarity_search(query, k)
```

### Context-Aware Answering

```python
def ask2_improved(query, context):
    prompt = f"""You are a medical expert. Use the following medical information to answer the question comprehensively.

Medical Information:
{context}

Question: {query}

Instructions:
- Provide a detailed, helpful answer using the medical information above
- Organize your response clearly with relevant details
- If the information is incomplete, use what's available and note any limitations
- Only say "I don't know" if the context contains no relevant information at all"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=query)
    ]
    response = chat_model.invoke(messages)
    return response.content
```

### Complete RAG Query Function

```python
def ask_rag_improved(query, k=5):
    docs = retriev_docs(query, k=k)
    cleaned_contexts = [clean_text(doc.page_content) for doc in docs]
    combined_context = "\n\n---\n\n".join(cleaned_contexts)
    safe_context = combined_context[:6000]
    return ask2_improved(query, safe_context)
```


## ✅ Example Inference

```python
query = "What causes asthma?"
result = ask_rag_improved(query)
print(result)
```

## Save the FAISS vector store

```python
faiss_db.save_local("faiss_medquad_index")
```


## 📦 Load Saved Model

```python
from langchain.vectorstores.faiss import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_db = FAISS.load_local("faiss_medquad_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
```

