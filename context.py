
# -*- coding: utf-8 -*-
#in collab

!pip install gradio langchain-community faiss-cpu sentence-transformers groq

import os
import gradio as gr

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from groq import Groq

# ------------------------------
# 1. Load & Split Documents
# ------------------------------
loader = TextLoader("notes.txt")  # replace with your file
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# ------------------------------
# 2. FAISS Vector Store
# ------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_store = FAISS.from_documents(docs, embeddings)
retriever = faiss_store.as_retriever()

# ------------------------------
# 3. Groq LLM Setup
# ------------------------------
try:
    from google.colab import userdata
    groq_api_key = userdata.get("GROQ")
except Exception:
    groq_api_key = os.getenv("GROQ")

if not groq_api_key:
    raise ValueError("⚠️ Missing GROQ_API_KEY. Please set it in environment or Colab userdata.")

groq_client = Groq(api_key=groq_api_key)

def groq_llm(prompt: str) -> str:
    """Call Groq LLM with a prompt and return text response"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# ------------------------------
# 4. Contextual Compression Retriever
# ------------------------------
class GroqCompressor(BaseDocumentCompressor):
    """LLM-based compressor using Groq"""

    def compress_documents(self, docs, query, callbacks=None):
        compressed_docs = []
        for doc in docs:
            prompt = f"""
            Extract only the most relevant sentences for the query below.
            Drop irrelevant or off-topic info.

            Query: {query}
            Document:
            {doc.page_content}
            """
            compressed_text = groq_llm(prompt).strip()

            if compressed_text and compressed_text.lower() != "i don't know":
                compressed_docs.append(Document(page_content=compressed_text))
        return compressed_docs

compressor = GroqCompressor()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# ------------------------------
# 5. RAG Pipelines
# ------------------------------
def rag_answer(query: str, mode: str = "Compressed") -> str:
    try:
        if mode == "Compressed":
            docs = compression_retriever.get_relevant_documents(query)
        else:  # Raw FAISS
            docs = retriever.get_relevant_documents(query)
    except Exception as e:
        return f"⚠️ Error retrieving documents: {e}"

    if not docs:
        return "I don't know."

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant. Use ONLY the context below to answer.
    If unsure, say "I don't know."

    Context:
    {context}

    Question: {query}
    Answer:
    """
    try:
        return groq_llm(prompt)
    except Exception as e:
        return f"⚠️ Error calling Groq LLM: {e}"

# ------------------------------
# 6. Gradio Chatbot UI
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Contextual RAG Chatbot (Groq + FAISS)")
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything...", scale=4)
        mode = gr.Dropdown(choices=["Compressed", "Raw"], value="Compressed", label="Retrieval Mode", scale=1)
    clear = gr.Button("Clear Chat")

    def user_query(user_msg, history, mode_choice):
        answer = rag_answer(user_msg, mode_choice)
        history.append((f"[{mode_choice}] {user_msg}", answer))
        return history, "", mode_choice

    msg.submit(user_query, [msg, chatbot, mode], [chatbot, msg, mode])
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch()
