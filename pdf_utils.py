# pdf_utils.py

from sentence_transformers import SentenceTransformer
import faiss
import fitz  # PyMuPDF

def process_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return chunks, index

def search_chunks(query, index, chunks, top_k=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]
