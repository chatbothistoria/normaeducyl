"""
preprocess.py
─────────────
Extrae texto de los PDFs, realiza un particionado semántico, 
calcula los embeddings y guarda un índice FAISS real.
"""

import json
import re
from pathlib import Path
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

# Configuración
PDF_DIR       = Path("Normativa_Oficial")
INDEX_FILE    = Path("faiss_index.bin")
METADATA_FILE = Path("chunks_metadata.json")
MODEL_NAME    = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def smart_chunking(text: str, max_length=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, current_chunk = [], []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        
        if current_length >= max_length:
            chunks.append(" ".join(current_chunk))
            overlap_length = 0
            overlap_chunk = []
            for w in reversed(current_chunk):
                if overlap_length + len(w) > overlap: break
                overlap_chunk.insert(0, w)
                overlap_length += len(w) + 1
            current_chunk = overlap_chunk
            current_length = sum(len(w) + 1 for w in current_chunk)
            
    if current_chunk and len(" ".join(current_chunk)) > 50:
        chunks.append(" ".join(current_chunk))
    return chunks

def build_index():
    pdf_files = sorted([p for p in PDF_DIR.glob("*.pdf")])
    if not pdf_files:
        print("❌ No se encontraron PDFs en la carpeta.")
        return

    print(f"📄 Procesando {len(pdf_files)} PDFs...")
    all_chunks, all_metadata = [], []

    for pdf_path in pdf_files:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = clean_text(page.extract_text() or "")
                    if text:
                        for chunk in smart_chunking(text):
                            all_chunks.append(chunk)
                            all_metadata.append({
                                "doc_name": pdf_path.name,
                                "page_num": i + 1,
                                "chunk_text": chunk
                            })
        except Exception as e:
            print(f"⚠️ Error en {pdf_path.name}: {e}")

    print(f"🔢 {len(all_chunks)} fragmentos extraídos. Generando IA...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print("✅ Preprocesamiento semántico completado con éxito.")

if __name__ == "__main__":
    build_index()