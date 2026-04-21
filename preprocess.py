"""
preprocess.py
─────────────
Extrae texto de los 38 PDFs, construye el índice TF-IDF y guarda
los archivos necesarios para la app.

Uso:  python preprocess.py
"""

import json
import pickle
import re
from pathlib import Path
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

PDF_DIR       = Path("Normativa_Oficial")
INDEX_FILE    = Path("tfidf_index.pkl")
METADATA_FILE = Path("chunks_metadata.json")
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 120


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(str(pdf_path))
        return [
            (i + 1, clean_text(page.extract_text() or ""))
            for i, page in enumerate(reader.pages)
            if (page.extract_text() or "").strip()
        ]
    except Exception as e:
        print(f"    ⚠️  Error leyendo {pdf_path.name}: {e}")
        return []


def build_index():
    pdf_files = sorted([p for p in PDF_DIR.glob("*.pdf") if not p.name.startswith("01_test")])
    print(f"📄  {len(pdf_files)} PDFs encontrados.\n")

    all_chunks, all_metadata = [], []

    for pdf_path in pdf_files:
        print(f"  → {pdf_path.name}")
        for page_num, page_text in extract_pages(pdf_path):
            for chunk in split_text(page_text):
                all_chunks.append(chunk)
                all_metadata.append({
                    "doc_name":   pdf_path.name,
                    "page_num":   page_num,
                    "chunk_text": chunk,
                })

    print(f"\n🔢  Fragmentos generados: {len(all_chunks)}")
    print("⚙️   Construyendo índice TF-IDF...")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(all_chunks)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix}, f)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"✅  Índice guardado en '{INDEX_FILE}' ({INDEX_FILE.stat().st_size // 1024} KB)")
    print(f"✅  Metadatos guardados en '{METADATA_FILE}'")


if __name__ == "__main__":
    build_index()
