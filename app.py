import json, re, time, io, os
import streamlit as st
from pathlib import Path
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# --- CONFIGURACIÓN DE RUTAS ---
PDF_DIR       = Path("Normativa_Oficial")
INDEX_FILE    = Path("faiss_index.bin")
METADATA_FILE = Path("chunks_metadata.json")
GROQ_MODEL    = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# --- TUS DATOS (RELLENA CON TUS 38 DOCUMENTOS) ---
DOC_LABELS = {
    "01_Ley_Organica_2_2006_LOE_consolidada.pdf": "LOE – Ley Orgánica 2/2006 de Educación",
    # ... pega aquí el resto de tus etiquetas ...
}
DOC_URLS = {
    "01_Ley_Organica_2_2006_LOE_consolidada.pdf": "https://www.boe.es/buscar/pdf/2006/BOE-A-2006-7899-consolidado.pdf",
    # ... pega aquí el resto de tus URLs ...
}

# --- LÓGICA DE PREPROCESAMIENTO AUTOMÁTICO ---
def run_auto_preprocess():
    """Función que crea el índice si no existe."""
    st.info("🔄 Configurando el sistema por primera vez... Esto puede tardar unos minutos.")
    
    pdf_files = sorted([p for p in PDF_DIR.glob("*.pdf")])
    if not pdf_files:
        st.error(f"❌ No se encontraron PDFs en la carpeta {PDF_DIR}")
        return False

    all_chunks, all_metadata = [], []
    
    # Extracción de texto
    progress_bar = st.progress(0)
    for idx, pdf_path in enumerate(pdf_files):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    # Limpieza básica
                    text = re.sub(r'\s+', ' ', text).strip()
                    if len(text) > 100:
                        # Dividir en fragmentos sencillos
                        chunks = [text[i:i+800] for i in range(0, len(text), 650)]
                        for c in chunks:
                            all_chunks.append(c)
                            all_metadata.append({"doc_name": pdf_path.name, "page_num": i+1, "chunk_text": c})
        except: pass
        progress_bar.progress((idx + 1) / len(pdf_files))

    # Crear Embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    
    # Crear FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Guardar
    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False)
    
    st.success("✅ ¡Sistema listo!")
    return True

# --- FUNCIONES DE BÚSQUEDA Y RAG ---
@st.cache_resource
def load_system():
    if not INDEX_FILE.exists():
        if not run_auto_preprocess():
            st.stop()
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, index, meta

def semantic_search(query, model, index, meta):
    qv = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    _, indices = index.search(qv, 8)
    return [meta[i] for i in indices[0] if i != -1]

# --- INTERFAZ ---
def main():
    st.set_page_config(page_title="Buscador Normativa CyL", layout="centered")
    
    # Estilos CSS (puedes usar los mismos que tenías)
    st.markdown("<style>.stApp{background-color:#f8f6ff}</style>", unsafe_allow_html=True)

    groq_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        st.error("Configura tu clave de Groq en Secrets.")
        st.stop()

    model, index, meta = load_system()
    client = Groq(api_key=groq_key)

    st.title("📚 Buscador de Normativa Educativa")
    query = st.text_area("¿Qué quieres consultar?", height=100)

    if st.button("🔍 Buscar"):
        with st.spinner("Analizando semánticamente..."):
            results = semantic_search(query, model, index, meta)
            context = "\n\n".join([f"DOC: {r['doc_name']} Pág: {r['page_num']}\n{r['chunk_text']}" for r in results])
            
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "Eres un experto legal. Responde solo con el contexto dado."},
                    {"role": "user", "content": f"Pregunta: {query}\n\nContexto:\n{context}"}
                ],
                temperature=0.1
            )
            
            st.markdown("### Respuesta:")
            st.write(resp.choices[0].message.content)

if __name__ == "__main__":
    main()