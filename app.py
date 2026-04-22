"""
app.py  –  Buscador de Normativa Educativa (100% Semántico)
Motor: FAISS (SentenceTransformers) + Generación con Groq
"""

import json, re, time, io
import streamlit as st
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

INDEX_FILE    = Path("faiss_index.bin")
METADATA_FILE = Path("chunks_metadata.json")
GROQ_MODEL    = "llama-3.3-70b-versatile"
FINAL_CHUNKS  = 8
MAX_SOURCES   = 5
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# (Mantén aquí tus diccionarios DOC_LABELS y DOC_URLS exactamente igual que en tu versión original)
DOC_LABELS = {
    "01_Ley_Organica_2_2006_LOE_consolidada.pdf": "LOE – Ley Orgánica 2/2006 de Educación",
    # ... añade el resto de tus documentos aquí ...
}
DOC_URLS = {
    "01_Ley_Organica_2_2006_LOE_consolidada.pdf": "https://www.boe.es/buscar/pdf/2006/BOE-A-2006-7899-consolidado.pdf",
    # ... añade el resto de tus URLs aquí ...
}

def get_label(fn): return DOC_LABELS.get(fn, fn.replace("_"," ").replace(".pdf",""))
def get_url(fn):   return DOC_URLS.get(fn, "#")

@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Carga el modelo de lenguaje, el índice vectorial y los metadatos."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(str(INDEX_FILE))
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, index, meta

def semantic_search(query: str, model, index, meta, top_k=FINAL_CHUNKS) -> list[dict]:
    """Convierte la pregunta a vector y busca los fragmentos más cercanos en FAISS."""
    query_vector = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    # Buscar en FAISS
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(meta):
            results.append(meta[idx])
    return results

def build_context(chunks: list[dict]) -> str:
    sep = "\n\n" + "=" * 60 + "\n\n"
    parts = []
    for i, r in enumerate(chunks, 1):
        parts.append(
            f"[FRAGMENTO {i}]\n"
            f"Documento: {get_label(r['doc_name'])} | Página: {r['page_num']}\n"
            f"{r['chunk_text']}"
        )
    return sep.join(parts)

def ask_groq(query: str, context: str, client: Groq, retries: int = 3) -> str:
    system = """Eres un experto en normativa educativa española con dominio profundo de la LOE, LOMLOE, 
y decretos de Castilla y León. Tu tarea es responder con MÁXIMA PROFUNDIDAD Y DETALLE basándote 
EXCLUSIVAMENTE en los fragmentos proporcionados.

INSTRUCCIONES:
1. Extrae artículos, cifras, plazos, porcentajes y fechas concretas.
2. Estructura con markdown (##, **, listas).
3. Integra información de varios fragmentos.
4. Si no está en los fragmentos di: "No he encontrado información sobre esto en la normativa disponible."
Responde en español."""

    user = f"PREGUNTA: {query}\n\nFRAGMENTOS DE NORMATIVA:\n{context}"
    
    for intento in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 25 * (intento + 1)
                st.warning(f"Límite de Groq. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("No se pudo obtener respuesta de Groq.")

def deduplicate(results: list[dict]) -> list[dict]:
    seen = {}
    for r in results:
        k = r["doc_name"]
        if k not in seen:
            seen[k] = r
    return list(seen.values())[:MAX_SOURCES]

def limpiar():
    st.session_state.query_text = ""
    st.session_state.answer = None
    st.session_state.results = None
    st.session_state.clear_counter += 1

# ── FUNCIONES DE GENERACIÓN DE PDF (Mantenidas de tu original) ──
_CP = HexColor("#4a3f7a"); _CL = HexColor("#7c6fae"); _CB = HexColor("#2d2244"); _CG = HexColor("#888888"); _CA = HexColor("#a78bfa")

def _pdf_styles():
    b = getSampleStyleSheet()
    def s(n, **kw): return ParagraphStyle(n, parent=b["Normal"], **kw)
    return {
        "title":  s("t",  fontSize=17, textColor=_CP, fontName="Helvetica-Bold", spaceAfter=2),
        "sub":    s("su", fontSize=9,  textColor=_CG, spaceAfter=8),
        "lbl":    s("lb", fontSize=11, textColor=_CP, fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=2),
        "q":      s("q",  fontSize=11, textColor=_CB, fontName="Helvetica-Oblique", spaceAfter=6),
        "h2":     s("h2", fontSize=11, textColor=_CP, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3),
        "body":   s("bo", fontSize=10, textColor=_CB, leading=15, spaceAfter=3),
        "bullet": s("bu", fontSize=10, textColor=_CB, leading=14, leftIndent=12, spaceAfter=2),
        "sh":     s("sh", fontSize=10, textColor=_CL, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3),
        "si":     s("si", fontSize=9,  textColor=_CP, leading=13, leftIndent=10, spaceAfter=2),
    }

def _md(t): return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)

def _flow(answer, styles):
    fl = []
    for line in answer.split("\n"):
        s = line.strip()
        if not s: fl.append(Spacer(1, 4)); continue
        if re.match(r'^#{1,3}\s', s):
            fl.append(Paragraph(re.sub(r'\*\*(.+?)\*\*', r'\1', re.sub(r'^#{1,3}\s*', '', s)), styles["h2"]))
        elif re.match(r'^[-*•]\s', s):
            fl.append(Paragraph(f"• &nbsp;{_md(re.sub(r'^[-*•]\s+', '', s))}", styles["bullet"]))
        elif re.match(r'^\d+[\.\)]\s', s):
            m = re.match(r'^(\d+[\.\)])\s+(.*)', s)
            if m: fl.append(Paragraph(f"<b>{m.group(1)}</b> &nbsp;{_md(m.group(2))}", styles["bullet"]))
            else: fl.append(Paragraph(_md(s), styles["body"]))
        else:
            fl.append(Paragraph(_md(s), styles["body"]))
    return fl

def generate_pdf(query, answer, sources):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=22*mm, rightMargin=22*mm, topMargin=22*mm, bottomMargin=22*mm)
    st2 = _pdf_styles(); story = []
    story.append(Paragraph("Buscador de Normativa Educativa", st2["title"]))
    story.append(Paragraph("Castilla y León", st2["sub"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_CA, spaceAfter=10))
    story.append(Paragraph("Pregunta:", st2["lbl"]))
    story.append(Paragraph(query, st2["q"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Respuesta:", st2["lbl"]))
    story.append(Spacer(1, 4))
    story.extend(_flow(answer, st2))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#d4c9f7"), spaceAfter=6))
    story.append(Paragraph("Fuentes consultadas:", st2["sh"]))
    for src in sources:
        story.append(Paragraph(f"• &nbsp;{src.get('label','')} — pág. {src.get('page_num','')}", st2["si"]))
    doc.build(story)
    return buf.getvalue()

# ── INTERFAZ STREAMLIT ──
def main():
    st.set_page_config(page_title="Buscador de Normativa", page_icon="📚", layout="centered")

    for k, v in [("query_text",""), ("answer",None), ("results",None), ("clear_counter",0)]:
        if k not in st.session_state: st.session_state[k] = v

    st.markdown("""<style>
    .stApp{background-color:#f8f6ff}
    .header-box{background:linear-gradient(135deg,#d6eaff 0%,#ffe8f0 100%);border-radius:18px;padding:28px 32px 20px;margin-bottom:28px;box-shadow:0 2px 12px rgba(180,160,220,.13)}
    .header-box h1{color:#4a3f7a;margin:0;font-size:2rem}
    .sources-title{color:#7c6fae;font-weight:600;font-size:.93rem;margin:20px 0 8px;letter-spacing:.05em;text-transform:uppercase}
    .source-card{background:#f0ebff;border:1px solid #d4c9f7;border-radius:10px;padding:11px 16px;margin-bottom:8px;display:flex;align-items:center;gap:10px}
    .source-card a{color:#5b4ba0;text-decoration:none;font-weight:500}
    .source-card a:hover{text-decoration:underline}
    .source-page{background:#c4b5fd;color:#2d2244;border-radius:20px;padding:2px 11px;font-size:.81rem;font-weight:600;white-space:nowrap;margin-left:auto}
    .stTextArea textarea{border-radius:12px!important;border:1.5px solid #c4b5fd!important;font-size:1rem!important;background:#fdfcff!important}
    .answer-wrapper{background:#fff;border-left:5px solid #a78bfa;border-radius:12px;padding:22px 26px;margin:18px 0 10px;box-shadow:0 2px 10px rgba(167,139,250,.10);color:#2d2244;font-size:1.02rem;line-height:1.75}
    div[data-testid="column"] .stButton>button{width:100%;white-space:nowrap;padding:11px 18px!important;font-size:1rem!important;font-weight:600!important;border-radius:10px!important;border:none!important}
    div[data-testid="column"]:first-child .stButton>button{background:linear-gradient(135deg,#a78bfa,#f9a8d4)!important;color:white!important;box-shadow:0 2px 8px rgba(167,139,250,.30)!important}
    div[data-testid="column"]:nth-child(2) .stButton>button{background:#ede9fe!important;color:#5b4ba0!important}
    </style>""", unsafe_allow_html=True)

    st.markdown('<div class="header-box"><h1>📚 Buscador de Normativa Educativa</h1></div>', unsafe_allow_html=True)

    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("⚠️ Clave GROQ_API_KEY no encontrada en los Secrets.")
        st.stop()

    if not INDEX_FILE.exists():
        st.error("⚠️ Índice FAISS no encontrado. Ejecuta preprocess.py primero.")
        st.stop()

    with st.spinner("Cargando cerebro semántico..."):
        model, index, meta = load_rag_system()
    client = Groq(api_key=groq_api_key)

    query = st.text_area("🔍 ¿Qué quieres consultar?", value=st.session_state.query_text, height=110, key=f"qi_{st.session_state.clear_counter}")

    col1, col2, col3 = st.columns([2, 2, 6])
    with col1: buscar = st.button("🔍 Buscar")
    with col2: limpiar_btn = st.button("🗑️ Limpiar")

    if limpiar_btn:
        limpiar()
        st.rerun()

    if buscar and query.strip():
        st.session_state.query_text = query
        with st.spinner("🧠 Buscando normativas semánticamente..."):
            final_chunks = semantic_search(query, model, index, meta)
            context = build_context(final_chunks)
        
        with st.spinner("🤖 Generando respuesta..."):
            try:
                answer = ask_groq(query, context, client)
                st.session_state.answer = answer
                st.session_state.results = final_chunks
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.answer:
        st.markdown('<div class="answer-wrapper">', unsafe_allow_html=True)
        st.markdown(st.session_state.answer)
        st.markdown('</div>', unsafe_allow_html=True)

        sources = deduplicate(st.session_state.results)
        st.markdown('<p class="sources-title">📄 Fuentes consultadas</p>', unsafe_allow_html=True)
        for src in sources:
            st.markdown(f'<div class="source-card"><span>📄</span><a href="{get_url(src["doc_name"])}" target="_blank">{get_label(src["doc_name"])}</a><span class="source-page">Pág. {src["page_num"]}</span></div>', unsafe_allow_html=True)

        pdf_sources = [{"label": get_label(s["doc_name"]), "page_num": s["page_num"]} for s in sources]
        st.download_button("⬇️ Descargar respuesta en PDF", data=generate_pdf(st.session_state.query_text, st.session_state.answer, pdf_sources), file_name="respuesta_normativa.pdf", mime="application/pdf")

        with st.expander("🔬 Ver fragmentos enviados a la IA"):
            for i, r in enumerate(st.session_state.results, 1):
                st.markdown(f"**[{i}] {get_label(r['doc_name'])} – Pág. {r['page_num']}**\n\n{r['chunk_text']}")
                st.divider()

if __name__ == "__main__":
    main()