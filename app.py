"""
app.py  –  Buscador de Normativa Educativa
"""

import pickle, json, time
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

INDEX_FILE    = Path("tfidf_index.pkl")
METADATA_FILE = Path("chunks_metadata.json")
GROQ_MODEL    = "llama-3.3-70b-versatile"
TOP_K         = 8
MAX_SOURCES   = 4

# ── Etiquetas legibles de cada documento ───────────────────────────────────────
DOC_LABELS = {
    "01_Ley_Organica_2_2006_LOE_consolidada.pdf":            "LOE – Ley Orgánica 2/2006 de Educación (consolidada)",
    "02_Ley_Organica_3_2020_LOMLOE.pdf":                     "LOMLOE – Ley Orgánica 3/2020",
    "03_Decreto_52_2018_admision.pdf":                       "Decreto 52/2018 – Admisión",
    "04_Decreto_32_2021_modifica_Decreto_52_2018.pdf":       "Decreto 32/2021 – Modifica Decreto 52/2018",
    "05_Orden_EDU_70_2019_admision.pdf":                     "Orden EDU/70/2019 – Admisión",
    "06_Orden_EDU_1623_2021_modifica_Orden_EDU_70_2019.pdf": "Orden EDU/1623/2021 – Modifica Orden EDU/70/2019",
    "07_Resolucion_26_01_2026_admision_2ciclo_infantil_primaria.pdf": "Resolución 26/01/2026 – Admisión 2.º ciclo Infantil y Primaria",
    "08_Orden_07_02_2001_jornada_escolar.pdf":               "Orden 07/02/2001 – Jornada Escolar",
    "09_Orden_EDU_1766_2003_modifica_jornada_escolar.pdf":   "Orden EDU/1766/2003 – Modifica Jornada Escolar",
    "10_Orden_EDU_20_2014_modifica_jornada_escolar.pdf":     "Orden EDU/20/2014 – Modifica Jornada Escolar",
    "11_Orden_EDU_13_2015_intervencion_inspeccion_educativa.pdf": "Orden EDU/13/2015 – Inspección Educativa",
    "12_Real_Decreto_95_2022_ordenacion_infantil.pdf":       "Real Decreto 95/2022 – Ordenación Educación Infantil",
    "13_Decreto_37_2022_curriculo_infantil_CyL.pdf":         "Decreto 37/2022 – Currículo Infantil CyL",
    "14_Decreto_12_2008_primer_ciclo_infantil.pdf":          "Decreto 12/2008 – Primer Ciclo Infantil",
    "15_Orden_EDU_904_2011_desarrolla_Decreto_12_2008.pdf":  "Orden EDU/904/2011 – Desarrolla Decreto 12/2008",
    "16_Orden_EDU_1511_2023_modifica_Orden_EDU_904_2011.pdf":"Orden EDU/1511/2023 – Modifica Orden EDU/904/2011",
    "17_Orden_EDU_95_2022_admision_primer_ciclo_infantil.pdf":"Orden EDU/95/2022 – Admisión Primer Ciclo Infantil",
    "18_Orden_EDU_117_2023_modifica_Orden_EDU_95_2022.pdf":  "Orden EDU/117/2023 – Modifica Orden EDU/95/2022",
    "19_Resolucion_26_01_2026_admision_primer_ciclo_infantil.pdf": "Resolución 26/01/2026 – Admisión Primer Ciclo Infantil",
    "20_Orden_EDU_1063_2022_calendario_horario_primer_ciclo.pdf": "Orden EDU/1063/2022 – Calendario y Horario Primer Ciclo",
    "21_Decreto_11_2023_precios_publicos_primer_ciclo.pdf":  "Decreto 11/2023 – Precios Públicos Primer Ciclo",
    "22_Decreto_17_2024_modifica_Decreto_11_2023.pdf":       "Decreto 17/2024 – Modifica Decreto 11/2023",
    "23_Orden_EDU_593_2018_permanencia_NEE_infantil.pdf":    "Orden EDU/593/2018 – Permanencia NEE en Infantil",
    "24_Real_Decreto_157_2022_ordenacion_primaria.pdf":      "Real Decreto 157/2022 – Ordenación Educación Primaria",
    "25_Decreto_38_2022_curriculo_primaria_CyL.pdf":         "Decreto 38/2022 – Currículo Primaria CyL",
    "26_Orden_EDU_423_2024_evaluacion_promocion_primaria.pdf":"Orden EDU/423/2024 – Evaluación y Promoción Primaria",
    "27_Orden_EDU_17_2024_evaluacion_diagnostico.pdf":       "Orden EDU/17/2024 – Evaluación de Diagnóstico",
    "28_Orden_EDU_286_2016_vigencia_libros_texto.pdf":       "Orden EDU/286/2016 – Vigencia Libros de Texto",
    "29_Decreto_3_2019_Releo_Plus.pdf":                      "Decreto 3/2019 – RELEO Plus",
    "30_Orden_EDU_167_2019_Releo_Plus_bases.pdf":            "Orden EDU/167/2019 – RELEO Plus (bases)",
    "31_Orden_EDU_49_2020_modifica_Orden_EDU_167_2019.pdf":  "Orden EDU/49/2020 – Modifica Orden EDU/167/2019",
    "32_Orden_EDU_1861_2022_mejora_exito_educativo.pdf":     "Orden EDU/1861/2022 – Mejora del Éxito Educativo",
    "33_Orden_EDU_1152_2010_respuesta_educativa_NEAE.pdf":   "Orden EDU/1152/2010 – Respuesta Educativa NEAE",
    "34_Orden_EDU_371_2018_modifica_Orden_EDU_1152_2010.pdf":"Orden EDU/371/2018 – Modifica Orden EDU/1152/2010",
    "35_Resolucion_17_08_2009_adaptaciones_curriculares_significativas.pdf": "Resolución 17/08/2009 – Adaptaciones Curriculares Significativas",
    "36_Orden_EDU_865_2009_evaluacion_NEE.pdf":              "Orden EDU/865/2009 – Evaluación NEE",
    "37_Orden_EDU_1865_2004_flexibilizacion_alumnado_superdotado.pdf": "Orden EDU/1865/2004 – Flexibilización Alumnado Superdotado",
    "38_Orden_EDU_641_2012_practicum_grados_infantil_primaria.pdf": "Orden EDU/641/2012 – Prácticum Grados Infantil y Primaria",
}

# ── URL oficial de cada documento (fuentes BOE / BOCYL) ────────────────────────
DOC_URLS = {
    "01_Ley_Organica_2_2006_LOE_consolidada.pdf":            "https://www.boe.es/buscar/pdf/2006/BOE-A-2006-7899-consolidado.pdf",
    "02_Ley_Organica_3_2020_LOMLOE.pdf":                     "https://www.boe.es/boe/dias/2020/12/30/pdfs/BOE-A-2020-17264.pdf",
    "03_Decreto_52_2018_admision.pdf":                       "https://bocyl.jcyl.es/boletines/2018/12/28/pdf/BOCYL-D-28122018-2.pdf",
    "04_Decreto_32_2021_modifica_Decreto_52_2018.pdf":       "https://bocyl.jcyl.es/boletines/2021/11/29/pdf/BOCYL-D-29112021-1.pdf",
    "05_Orden_EDU_70_2019_admision.pdf":                     "https://bocyl.jcyl.es/boletines/2019/02/01/pdf/BOCYL-D-01022019-1.pdf",
    "06_Orden_EDU_1623_2021_modifica_Orden_EDU_70_2019.pdf": "https://bocyl.jcyl.es/boletines/2021/12/27/pdf/BOCYL-D-27122021-1.pdf",
    "07_Resolucion_26_01_2026_admision_2ciclo_infantil_primaria.pdf": "https://bocyl.jcyl.es/boletines/2026/02/02/pdf/BOCYL-D-02022026-21-17.pdf",
    "08_Orden_07_02_2001_jornada_escolar.pdf":               "https://bocyl.jcyl.es/boletines/2001/02/09/pdf/BOCYL-D-09022001-2.pdf",
    "09_Orden_EDU_1766_2003_modifica_jornada_escolar.pdf":   "https://bocyl.jcyl.es/boletines/2004/01/05/pdf/BOCYL-D-05012004-15.pdf",
    "10_Orden_EDU_20_2014_modifica_jornada_escolar.pdf":     "https://bocyl.jcyl.es/boletines/2014/01/28/pdf/BOCYL-D-28012014-1.pdf",
    "11_Orden_EDU_13_2015_intervencion_inspeccion_educativa.pdf": "https://bocyl.jcyl.es/boletines/2015/01/22/pdf/BOCYL-D-22012015-1.pdf",
    "12_Real_Decreto_95_2022_ordenacion_infantil.pdf":       "https://www.boe.es/boe/dias/2022/02/02/pdfs/BOE-A-2022-1654.pdf",
    "13_Decreto_37_2022_curriculo_infantil_CyL.pdf":         "https://bocyl.jcyl.es/boletines/2022/09/30/pdf/BOCYL-D-30092022-1.pdf",
    "14_Decreto_12_2008_primer_ciclo_infantil.pdf":          "https://bocyl.jcyl.es/boletines/2008/02/20/pdf/BOCYL-D-20022008-3.pdf",
    "15_Orden_EDU_904_2011_desarrolla_Decreto_12_2008.pdf":  "https://bocyl.jcyl.es/boletines/2011/07/22/pdf/BOCYL-D-22072011-1.pdf",
    "16_Orden_EDU_1511_2023_modifica_Orden_EDU_904_2011.pdf":"https://bocyl.jcyl.es/boletines/2024/01/11/pdf/BOCYL-D-11012024-2.pdf",
    "17_Orden_EDU_95_2022_admision_primer_ciclo_infantil.pdf":"https://bocyl.jcyl.es/boletines/2022/02/17/pdf/BOCYL-D-17022022-1.pdf",
    "18_Orden_EDU_117_2023_modifica_Orden_EDU_95_2022.pdf":  "https://bocyl.jcyl.es/boletines/2023/02/06/pdf/BOCYL-D-06022023-1.pdf",
    "19_Resolucion_26_01_2026_admision_primer_ciclo_infantil.pdf": "https://bocyl.jcyl.es/boletines/2026/02/02/pdf/BOCYL-D-02022026-21-19.pdf",
    "20_Orden_EDU_1063_2022_calendario_horario_primer_ciclo.pdf": "https://bocyl.jcyl.es/boletines/2022/08/24/pdf/BOCYL-D-24082022-1.pdf",
    "21_Decreto_11_2023_precios_publicos_primer_ciclo.pdf":  "https://bocyl.jcyl.es/boletines/2023/06/30/pdf/BOCYL-D-30062023-1.pdf",
    "22_Decreto_17_2024_modifica_Decreto_11_2023.pdf":       "https://bocyl.jcyl.es/boletines/2024/09/09/pdf/BOCYL-D-09092024-1.pdf",
    "23_Orden_EDU_593_2018_permanencia_NEE_infantil.pdf":    "https://bocyl.jcyl.es/boletines/2018/06/06/pdf/BOCYL-D-06062018-1.pdf",
    "24_Real_Decreto_157_2022_ordenacion_primaria.pdf":      "https://www.boe.es/boe/dias/2022/03/02/pdfs/BOE-A-2022-3296.pdf",
    "25_Decreto_38_2022_curriculo_primaria_CyL.pdf":         "https://bocyl.jcyl.es/boletines/2022/09/30/pdf/BOCYL-D-30092022-2.pdf",
    "26_Orden_EDU_423_2024_evaluacion_promocion_primaria.pdf":"https://bocyl.jcyl.es/boletines/2024/05/17/pdf/BOCYL-D-17052024-2.pdf",
    "27_Orden_EDU_17_2024_evaluacion_diagnostico.pdf":       "https://bocyl.jcyl.es/boletines/2024/01/23/pdf/BOCYL-D-23012024-1.pdf",
    "28_Orden_EDU_286_2016_vigencia_libros_texto.pdf":       "https://bocyl.jcyl.es/boletines/2016/04/15/pdf/BOCYL-D-15042016-19.pdf",
    "29_Decreto_3_2019_Releo_Plus.pdf":                      "https://bocyl.jcyl.es/boletines/2019/02/22/pdf/BOCYL-D-22022019-1.pdf",
    "30_Orden_EDU_167_2019_Releo_Plus_bases.pdf":            "https://bocyl.jcyl.es/boletines/2019/02/28/pdf/BOCYL-D-28022019-4.pdf",
    "31_Orden_EDU_49_2020_modifica_Orden_EDU_167_2019.pdf":  "https://bocyl.jcyl.es/boletines/2020/01/31/pdf/BOCYL-D-31012020-8.pdf",
    "32_Orden_EDU_1861_2022_mejora_exito_educativo.pdf":     "https://bocyl.jcyl.es/boletines/2022/12/27/pdf/BOCYL-D-27122022-1.pdf",
    "33_Orden_EDU_1152_2010_respuesta_educativa_NEAE.pdf":   "https://bocyl.jcyl.es/boletines/2010/08/13/pdf/BOCYL-D-13082010-1.pdf",
    "34_Orden_EDU_371_2018_modifica_Orden_EDU_1152_2010.pdf":"https://bocyl.jcyl.es/boletines/2018/04/12/pdf/BOCYL-D-12042018-2.pdf",
    "35_Resolucion_17_08_2009_adaptaciones_curriculares_significativas.pdf": "https://bocyl.jcyl.es/boletines/2009/08/26/pdf/BOCYL-D-26082009-19.pdf",
    "36_Orden_EDU_865_2009_evaluacion_NEE.pdf":              "https://bocyl.jcyl.es/boletines/2009/04/22/pdf/BOCYL-D-22042009-8.pdf",
    "37_Orden_EDU_1865_2004_flexibilizacion_alumnado_superdotado.pdf": "https://bocyl.jcyl.es/boletines/2004/12/17/pdf/BOCYL-D-17122004-13.pdf",
    "38_Orden_EDU_641_2012_practicum_grados_infantil_primaria.pdf": "https://bocyl.jcyl.es/boletines/2012/07/31/pdf/BOCYL-D-31072012-2.pdf",
}


def get_label(fn):
    return DOC_LABELS.get(fn, fn.replace("_", " ").replace(".pdf", ""))

def get_url(fn):
    return DOC_URLS.get(fn, "#")


@st.cache_resource(show_spinner=False)
def load_resources():
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return data["vectorizer"], data["matrix"], meta


def search(query, vectorizer, matrix, metadata):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).flatten()
    top_indices = scores.argsort()[::-1][:TOP_K]
    results = []
    for i in top_indices:
        item = metadata[i].copy()
        item["score"] = float(scores[i])
        results.append(item)
    return results


def build_context(results):
    parts = []
    for i, r in enumerate(results, 1):
        chunk = r["chunk_text"][:500]
        parts.append(
            f"[Fragmento {i}] Documento: «{get_label(r['doc_name'])}» | Página: {r['page_num']}\n{chunk}"
        )
    return "\n\n---\n\n".join(parts)


def ask_groq(query, context, api_key, retries=3):
    client = Groq(api_key=api_key)
    system = (
        "Eres un asistente experto en normativa educativa española. "
        "Tu ÚNICA fuente de información son los fragmentos de documentos oficiales del contexto. "
        "REGLAS: 1) Responde SOLO con la información de los fragmentos. "
        "2) Si la respuesta no está en los fragmentos, di exactamente: "
        "'No he encontrado información sobre esto en la normativa disponible.' "
        "3) Sé claro, preciso y bien estructurado. 4) Responde siempre en español."
    )
    user = (
        f"PREGUNTA: {query}\n\n"
        f"FRAGMENTOS DE NORMATIVA:\n{context}\n\n"
        "Responde basándote exclusivamente en los fragmentos anteriores."
    )
    for intento in range(retries):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=1024,
                timeout=30,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg.lower() or "429" in msg:
                wait = 20 * (intento + 1)
                st.warning(f"Límite de tasa de Groq. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("No se pudo obtener respuesta de Groq tras varios intentos.")


def deduplicate(results):
    seen = {}
    for r in results:
        k = r["doc_name"]
        if k not in seen or r["score"] > seen[k]["score"]:
            seen[k] = r
    return list(seen.values())[:MAX_SOURCES]


def limpiar():
    st.session_state.query_text = ""
    st.session_state.answer     = None
    st.session_state.results    = None


def main():
    st.set_page_config(
        page_title="Buscador de Normativa Educativa",
        page_icon="📚",
        layout="centered",
    )

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""
    if "answer" not in st.session_state:
        st.session_state.answer = None
    if "results" not in st.session_state:
        st.session_state.results = None

    st.markdown("""<style>
.stApp { background-color: #f8f6ff; }

.header-box {
    background: linear-gradient(135deg, #d6eaff 0%, #ffe8f0 100%);
    border-radius: 18px;
    padding: 28px 32px 20px;
    margin-bottom: 28px;
    box-shadow: 0 2px 12px rgba(180,160,220,.13);
}
.header-box h1 { color: #4a3f7a; margin: 0; font-size: 2rem; }

.answer-box {
    background: #fff;
    border-left: 5px solid #a78bfa;
    border-radius: 12px;
    padding: 22px 26px;
    margin: 18px 0 10px;
    box-shadow: 0 2px 10px rgba(167,139,250,.10);
    color: #2d2244;
    font-size: 1.02rem;
    line-height: 1.75;
    white-space: pre-wrap;
}
.sources-title {
    color: #7c6fae;
    font-weight: 600;
    font-size: .93rem;
    margin: 20px 0 8px;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.source-card {
    background: #f0ebff;
    border: 1px solid #d4c9f7;
    border-radius: 10px;
    padding: 11px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.source-card a { color: #5b4ba0; text-decoration: none; font-weight: 500; }
.source-card a:hover { text-decoration: underline; }
.source-page {
    background: #c4b5fd;
    color: #2d2244;
    border-radius: 20px;
    padding: 2px 11px;
    font-size: .81rem;
    font-weight: 600;
    white-space: nowrap;
    margin-left: auto;
}

.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #c4b5fd !important;
    font-size: 1rem !important;
    background: #fdfcff !important;
}

div[data-testid="column"] .stButton > button {
    width: 100%;
    white-space: nowrap;
    padding: 11px 18px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    cursor: pointer;
    line-height: 1.2;
}
div[data-testid="column"]:first-child .stButton > button {
    background: linear-gradient(135deg, #a78bfa, #f9a8d4) !important;
    color: white !important;
    box-shadow: 0 2px 8px rgba(167,139,250,.30) !important;
}
div[data-testid="column"]:first-child .stButton > button:hover { opacity: .88; }
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: #ede9fe !important;
    color: #5b4ba0 !important;
    box-shadow: 0 2px 6px rgba(167,139,250,.15) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover { background: #ddd6fe !important; }

/* Ocultar sidebar completamente */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }
</style>""", unsafe_allow_html=True)

    # ── Cabecera (solo título, sin subtítulo) ──
    st.markdown("""<div class="header-box">
<h1>📚 Buscador de Normativa Educativa</h1>
</div>""", unsafe_allow_html=True)

    # API key desde Secrets de Streamlit
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("⚠️ Clave GROQ_API_KEY no encontrada en los Secrets de Streamlit.")
        st.stop()

    if not INDEX_FILE.exists():
        st.error("Índice no encontrado. Asegúrate de que `tfidf_index.pkl` está en el repositorio.")
        return

    with st.spinner("Cargando índice..."):
        vectorizer, matrix, metadata = load_resources()

    query = st.text_area(
        "🔍 ¿Qué quieres consultar?",
        value=st.session_state.query_text,
        placeholder="Ej: ¿Cuáles son los criterios de admisión en el primer ciclo de Infantil?",
        height=110,
        key="query_input",
    )

    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        buscar = st.button("🔍 Buscar", use_container_width=True)
    with col2:
        limpiar_btn = st.button("🗑️ Limpiar", use_container_width=True)

    if limpiar_btn:
        limpiar()
        st.rerun()

    if buscar:
        if not query.strip():
            st.warning("Escribe una pregunta antes de buscar.")
        else:
            st.session_state.query_text = query
            with st.spinner("🔎 Buscando en la normativa..."):
                results = search(query, vectorizer, matrix, metadata)
                context = build_context(results)
            with st.spinner("🤖 Generando respuesta con llama-3.3-70b..."):
                try:
                    answer = ask_groq(query, context, groq_api_key)
                    st.session_state.answer  = answer
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"❌ Error al obtener respuesta: {e}")
                    st.info("Comprueba que la GROQ_API_KEY en los Secrets es correcta.")

    if st.session_state.answer:
        st.markdown(
            f'<div class="answer-box">{st.session_state.answer}</div>',
            unsafe_allow_html=True,
        )

        sources = deduplicate(st.session_state.results)
        st.markdown('<p class="sources-title">📄 Fuentes consultadas</p>', unsafe_allow_html=True)
        for src in sources:
            st.markdown(
                f'<div class="source-card">'
                f'<span>📄</span>'
                f'<a href="{get_url(src["doc_name"])}" target="_blank">{get_label(src["doc_name"])}</a>'
                f'<span class="source-page">Pág. {src["page_num"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🔬 Ver fragmentos recuperados (contexto enviado a la IA)"):
            for i, r in enumerate(st.session_state.results, 1):
                st.markdown(
                    f"**[{i}] {get_label(r['doc_name'])} – Pág. {r['page_num']}** "
                    f"*(puntuación: {r['score']:.3f})*\n\n{r['chunk_text']}"
                )
                st.divider()


if __name__ == "__main__":
    main()
