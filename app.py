"""
app.py  –  Buscador de Normativa Educativa
Motor: BM25 + búsqueda triple + reranking por Groq + respuesta profunda
"""

import json, re, time, io
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

METADATA_FILE = Path("chunks_metadata.json")
GROQ_MODEL    = "llama-3.3-70b-versatile"
CANDIDATES    = 20   # fragmentos candidatos tras BM25
FINAL_CHUNKS  = 8    # fragmentos tras reranking que van al prompt final
MAX_SOURCES   = 5

# ── Diccionarios de etiquetas y URLs ──────────────────────────────────────────
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

def get_label(fn): return DOC_LABELS.get(fn, fn.replace("_"," ").replace(".pdf",""))
def get_url(fn):   return DOC_URLS.get(fn, "#")
def tokenize(t):   return re.findall(r'\b[a-záéíóúüñ]{3,}\b', t.lower())


@st.cache_resource(show_spinner=False)
def load_bm25():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    corpus = [tokenize(m["chunk_text"]) for m in meta]
    return BM25Okapi(corpus), meta


# ── PASO 1: expansión de consulta ─────────────────────────────────────────────
def expand_query(query: str, client: Groq) -> tuple[str, str]:
    """Genera keywords jurídicas + reformulación formal."""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Eres experto en normativa educativa española (LOE, LOMLOE, decretos, órdenes, BOCYL, BOE). "
                    "Para la pregunta recibida responde SOLO con JSON con dos campos:\n"
                    "- \"keywords\": 10-18 términos técnicos jurídico-administrativos separados por espacios\n"
                    "- \"reformulation\": la pregunta reescrita en lenguaje normativo formal\n"
                    "Sin texto adicional, sin markdown."
                )},
                {"role": "user", "content": f"Pregunta: {query}"},
            ],
            temperature=0.0, max_tokens=120, timeout=15,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        data = json.loads(raw)
        return data.get("keywords",""), data.get("reformulation","")
    except Exception:
        return "", ""


# ── PASO 2: recuperación BM25 multi-consulta ──────────────────────────────────
def bm25_search(queries: list[str], bm25, meta) -> list[dict]:
    """Fusiona resultados de varias consultas tomando el score máximo."""
    best: dict[int, float] = {}
    for q in queries:
        if not q.strip(): continue
        scores = bm25.get_scores(tokenize(q))
        for idx in scores.argsort()[::-1][:CANDIDATES]:
            s = float(scores[idx])
            if idx not in best or s > best[idx]:
                best[idx] = s
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:CANDIDATES]
    results = []
    for idx, score in ranked:
        item = meta[idx].copy()
        item["score"] = score
        results.append(item)
    return results


# ── PASO 3: reranking por relevancia con Groq ─────────────────────────────────
def rerank(query: str, candidates: list[dict], client: Groq) -> list[dict]:
    """
    Pide a Groq que puntúe cada fragmento del 0 al 10 por relevancia.
    Devuelve los FINAL_CHUNKS más relevantes.
    """
    # Construir lista numerada de fragmentos (truncada para no exceder tokens)
    snippets = []
    for i, c in enumerate(candidates, 1):
        # Usamos solo los primeros 400 chars para el reranking (rápido y barato)
        snippet = c["chunk_text"][:400].replace("\n", " ")
        snippets.append(f"[{i}] {snippet}")

    prompt = (
        f"Pregunta: {query}\n\n"
        f"Fragmentos de normativa educativa:\n" + "\n\n".join(snippets) +
        f"\n\nDevuelve SOLO un JSON con una lista llamada \"ranking\" que contenga los números "
        f"de los {FINAL_CHUNKS} fragmentos MÁS relevantes para responder la pregunta, "
        f"ordenados de mayor a menor relevancia. Ejemplo: {{\"ranking\": [3,1,7,2,5,8,4,6]}}"
    )
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Eres un experto en normativa educativa española. Selecciona los fragmentos más relevantes."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0, max_tokens=80, timeout=15,
        )
        raw = re.sub(r"```json|```", "", resp.choices[0].message.content).strip()
        data = json.loads(raw)
        ranking = data.get("ranking", [])
        # Convertir a índices válidos
        selected = []
        seen = set()
        for r in ranking:
            idx = int(r) - 1
            if 0 <= idx < len(candidates) and idx not in seen:
                selected.append(candidates[idx])
                seen.add(idx)
            if len(selected) >= FINAL_CHUNKS:
                break
        # Si el reranking falla parcialmente, completar con los mejores por BM25
        if len(selected) < FINAL_CHUNKS:
            for c in candidates:
                if c not in selected:
                    selected.append(c)
                if len(selected) >= FINAL_CHUNKS:
                    break
        return selected
    except Exception:
        # Si falla el reranking, devolver los mejores por BM25
        return candidates[:FINAL_CHUNKS]


# ── PASO 4: generación de respuesta profunda ──────────────────────────────────
def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, r in enumerate(chunks, 1):
        parts.append(
            f"[FRAGMENTO {i}]\n"
            f"Documento: {get_label(r['doc_name'])} | Página: {r['page_num']}\n"
            f"{r['chunk_text']}"
        )
    return "\n\n{'='*60}\n\n".join(parts)


def ask_groq(query: str, context: str, client: Groq, retries: int = 3) -> str:
    system = """Eres un experto en normativa educativa española con dominio profundo de la LOE, LOMLOE, 
decretos y órdenes educativas de Castilla y León.

Tu tarea es responder preguntas con MÁXIMA PROFUNDIDAD Y DETALLE, basándote EXCLUSIVAMENTE en los 
fragmentos de normativa proporcionados.

INSTRUCCIONES PARA LA RESPUESTA:
1. CONTENIDO: Extrae y explica TODA la información relevante que aparezca en los fragmentos:
   - Artículos y apartados específicos con su numeración exacta
   - Plazos, fechas, porcentajes y cifras concretas
   - Condiciones, requisitos y excepciones
   - Procedimientos paso a paso cuando los haya
   - Referencias cruzadas entre normativas si aparecen

2. ESTRUCTURA: Organiza la respuesta de forma clara:
   - Usa encabezados (##) para separar aspectos distintos del tema
   - Usa listas numeradas para procedimientos o requisitos ordenados
   - Usa listas con viñetas para enumeraciones sin orden específico
   - Cita entre paréntesis el nombre corto de la norma cuando menciones algo concreto

3. RIGOR: 
   - No omitas datos relevantes presentes en los fragmentos
   - No generalices cuando hay datos específicos disponibles
   - Si hay información complementaria entre varios fragmentos, intégrala
   - Si algo no está en los fragmentos, NO lo incluyas

4. Si la respuesta no se encuentra en ningún fragmento, di exactamente:
   "No he encontrado información sobre esto en la normativa disponible."

Responde siempre en español."""

    user = (
        f"PREGUNTA: {query}\n\n"
        f"FRAGMENTOS DE NORMATIVA:\n{context}\n\n"
        "Elabora una respuesta COMPLETA Y DETALLADA con toda la información relevante "
        "que aparece en los fragmentos anteriores."
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
                max_tokens=2048,
                timeout=60,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "rate_limit" in msg.lower() or "429" in msg:
                wait = 25 * (intento + 1)
                st.warning(f"Límite de tasa de Groq. Reintentando en {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("No se pudo obtener respuesta de Groq.")


def deduplicate(results: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in results:
        k = r["doc_name"]
        if k not in seen or r["score"] > seen[k]["score"]:
            seen[k] = r
    return list(seen.values())[:MAX_SOURCES]


def limpiar():
    for k, v in [("query_text",""), ("answer",None), ("results",None)]:
        st.session_state[k] = v
    # Borrar también el widget directamente
    st.session_state["query_input"] = ""


# ── PDF ───────────────────────────────────────────────────────────────────────
_CP = HexColor("#4a3f7a"); _CL = HexColor("#7c6fae")
_CB = HexColor("#2d2244"); _CG = HexColor("#888888"); _CA = HexColor("#a78bfa")

def _styles():
    b = getSampleStyleSheet()
    def s(n,**kw): return ParagraphStyle(n, parent=b["Normal"], **kw)
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
        if not s: fl.append(Spacer(1,4)); continue
        if re.match(r'^#{1,3}\s',s):
            fl.append(Paragraph(re.sub(r'\*\*(.+?)\*\*',r'\1',re.sub(r'^#{1,3}\s*','',s)), styles["h2"]))
        elif re.match(r'^[-*•]\s',s):
            fl.append(Paragraph(f"• &nbsp;{_md(re.sub(r'^[-*•]\s+','',s))}", styles["bullet"]))
        elif re.match(r'^\d+[\.\)]\s',s):
            m = re.match(r'^(\d+[\.\)])\s+(.*)',s)
            if m: fl.append(Paragraph(f"<b>{m.group(1)}</b> &nbsp;{_md(m.group(2))}", styles["bullet"]))
            else: fl.append(Paragraph(_md(s), styles["body"]))
        else:
            fl.append(Paragraph(_md(s), styles["body"]))
    return fl

def generate_pdf(query, answer, sources):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=22*mm, rightMargin=22*mm, topMargin=22*mm, bottomMargin=22*mm)
    st2 = _styles(); story = []
    story.append(Paragraph("Buscador de Normativa Educativa", st2["title"]))
    story.append(Paragraph("Castilla y León", st2["sub"]))
    story.append(HRFlowable(width="100%", thickness=1, color=_CA, spaceAfter=10))
    story.append(Paragraph("Pregunta:", st2["lbl"]))
    story.append(Paragraph(query, st2["q"]))
    story.append(Spacer(1,6))
    story.append(Paragraph("Respuesta:", st2["lbl"]))
    story.append(Spacer(1,4))
    story.extend(_flow(answer, st2))
    story.append(Spacer(1,8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#d4c9f7"), spaceAfter=6))
    story.append(Paragraph("Fuentes consultadas:", st2["sh"]))
    for src in sources:
        story.append(Paragraph(f"• &nbsp;{src.get('label','')} — pág. {src.get('page_num','')}", st2["si"]))
    doc.build(story)
    return buf.getvalue()


# ── INTERFAZ ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Buscador de Normativa Educativa", page_icon="📚", layout="centered")

    for k,v in [("query_text",""),("answer",None),("results",None)]:
        if k not in st.session_state: st.session_state[k]=v

    st.markdown("""<style>
.stApp{background-color:#f8f6ff}
.header-box{background:linear-gradient(135deg,#d6eaff 0%,#ffe8f0 100%);border-radius:18px;
    padding:28px 32px 20px;margin-bottom:28px;box-shadow:0 2px 12px rgba(180,160,220,.13)}
.header-box h1{color:#4a3f7a;margin:0;font-size:2rem}
.answer-box{background:#fff;border-left:5px solid #a78bfa;border-radius:12px;padding:22px 26px;
    margin:18px 0 10px;box-shadow:0 2px 10px rgba(167,139,250,.10);color:#2d2244;
    font-size:1.02rem;line-height:1.75;white-space:pre-wrap}
.sources-title{color:#7c6fae;font-weight:600;font-size:.93rem;margin:20px 0 8px;
    letter-spacing:.05em;text-transform:uppercase}
.source-card{background:#f0ebff;border:1px solid #d4c9f7;border-radius:10px;padding:11px 16px;
    margin-bottom:8px;display:flex;align-items:center;gap:10px}
.source-card a{color:#5b4ba0;text-decoration:none;font-weight:500}
.source-card a:hover{text-decoration:underline}
.source-page{background:#c4b5fd;color:#2d2244;border-radius:20px;padding:2px 11px;
    font-size:.81rem;font-weight:600;white-space:nowrap;margin-left:auto}
.stTextArea textarea{border-radius:12px!important;border:1.5px solid #c4b5fd!important;
    font-size:1rem!important;background:#fdfcff!important}
div[data-testid="column"] .stButton>button{width:100%;white-space:nowrap;padding:11px 18px!important;
    font-size:1rem!important;font-weight:600!important;border-radius:10px!important;border:none!important;line-height:1.2}
div[data-testid="column"]:first-child .stButton>button{
    background:linear-gradient(135deg,#a78bfa,#f9a8d4)!important;color:white!important;
    box-shadow:0 2px 8px rgba(167,139,250,.30)!important}
div[data-testid="column"]:first-child .stButton>button:hover{opacity:.88}
div[data-testid="column"]:nth-child(2) .stButton>button{background:#ede9fe!important;color:#5b4ba0!important}
div[data-testid="column"]:nth-child(2) .stButton>button:hover{background:#ddd6fe!important}
section[data-testid="stSidebar"]{display:none!important}
[data-testid="collapsedControl"]{display:none!important}
</style>""", unsafe_allow_html=True)

    st.markdown('<div class="header-box"><h1>📚 Buscador de Normativa Educativa</h1></div>',
                unsafe_allow_html=True)

    groq_api_key = st.secrets.get("GROQ_API_KEY","")
    if not groq_api_key:
        st.error("⚠️ Clave GROQ_API_KEY no encontrada en los Secrets de Streamlit.")
        st.stop()

    if not METADATA_FILE.exists():
        st.error("Archivo `chunks_metadata.json` no encontrado.")
        return

    with st.spinner("Cargando buscador..."):
        bm25, meta = load_bm25()

    client = Groq(api_key=groq_api_key)

    query = st.text_area(
        "🔍 ¿Qué quieres consultar?",
        value=st.session_state.query_text,
        placeholder="Ej: ¿Cuáles son los criterios y el procedimiento de admisión en el primer ciclo de Infantil?",
        height=110, key="query_input",
    )

    col1, col2, col3 = st.columns([2,2,6])
    with col1: buscar = st.button("🔍 Buscar", use_container_width=True)
    with col2: limpiar_btn = st.button("🗑️ Limpiar", use_container_width=True)

    if limpiar_btn:
        limpiar(); st.rerun()

    if buscar:
        if not query.strip():
            st.warning("Escribe una pregunta antes de buscar.")
        else:
            st.session_state.query_text = query

            # Paso 1: expansión
            with st.spinner("🧠 Analizando la consulta..."):
                keywords, reformulation = expand_query(query, client)

            # Paso 2: BM25 multi-consulta
            with st.spinner("📄 Recuperando fragmentos relevantes..."):
                queries = [q for q in [query, keywords, reformulation] if q.strip()]
                candidates = bm25_search(queries, bm25, meta)

            # Paso 3: reranking
            with st.spinner("🎯 Seleccionando los fragmentos más relevantes..."):
                final_chunks = rerank(query, candidates, client)
                context = build_context(final_chunks)

            # Paso 4: respuesta profunda
            with st.spinner("🤖 Generando respuesta detallada..."):
                try:
                    answer = ask_groq(query, context, client)
                    st.session_state.answer  = answer
                    st.session_state.results = final_chunks
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    if st.session_state.answer:
        st.markdown(f'<div class="answer-box">{st.session_state.answer}</div>',
                    unsafe_allow_html=True)

        sources = deduplicate(st.session_state.results)
        st.markdown('<p class="sources-title">📄 Fuentes consultadas</p>', unsafe_allow_html=True)
        for src in sources:
            st.markdown(
                f'<div class="source-card"><span>📄</span>'
                f'<a href="{get_url(src["doc_name"])}" target="_blank">{get_label(src["doc_name"])}</a>'
                f'<span class="source-page">Pág. {src["page_num"]}</span></div>',
                unsafe_allow_html=True)

        # Botón PDF
        pdf_sources = [{"label": get_label(s["doc_name"]), "page_num": s["page_num"]}
                       for s in sources]
        pdf_bytes = generate_pdf(st.session_state.query_text,
                                 st.session_state.answer, pdf_sources)
        st.download_button("⬇️ Descargar respuesta en PDF", data=pdf_bytes,
                           file_name="respuesta_normativa.pdf", mime="application/pdf")

        with st.expander("🔬 Ver fragmentos enviados a la IA"):
            for i, r in enumerate(st.session_state.results, 1):
                st.markdown(f"**[{i}] {get_label(r['doc_name'])} – Pág. {r['page_num']}** "
                            f"*(BM25: {r['score']:.1f})*\n\n{r['chunk_text']}")
                st.divider()


if __name__ == "__main__":
    main()
