import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import csv, os, json, textwrap, time
import numpy as np
from fpdf import FPDF

# =============================================================================
# CONFIGURACIÓN CENTRAL
# Cambia estos valores para ajustar el comportamiento sin tocar el resto.
# =============================================================================
GROQ_MODEL_PRINCIPAL = "llama-3.3-70b-versatile"  # 1.000 req/día gratis
GROQ_MODEL_RAPIDO    = "llama-3.1-8b-instant"      # 14.400 req/día gratis
MAX_TOKENS_RESPUESTA = 1200   # tokens máximos para la respuesta principal
MAX_TOKENS_RAPIDO    = 380    # tokens para expansión ortográfica y re-ranking
MAX_CONSULTAS_SESION = 30     # máximo de consultas por usuario y sesión
MAX_CHARS_PREGUNTA   = 500    # longitud máxima de la pregunta
MATCH_THRESHOLD_ALTO = 0.55   # umbral semántico (primer intento)
MATCH_THRESHOLD_BAJO = 0.42   # umbral semántico (segundo intento si no hay resultados)
MATCH_COUNT          = 6      # fragmentos máximos a recuperar
HISTORIAL_TURNOS     = 3      # turnos previos que se incluyen en el contexto del LLM

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Normativa Educativa CyL",
    page_icon="📚",
    layout="centered",
)

# =============================================================================
# SESSION STATE — valores por defecto
# =============================================================================
_DEFAULTS: dict = {
    "historial_completo":  [],
    "ultima_pregunta":     None,
    "ultima_respuesta":    None,
    "ultimas_fuentes":     [],
    "consultas_sesion":    0,
    "confirmar_borrar":    False,
    "feedback_pendiente":  False,
    "feedback_pregunta":   None,
    "feedback_respuesta":  None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =============================================================================
# PDF — GENERACIÓN CON ENCODING CORRECTO
# Los caracteres españoles (á é í ó ú ñ) están en Latin-1 y FPDF los soporta.
# Lo que fallaba eran los caracteres tipográficos que insertan los LLMs:
# comillas curvas, rayas largas, etc. Los sustituimos antes de codificar.
# =============================================================================
_UNICODE_FIX = {
    "\u2018": "'",  "\u2019": "'",  "\u201C": '"',  "\u201D": '"',
    "\u2013": "-",  "\u2014": "-",  "\u2022": "-",  "\u00B7": "-",
    "\u2026": "...", "\u00A0": " ", "\u00AD": "-",
}

def _limpiar(texto: str) -> str:
    for orig, repl in _UNICODE_FIX.items():
        texto = texto.replace(orig, repl)
    return texto.encode("latin-1", "replace").decode("latin-1")

def generar_pdf(lista_interacciones: list, titulo: str = "Normativa Educativa") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _limpiar(titulo), ln=True, align="C")
    pdf.ln(8)
    for item in lista_interacciones:
        pdf.set_font("Helvetica", "B", 12)
        for linea in textwrap.wrap(f"PREGUNTA: {_limpiar(item['pregunta'])}", 80):
            pdf.cell(0, 6, linea, ln=True)
        corr = item.get("pregunta_corregida", "")
        if corr and corr.strip().lower() != item["pregunta"].strip().lower():
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 5, _limpiar(f"(Corregida a: {corr})"), ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        for linea in textwrap.wrap(_limpiar(item["respuesta"]), 90):
            pdf.cell(0, 6, linea, ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 5, "FUENTES CONSULTADAS:", ln=True)
        for fuente in item.get("fuentes", []):
            for linea in textwrap.wrap(f"- {_limpiar(fuente)}", 90):
                pdf.cell(0, 5, linea, ln=True)
        pdf.ln(8)
    return bytes(pdf.output())

# =============================================================================
# CLAVES Y SERVICIOS
# =============================================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_model():
    # ⚠️ NO cambiar sin re-vectorizar los documentos en Supabase.
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data
def cargar_enlaces() -> dict:
    enlaces: dict = {}
    if os.path.exists("enlaces.csv"):
        with open("enlaces.csv", encoding="utf-8") as f:
            for i, fila in enumerate(csv.reader(f)):
                if i == 0:
                    continue
                if len(fila) >= 2:
                    enlaces[fila[0].strip()] = fila[1].strip()
    return enlaces

supabase    = init_supabase()
model       = load_model()
groq_client = Groq(api_key=GROQ_API_KEY)
enlaces     = cargar_enlaces()

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _parse_json(text: str, default: dict) -> dict:
    text = text.strip()
    if "```" in text:
        partes = text.split("```")
        text = partes[1] if len(partes) > 1 else partes[0]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text.strip())
    except Exception:
        return default

def validar_input(pregunta: str) -> tuple[bool, str]:
    if not pregunta or not pregunta.strip():
        return False, "La pregunta no puede estar vacía."
    if len(pregunta) > MAX_CHARS_PREGUNTA:
        return False, f"Pregunta demasiado larga (máximo {MAX_CHARS_PREGUNTA} caracteres)."
    patrones = ["ignore previous", "ignora las instrucciones", "system:", "</s>", "[inst]", "###"]
    if any(p in pregunta.lower() for p in patrones):
        return False, "La pregunta contiene contenido no válido."
    return True, ""

def expandir_y_corregir(pregunta: str) -> tuple[str, list[str]]:
    """
    Una llamada al modelo 8B que hace dos cosas a la vez:
      1. Corrige errores ortográficos ("coemdor" → "comedor")
      2. Genera 3 reformulaciones con vocabulario jurídico-educativo español

    Usa el modelo RAPIDO (8B, 14.400 req/día) para no consumir cuota del 70B.
    Devuelve (pregunta_corregida, [reformulacion1, reformulacion2, reformulacion3]).
    """
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_RAPIDO,
            messages=[{
                "role": "user",
                "content": (
                    "Eres un asistente especializado en normativa educativa española.\n"
                    "Dado el siguiente texto:\n"
                    "  1. Corrige todos los errores ortográficos y tipográficos\n"
                    "  2. Genera 3 reformulaciones usando terminología jurídica y educativa\n\n"
                    "Responde ÚNICAMENTE con JSON válido, sin texto adicional:\n"
                    '{"corregida": "texto corregido", '
                    '"reformulaciones": ["opcion1", "opcion2", "opcion3"]}\n\n'
                    f"Texto: {pregunta}"
                ),
            }],
            temperature=0.2,
            max_tokens=MAX_TOKENS_RAPIDO,
        )
        data = _parse_json(
            resp.choices[0].message.content,
            {"corregida": pregunta, "reformulaciones": []},
        )
        corregida       = data.get("corregida") or pregunta
        reformulaciones = [r for r in (data.get("reformulaciones") or []) if r]
        return corregida, reformulaciones
    except Exception:
        return pregunta, []

def buscar_normativa_hibrida(embedding: list, pregunta_texto: str, bloque: str) -> list:
    """
    Búsqueda HÍBRIDA: semántica (vectorial) + léxica (texto exacto).

    Semántica: similitud coseno con pgvector. Umbral adaptativo:
      intenta 0.55 primero; si no hay resultados, baja a 0.42 automáticamente.

    Léxica: coincidencia de texto con la función buscar_normativa_texto
      (ver supabase_funciones.sql). Si no existe, se ignora silenciosamente.
      Esto soluciona búsquedas por número de decreto, artículo o frase exacta.

    Los dos conjuntos se fusionan eliminando duplicados por id.
    """
    # Búsqueda semántica con umbral adaptativo
    resultados_v: list = []
    for threshold in [MATCH_THRESHOLD_ALTO, MATCH_THRESHOLD_BAJO]:
        resp = supabase.rpc("buscar_normativa", {
            "query_embedding": embedding,
            "filtro_bloque":   bloque,
            "match_threshold": threshold,
            "match_count":     MATCH_COUNT,
        }).execute()
        resultados_v = resp.data or []
        if resultados_v:
            break

    # Búsqueda léxica (falla silenciosamente si la función no existe)
    resultados_t: list = []
    try:
        resp_t = supabase.rpc("buscar_normativa_texto", {
            "query_texto":   pregunta_texto,
            "filtro_bloque": bloque,
            "match_count":   3,
        }).execute()
        resultados_t = resp_t.data or []
    except Exception:
        pass

    # Fusionar eliminando duplicados
    ids_vistos: set = set()
    combinados: list = []
    for r in resultados_v + resultados_t:
        rid = r.get("id") or (r.get("nombre_archivo", "") + str(r.get("pagina_num", "")))
        if rid not in ids_vistos:
            ids_vistos.add(rid)
            combinados.append(r)

    return combinados[:MATCH_COUNT + 2]  # margen extra para el re-ranking

def reranquear(pregunta: str, fragmentos: list) -> list:
    """
    El modelo 8B puntúa cada fragmento del 1 al 5 según su relevancia
    para responder la pregunta. Los reordena de mayor a menor.
    1 llamada al 8B (NO al 70B). Mejora la calidad del contexto
    que recibe el modelo principal.
    """
    if len(fragmentos) <= 2:
        return fragmentos
    try:
        lista_txt = "\n".join([
            f"[{i+1}] {f.get('contenido', '')[:250]}"
            for i, f in enumerate(fragmentos)
        ])
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_RAPIDO,
            messages=[{
                "role": "user",
                "content": (
                    f'Pregunta: "{pregunta}"\n\n'
                    "Puntúa del 1 al 5 la relevancia de cada fragmento "
                    "para responder esa pregunta.\n"
                    f'Responde SOLO con JSON: {{"puntuaciones": [n, n, n, ...]}}\n\n'
                    f"Fragmentos:\n{lista_txt}"
                ),
            }],
            temperature=0,
            max_tokens=80,
        )
        data = _parse_json(resp.choices[0].message.content, {"puntuaciones": []})
        punts = data.get("puntuaciones", [])
        if len(punts) == len(fragmentos):
            pares = sorted(zip(fragmentos, punts), key=lambda x: x[1], reverse=True)
            return [f for f, _ in pares]
    except Exception:
        pass
    return fragmentos

def construir_contexto_xml(fragmentos: list, enlaces_dict: dict) -> tuple[str, list, list]:
    """
    Estructura el contexto en XML etiquetado.
    Los LLMs siguen instrucciones de citación mucho más precisamente
    cuando los fragmentos tienen etiquetas claras con id, documento y página.
    """
    contexto_xml = ""
    links_screen = []
    fuentes_pdf  = []

    for i, res in enumerate(fragmentos, 1):
        nombre   = res.get("nombre_archivo", "")
        pagina   = res.get("pagina_num", "")
        score    = res.get("similarity", res.get("score", ""))
        nombre_l = nombre.replace(".pdf", "").replace("_", " ")
        score_s  = f"{score:.2f}" if isinstance(score, float) else ""

        contexto_xml += (
            f'<fragmento id="{i}" documento="{nombre_l}" '
            f'pagina="{pagina}" relevancia="{score_s}">\n'
            f'{res.get("contenido", "")}\n'
            f'</fragmento>\n\n'
        )

        url = enlaces_dict.get(nombre)
        if url:
            links_screen.append(f"[{nombre_l} (Pág. {pagina})]({url}#page={pagina})")
            fuentes_pdf.append(f"{nombre_l} (Pág. {pagina}) — {url}")
        else:
            links_screen.append(f"**{nombre_l}** (Pág. {pagina})")
            fuentes_pdf.append(f"{nombre_l} (Pág. {pagina})")

    return contexto_xml, links_screen, fuentes_pdf

def construir_mensajes(pregunta: str, contexto_xml: str) -> list:
    """
    Construye el array de mensajes con:
      - System prompt con formato obligatorio + ejemplo few-shot
      - Historial real de conversación (últimos HISTORIAL_TURNOS turnos)
      - Turno actual con el contexto XML
    """
    PROMPT_SISTEMA = """\
Eres un asesor jurídico experto en normativa educativa española \
(legislación estatal y de Castilla y León).

REGLAS ESTRICTAS:
- Responde SOLO con la información de los <fragmento> proporcionados.
- NUNCA inventes, supongas ni cites normativas que no aparezcan en el contexto.
- Si la información es insuficiente, indícalo claramente.
- Cita siempre el documento y la página a la que te refieres.

FORMATO OBLIGATORIO — usa SIEMPRE esta estructura:

**Resumen:** [respuesta directa en 2-3 frases]

**Normativa aplicable:**
[artículos y normas con referencia exacta al fragmento, documento y página]

**Implicaciones prácticas:**
[qué debe hacer o tener en cuenta el docente, equipo directivo o familia]

---
EJEMPLO DE RESPUESTA CORRECTA:

Pregunta: ¿Cuántos días de permiso tiene un docente por fallecimiento de familiar?

**Resumen:** Los docentes tienen derecho a permiso retribuido por fallecimiento \
de familiar, con duración variable según el grado de parentesco y la distancia \
al lugar del suceso.

**Normativa aplicable:**
Según el EBEP, RD Legislativo 5/2015 (fragmento 2, pág. 14), artículo 48.a): \
3 días hábiles para familiares de primer grado en la misma localidad, o 5 días \
si requiere desplazamiento. Para segundo grado: 2 días sin desplazamiento, \
4 con desplazamiento.

**Implicaciones prácticas:**
El docente comunica el permiso a la dirección aportando el certificado de \
defunción. Los días son hábiles: no cuentan fines de semana ni festivos locales."""

    mensajes: list = [{"role": "system", "content": PROMPT_SISTEMA}]

    # Historial real: últimos N turnos como pares user/assistant
    ultimos = st.session_state.historial_completo[-HISTORIAL_TURNOS:]
    for turno in ultimos:
        resp_prev = turno["respuesta"]
        if len(resp_prev) > 1200:
            resp_prev = resp_prev[:1200] + "..."
        mensajes.append({"role": "user",      "content": turno["pregunta"]})
        mensajes.append({"role": "assistant", "content": resp_prev})

    # Turno actual
    mensajes.append({
        "role": "user",
        "content": f"CONTEXTO NORMATIVO:\n{contexto_xml}\n\nPREGUNTA: {pregunta}",
    })
    return mensajes

def guardar_log(bloque, preg_orig, preg_corr, num_res, tiempo_ms, tiene_resp) -> None:
    """Logging a Supabase. Falla silenciosamente si la tabla no existe."""
    try:
        supabase.table("consultas_log").insert({
            "bloque":             bloque,
            "pregunta_original":  preg_orig[:500],
            "pregunta_corregida": preg_corr[:500],
            "num_resultados":     num_res,
            "tiempo_ms":          int(tiempo_ms),
            "tiene_respuesta":    tiene_resp,
        }).execute()
    except Exception:
        pass

def guardar_feedback(pregunta: str, respuesta: str, util: bool) -> None:
    """Feedback a Supabase. Falla silenciosamente si la tabla no existe."""
    try:
        supabase.table("feedback").insert({
            "pregunta":  pregunta[:500],
            "respuesta": respuesta[:2000],
            "util":      util,
        }).execute()
    except Exception:
        pass

# =============================================================================
# INTERFAZ — BARRA LATERAL
# =============================================================================
with st.sidebar:
    st.markdown("### 📊 Sesión actual")
    n = st.session_state.consultas_sesion
    st.progress(
        min(n / MAX_CONSULTAS_SESION, 1.0),
        text=f"{n} / {MAX_CONSULTAS_SESION} consultas usadas",
    )
    if st.session_state.historial_completo:
        st.caption(f"Consultas en historial: {len(st.session_state.historial_completo)}")
    st.divider()
    st.caption(
        "ℹ️ El límite se restablece al cerrar y reabrir el navegador. "
        "El servicio de IA (Groq) tiene un límite global de 1.000 req/día."
    )

# =============================================================================
# INTERFAZ — CUERPO PRINCIPAL
# =============================================================================
st.title("📚 Buscador Inteligente de Normativa Educativa")

bloque_elegido = st.selectbox(
    "Nivel educativo:",
    ["ninguno", "infantil_primaria", "secundaria_bachillerato", "fp"],
    format_func=lambda x: {
        "ninguno":                 "— Selecciona un nivel educativo —",
        "infantil_primaria":       "Infantil y Primaria",
        "secundaria_bachillerato": "Secundaria y Bachillerato",
        "fp":                      "Formación Profesional",
    }[x],
)

# Preguntas de ejemplo clicables
with st.expander("💡 Ver ejemplos de preguntas"):
    ejemplos = [
        "¿Cuántos días de permiso tiene un docente por nacimiento de hijo?",
        "¿Cuáles son los requisitos para solicitar una excedencia voluntaria?",
        "¿Qué ratio de alumnos por aula establece la normativa en Primaria?",
        "¿Cómo se tramita una baja por enfermedad de un docente interino?",
        "¿Qué documentos necesita aportar un alumno para matricularse en 1º ESO?",
    ]
    for ej in ejemplos:
        if st.button(ej, use_container_width=True, key=f"ej_{ej[:30]}"):
            st.session_state["_ejemplo"] = ej
            st.rerun()

# Campo de pregunta
valor_inicial = st.session_state.pop("_ejemplo", "")
with st.form(key="form_busqueda"):
    pregunta_input = st.text_area(
        "Haz tu pregunta sobre la normativa:",
        value=valor_inicial,
        height=100,
        max_chars=MAX_CHARS_PREGUNTA,
        placeholder="Escribe aquí tu consulta... (no te preocupes por los errores ortográficos, los corrijo automáticamente)",
    )
    submit = st.form_submit_button("🔍 Buscar", use_container_width=True)

# =============================================================================
# PROCESAMIENTO
# =============================================================================
if submit and pregunta_input:

    if bloque_elegido == "ninguno":
        st.warning("⚠️ Selecciona un nivel educativo antes de buscar.")

    elif st.session_state.consultas_sesion >= MAX_CONSULTAS_SESION:
        st.error(
            f"Has alcanzado el límite de {MAX_CONSULTAS_SESION} consultas por sesión. "
            "Recarga la página para empezar una nueva sesión."
        )

    else:
        valido, msg_error = validar_input(pregunta_input)
        if not valido:
            st.warning(f"⚠️ {msg_error}")
        else:
            try:
                t0 = time.time()

                # ── PASO 1: Corrección ortográfica + expansión ────────────────
                # Una sola llamada al 8B que corrige errores ("coemdor"→"comedor")
                # y genera 3 reformulaciones con vocabulario jurídico.
                with st.spinner("✏️ Analizando la consulta..."):
                    pregunta_corregida, reformulaciones = expandir_y_corregir(pregunta_input)

                if pregunta_corregida.strip().lower() != pregunta_input.strip().lower():
                    st.info(f"✏️ He corregido tu consulta a: **{pregunta_corregida}**")

                # ── PASO 2: Embedding promediado (centroide semántico) ────────
                # Vectorizamos la pregunta corregida + 2 reformulaciones y
                # calculamos la media aritmética. El vector resultante representa
                # el centro del espacio semántico de todas las variantes, lo que
                # aumenta significativamente el recall de la búsqueda.
                with st.spinner("🔎 Buscando en la normativa..."):
                    todas = [pregunta_corregida] + reformulaciones[:2]
                    embedding_avg = np.mean(
                        [model.encode(q) for q in todas], axis=0
                    ).tolist()

                    # ── PASO 3: Búsqueda híbrida ─────────────────────────────
                    resultados = buscar_normativa_hibrida(
                        embedding_avg, pregunta_corregida, bloque_elegido
                    )

                if not resultados:
                    st.warning(
                        "No encontré normativa relacionada con tu pregunta en este nivel "
                        "educativo. Prueba a reformularla o usa términos más específicos."
                    )
                    guardar_log(bloque_elegido, pregunta_input, pregunta_corregida,
                                0, (time.time()-t0)*1000, False)
                else:
                    # ── PASO 4: Re-ranking ────────────────────────────────────
                    # El 8B puntúa y reordena los fragmentos por relevancia real.
                    with st.spinner("📊 Ordenando resultados por relevancia..."):
                        resultados = reranquear(pregunta_corregida, resultados)
                        resultados = resultados[:MATCH_COUNT]

                    # ── PASO 5: Contexto XML + fuentes ────────────────────────
                    contexto_xml, links_screen, fuentes_pdf = construir_contexto_xml(
                        resultados, enlaces
                    )

                    # ── PASO 6: Mensajes con historial real ───────────────────
                    mensajes = construir_mensajes(pregunta_corregida, contexto_xml)

                    # ── PASO 7: Generación con streaming (70B) ────────────────
                    st.write("---")
                    st.markdown("### 📝 Respuesta:")

                    stream = groq_client.chat.completions.create(
                        model=GROQ_MODEL_PRINCIPAL,
                        messages=mensajes,
                        temperature=0.1,
                        max_tokens=MAX_TOKENS_RESPUESTA,
                        stream=True,
                    )

                    def _gen():
                        for chunk in stream:
                            delta = chunk.choices[0].delta.content
                            if delta:
                                yield delta

                    texto_final = st.write_stream(_gen())

                    # ── PASO 8: Fuentes ───────────────────────────────────────
                    fuentes_u  = list(dict.fromkeys(links_screen))
                    fuentes_up = list(dict.fromkeys(fuentes_pdf))
                    st.markdown("### 📚 Fuentes consultadas:")
                    for f in fuentes_u:
                        st.markdown(f"- 📄 {f}")

                    # ── PASO 9: Guardar sesión ────────────────────────────────
                    st.session_state.ultima_pregunta   = pregunta_input
                    st.session_state.ultima_respuesta  = texto_final
                    st.session_state.ultimas_fuentes   = fuentes_u
                    st.session_state.historial_completo.append({
                        "pregunta":           pregunta_input,
                        "pregunta_corregida": pregunta_corregida,
                        "respuesta":          texto_final,
                        "fuentes":            fuentes_up,
                    })
                    # Limitar historial en memoria a los últimos 20 registros
                    if len(st.session_state.historial_completo) > 20:
                        st.session_state.historial_completo = \
                            st.session_state.historial_completo[-20:]

                    st.session_state.consultas_sesion  += 1
                    st.session_state.feedback_pendiente = True
                    st.session_state.feedback_pregunta  = pregunta_input
                    st.session_state.feedback_respuesta = texto_final

                    # ── PASO 10: Log silencioso ───────────────────────────────
                    guardar_log(bloque_elegido, pregunta_input, pregunta_corregida,
                                len(resultados), (time.time()-t0)*1000, True)

            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate_limit" in err or "rate limit" in err:
                    st.error(
                        "⏳ Límite diario de consultas alcanzado (1.000 req/día en el plan "
                        "gratuito de Groq). Se restablece automáticamente a medianoche UTC."
                    )
                else:
                    st.error(f"Error técnico al procesar la consulta: {e}")

elif st.session_state.ultima_respuesta:
    st.write("---")
    st.markdown(st.session_state.ultima_respuesta)
    st.markdown("### 📚 Fuentes consultadas:")
    for f in st.session_state.ultimas_fuentes:
        st.markdown(f"- 📄 {f}")

# =============================================================================
# FEEDBACK — 👍 / 👎 tras cada respuesta
# =============================================================================
if st.session_state.feedback_pendiente:
    st.markdown("---")
    st.markdown("**¿Te ha resultado útil esta respuesta?**")
    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        if st.button("👍 Sí"):
            guardar_feedback(st.session_state.feedback_pregunta,
                             st.session_state.feedback_respuesta, True)
            st.session_state.feedback_pendiente = False
            st.success("¡Gracias por tu valoración!")
            st.rerun()
    with c2:
        if st.button("👎 No"):
            guardar_feedback(st.session_state.feedback_pregunta,
                             st.session_state.feedback_respuesta, False)
            st.session_state.feedback_pendiente = False
            st.info("Lo tendremos en cuenta para mejorar.")
            st.rerun()

# =============================================================================
# HISTORIAL VISIBLE EN PANTALLA
# =============================================================================
historial = st.session_state.historial_completo
if len(historial) > 1:
    st.write("---")
    with st.expander(
        f"📋 Historial de esta sesión ({len(historial)} consultas)", expanded=False
    ):
        for item in reversed(historial[:-1]):
            st.markdown(f"**Pregunta:** {item['pregunta']}")
            prev = item["respuesta"]
            st.markdown(prev[:400] + "..." if len(prev) > 400 else prev)
            st.divider()

# =============================================================================
# BOTONES DE ACCIÓN
# =============================================================================
if historial:
    st.write("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "📄 Descargar esta consulta",
            data=generar_pdf([historial[-1]], "Consulta de Normativa Educativa"),
            file_name="consulta_normativa.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "📚 Descargar historial",
            data=generar_pdf(historial, "Historial Completo de Consultas"),
            file_name="historial_normativa.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with c3:
        if not st.session_state.confirmar_borrar:
            if st.button("🔄 Reiniciar chat", use_container_width=True):
                st.session_state.confirmar_borrar = True
                st.rerun()
        else:
            st.warning("⚠️ ¿Seguro? Se borrará todo el historial.")
            ca, cb = st.columns(2)
            with ca:
                if st.button("✅ Sí, borrar", use_container_width=True):
                    for k, v in _DEFAULTS.items():
                        st.session_state[k] = ([] if isinstance(v, list) else v)
                    st.rerun()
            with cb:
                if st.button("❌ Cancelar", use_container_width=True):
                    st.session_state.confirmar_borrar = False
                    st.rerun()
