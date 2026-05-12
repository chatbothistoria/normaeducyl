import streamlit as st
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import csv
import os
from fpdf import FPDF
import textwrap

# =============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(page_title="Buscador de Normativa Educativa", page_icon="📚")

# =============================================================================
# 2. SESSION STATE
# =============================================================================
for _key, _default in [
    ("ultima_pregunta",   None),
    ("ultima_respuesta",  None),
    ("ultimas_fuentes",   []),
    ("historial_completo", []),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# =============================================================================
# 3. PDF — FIX DE ENCODING
#
# Los caracteres españoles (á é í ó ú ñ ü Á É etc.) SÍ están en Latin-1
# y FPDF los renderiza perfectamente. El problema eran los caracteres Unicode
# que los LLMs insertan: comillas tipográficas, rayas, puntos suspensivos, etc.
# Los sustituimos antes de codificar, así el PDF sale limpio.
# =============================================================================
_UNICODE_FIX = {
    "\u2018": "'",   # '  comilla simple izquierda
    "\u2019": "'",   # '  comilla simple derecha (la más frecuente en LLMs)
    "\u201C": '"',   # "  comilla doble izquierda
    "\u201D": '"',   # "  comilla doble derecha
    "\u2013": "-",   # –  guion medio (en-dash)
    "\u2014": "-",   # —  guion largo (em-dash)
    "\u2022": "-",   # •  viñeta
    "\u00B7": "-",   # ·  punto medio
    "\u2026": "...", # …  puntos suspensivos
    "\u00A0": " ",   # espacio no separable
    "\u00AD": "-",   # guion suave
}

def _limpiar(texto: str) -> str:
    """Preserva á/é/í/ó/ú/ñ y sustituye los chars Unicode que FPDF no soporta."""
    for orig, repl in _UNICODE_FIX.items():
        texto = texto.replace(orig, repl)
    return texto.encode("latin-1", "replace").decode("latin-1")

def generar_pdf(lista_interacciones, titulo_documento="Normativa Educativa"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(0, 10, _limpiar(titulo_documento), ln=True, align="C")
    pdf.ln(10)

    for item in lista_interacciones:
        # Pregunta
        pdf.set_font("Helvetica", style="B", size=12)
        for linea in textwrap.wrap(f"PREGUNTA: {_limpiar(item['pregunta'])}", width=80):
            pdf.cell(0, 6, txt=linea, ln=True)
        pdf.ln(2)

        # Respuesta
        pdf.set_font("Helvetica", size=11)
        for linea in textwrap.wrap(_limpiar(item["respuesta"]), width=90):
            pdf.cell(0, 6, txt=linea, ln=True)
        pdf.ln(5)

        # Fuentes
        pdf.set_font("Helvetica", style="I", size=10)
        pdf.cell(0, 6, txt="FUENTES CONSULTADAS:", ln=True)
        for fuente in item["fuentes"]:
            for linea in textwrap.wrap(f"- {_limpiar(fuente)}", width=90):
                pdf.cell(0, 5, txt=linea, ln=True)
        pdf.ln(10)

    return bytes(pdf.output())

# =============================================================================
# 4. CLAVES DE ACCESO (desde st.secrets)
# =============================================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# =============================================================================
# 5. INICIALIZAR SERVICIOS (cacheados para no recargar en cada interacción)
# =============================================================================
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource
def load_model():
    # ⚠️  IMPORTANTE: NO cambiar este modelo.
    # Debe coincidir exactamente con el que se usó para vectorizar los
    # documentos en Supabase. Cambiarlo aquí sin re-vectorizar la BD
    # produciría resultados de búsqueda incorrectos.
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

@st.cache_data
def cargar_diccionario_enlaces():
    enlaces = {}
    if os.path.exists("enlaces.csv"):
        with open("enlaces.csv", mode="r", encoding="utf-8") as f:
            for i, fila in enumerate(csv.reader(f)):
                if i == 0:
                    continue  # cabecera
                if len(fila) >= 2:
                    enlaces[fila[0].strip()] = fila[1].strip()
    return enlaces

supabase            = init_supabase()
model               = load_model()
groq_client         = Groq(api_key=GROQ_API_KEY)
diccionario_enlaces = cargar_diccionario_enlaces()

# =============================================================================
# 6. INTERFAZ
# =============================================================================
st.title("📚 Buscador Inteligente de Normativa Educativa")

bloque_elegido = st.selectbox(
    "Nivel educativo:",
    ["ninguno", "infantil_primaria", "secundaria_bachillerato", "fp"],
    format_func=lambda x: {
        "ninguno":                   "Por favor, elige un nivel educativo",
        "infantil_primaria":         "Infantil y Primaria",
        "secundaria_bachillerato":   "Secundaria y Bachillerato",
        "fp":                        "Formación Profesional",
    }[x],
)

with st.form(key="search_form"):
    pregunta      = st.text_input("Haz tu pregunta sobre la normativa:")
    submit_button = st.form_submit_button(label="🔍 Buscar")

# =============================================================================
# 7. PROCESAMIENTO DE LA BÚSQUEDA
# =============================================================================
if submit_button and pregunta:

    if bloque_elegido == "ninguno":
        st.warning("⚠️ Por favor, selecciona un nivel educativo antes de buscar.")

    else:
        try:
            # --- 7a. Embedding + búsqueda vectorial en Supabase ---
            with st.spinner("🔎 Buscando en la normativa..."):
                embedding_pregunta = model.encode(pregunta).tolist()

                respuesta_bd = supabase.rpc(
                    "buscar_normativa",
                    {
                        "query_embedding": embedding_pregunta,
                        "filtro_bloque":   bloque_elegido,
                        "match_threshold": 0.5,
                        # ↓ Reducido de 12 a 6: menos tokens, contexto más enfocado
                        #   y sin riesgo de superar la ventana del modelo.
                        "match_count":     6,
                    },
                ).execute()
                resultados = respuesta_bd.data

            # --- 7b. Sin resultados ---
            if not resultados:
                st.warning(
                    "No he encontrado normativa relacionada con tu pregunta "
                    "en este nivel educativo. Prueba a reformularla o usar "
                    "términos más específicos."
                )

            else:
                # --- 7c. Construir contexto y fuentes ---
                contexto_para_ia  = ""
                enlaces_fuentes   = []
                textos_fuentes_pdf = []

                for res in resultados:
                    nombre_archivo = res["nombre_archivo"]
                    pagina         = res["pagina_num"]
                    nombre_limpio  = nombre_archivo.replace(".pdf", "").replace("_", " ")

                    contexto_para_ia += (
                        f"DOCUMENTO: {nombre_limpio} | PÁGINA: {pagina}\n"
                        f"CONTENIDO: {res['contenido']}\n\n"
                    )

                    url_base = diccionario_enlaces.get(nombre_archivo)
                    if url_base:
                        enlaces_fuentes.append(
                            f"[{nombre_limpio} (Pág. {pagina})]({url_base}#page={pagina})"
                        )
                        textos_fuentes_pdf.append(
                            f"{nombre_limpio} (Pág. {pagina}) - Enlace: {url_base}"
                        )
                    else:
                        enlaces_fuentes.append(
                            f"**{nombre_limpio}** (Pág. {pagina}) *(enlace no disponible)*"
                        )
                        textos_fuentes_pdf.append(f"{nombre_limpio} (Pág. {pagina})")

                # --- 7d. Construir los mensajes para la API ---
                #
                # MEJORA: el historial ya NO se pega como texto plano en el prompt.
                # Se pasa como turnos reales de conversación en el array `messages`,
                # que es el mecanismo nativo de los LLMs para el contexto conversacional.
                #
                prompt_sistema = (
                    "Eres un experto asesor jurista especializado en normativa educativa española. "
                    "Analiza el contexto normativo que se te proporciona y responde de forma clara "
                    "y bien estructurada, usando párrafos separados. "
                    "Cuando cites algo concreto, indica el nombre del documento y la página. "
                    "Responde ÚNICAMENTE con la información que aparece en el contexto. "
                    "Si la información no es suficiente para responder a la pregunta, dilo claramente. "
                    "Nunca inventes ni supongas leyes, artículos o normativas."
                )

                mensajes = [{"role": "system", "content": prompt_sistema}]

                # Añadir turno anterior como contexto conversacional real
                if st.session_state.ultima_pregunta and st.session_state.ultima_respuesta:
                    mensajes.append({
                        "role": "user",
                        "content": st.session_state.ultima_pregunta,
                    })
                    # Truncamos a 1500 chars para no desperdiciar tokens
                    respuesta_prev = st.session_state.ultima_respuesta
                    if len(respuesta_prev) > 1500:
                        respuesta_prev = respuesta_prev[:1500] + "..."
                    mensajes.append({
                        "role": "assistant",
                        "content": respuesta_prev,
                    })

                # Turno actual con el contexto normativo
                mensajes.append({
                    "role": "user",
                    "content": (
                        f"CONTEXTO NORMATIVO:\n{contexto_para_ia}\n\n"
                        f"PREGUNTA: {pregunta}"
                    ),
                })

                # --- 7e. Llamada a Groq con streaming ---
                #
                # MODELO: llama-3.3-70b-versatile
                #   - Gratis en Groq (1.000 req/día en plan gratuito)
                #   - 70 mil millones de parámetros vs 8B anterior
                #   - Ventana de 128K tokens (vs 8K del modelo anterior)
                #   - Calidad muy superior para textos jurídicos en español
                #
                # max_tokens=900: protege la cuota diaria gratuita.
                # Sin este límite el modelo puede generar 3.000-4.000 tokens
                # por respuesta, agotando el límite mucho antes.
                #
                st.write("---")
                st.markdown("### 📝 Respuesta:")

                stream = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=mensajes,
                    temperature=0.1,   # bajo para respuestas factuales y consistentes
                    max_tokens=900,    # ← protege la cuota gratuita (1.000 req/día)
                    stream=True,
                )

                def _stream_gen():
                    """Generador limpio para st.write_stream."""
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            yield delta

                texto_final = st.write_stream(_stream_gen())

                # --- 7f. Mostrar fuentes ---
                fuentes_unicas     = list(dict.fromkeys(enlaces_fuentes))
                fuentes_unicas_pdf = list(dict.fromkeys(textos_fuentes_pdf))

                st.markdown("### 📚 Fuentes consultadas:")
                for fuente in fuentes_unicas:
                    st.markdown(f"- 📄 {fuente}")

                # --- 7g. Guardar en sesión ---
                st.session_state.ultima_pregunta  = pregunta
                st.session_state.ultima_respuesta = texto_final
                st.session_state.ultimas_fuentes  = fuentes_unicas
                st.session_state.historial_completo.append({
                    "pregunta":  pregunta,
                    "respuesta": texto_final,
                    "fuentes":   fuentes_unicas_pdf,
                })

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate_limit" in err or "rate limit" in err:
                st.error(
                    "⏳ Se ha alcanzado el límite de consultas del servicio de IA por hoy "
                    "(1.000 consultas diarias en el plan gratuito). "
                    "El límite se restablece automáticamente a medianoche UTC. "
                    "Inténtalo de nuevo más tarde."
                )
            else:
                st.error(f"Error técnico al procesar la consulta: {e}")

# =============================================================================
# 8. MOSTRAR LA ÚLTIMA RESPUESTA (solo cuando NO se acaba de buscar)
#    Esto evita que la respuesta aparezca duplicada al hacer streaming.
# =============================================================================
elif st.session_state.ultima_respuesta:
    st.write("---")
    st.markdown(st.session_state.ultima_respuesta)

    st.markdown("### 📚 Fuentes consultadas:")
    for fuente in st.session_state.ultimas_fuentes:
        st.markdown(f"- 📄 {fuente}")

# =============================================================================
# 9. BOTONES DE ACCIÓN (siempre visibles cuando hay historial)
# =============================================================================
if st.session_state.historial_completo:
    st.write("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        pdf_actual = generar_pdf(
            [st.session_state.historial_completo[-1]],
            "Consulta de Normativa Educativa",
        )
        st.download_button(
            label="📄 Descargar esta consulta",
            data=pdf_actual,
            file_name="consulta_normativa.pdf",
            mime="application/pdf",
        )

    with col2:
        pdf_historial = generar_pdf(
            st.session_state.historial_completo,
            "Historial Completo",
        )
        st.download_button(
            label="📚 Descargar historial",
            data=pdf_historial,
            file_name="historial_normativa.pdf",
            mime="application/pdf",
        )

    with col3:
        if st.button("🔄 Reiniciar chat"):
            st.session_state.ultima_pregunta   = None
            st.session_state.ultima_respuesta  = None
            st.session_state.ultimas_fuentes   = []
            st.session_state.historial_completo = []
            st.rerun()
