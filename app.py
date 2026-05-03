import streamlit as st
import faiss
import json
import numpy as np
import os
import csv
import io
import re
from xml.sax.saxutils import escape
from sentence_transformers import SentenceTransformer
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ==============================================================
# 1. CONFIGURACIÓN DE PÁGINA Y PARÁMETROS GLOBALES
# ==============================================================
st.set_page_config(page_title="Asistente Normativa Educativa CyL", page_icon="📚", layout="centered")

FETCH_CHUNKS = 30           
MAX_CHUNKS_TO_LLM = 8       
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2" 

SYSTEM_PROMPT = """Eres un experto legal en normativa educativa de Castilla y León.
Tu objetivo es responder a las dudas de los usuarios basándote ÚNICAMENTE en el contexto proporcionado.

REGLAS:
1. Lee detenidamente el contexto proporcionado. A veces la información está dividida en varios fragmentos.
2. Si el contexto contiene la respuesta (aunque sea de forma parcial o con sinónimos), redacta una respuesta clara, profesional y empática.
3. Cita los documentos en el texto de tu respuesta, pero NUNCA generes un apartado final de "Fuentes consultadas", bibliografía o referencias (el sistema lo añadirá automáticamente).
4. Si la información no está en el contexto, di educadamente: "Lo siento, pero no encuentro esa información exacta en la normativa que tengo cargada."
5. NUNCA te inventes leyes, fechas o datos.

CONTEXTO DE BÚSQUEDA:
{context}
"""

# ==============================================================
# 2. CARGA DE RECURSOS EN CACHÉ
# ==============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_faiss_and_meta(etapa):
    archivos = {
        "Infantil y Primaria": ("faiss_primaria.bin", "meta_primaria.json"),
        "ESO y Bachillerato": ("faiss_secundaria.bin", "meta_secundaria.json"),
        "Formación Profesional": ("faiss_fp.bin", "meta_fp.json")
    }
    bin_file, json_file = archivos.get(etapa, (None, None))

    if not bin_file or not os.path.exists(bin_file) or not os.path.exists(json_file):
        return None, None

    index = faiss.read_index(bin_file)
    with open(json_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

@st.cache_data
def load_urls():
    diccionario = {}
    if os.path.exists("enlaces.csv"):
        try:
            with open("enlaces.csv", "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if "nombre_archivo" in reader.fieldnames and "url_oficial_verificada" in reader.fieldnames:
                    for row in reader:
                        pdf_name = row["nombre_archivo"].strip()
                        url = row.get("url_oficial_verificada", "").strip()
                        if url and url.lower() != "nan":
                            diccionario[pdf_name] = url
        except Exception as e:
            st.warning(f"⚠️ Error interno leyendo enlaces.csv: {e}")
    return diccionario

embed_model = load_embedding_model()
client = load_groq_client()
diccionario_enlaces = load_urls()

# ==============================================================
# 3. MOTOR DE GENERACIÓN DE PDF
# ==============================================================
def generar_pdf(mensajes, titulo="Documento Normativo"):
    """Genera un archivo PDF a partir de una lista de mensajes"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    estilo_normal = styles["Normal"]
    estilo_titulo = styles["Title"]
    
    flowables = [Paragraph(titulo, estilo_titulo), Spacer(1, 20)]
    
    for msg in mensajes:
        if msg["role"] == "system":
            continue
            
        rol = "USUARIO" if msg["role"] == "user" else "ASISTENTE NORMATIVO"
        
        texto = escape(msg["content"])
        texto = texto.encode('windows-1252', 'ignore').decode('windows-1252')
        texto = texto.replace('\n', '<br/>')
        texto = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', texto) 
        texto = re.sub(r'\*(.*?)\*', r'<i>\1</i>', texto)     
        
        flowables.append(Paragraph(f"<b>[{rol}]</b>", estilo_normal))
        flowables.append(Spacer(1, 5))
        flowables.append(Paragraph(texto, estilo_normal))
        flowables.append(Spacer(1, 15))
        
    doc.build(flowables)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ==============================================================
# 4. INTERFAZ DE USUARIO (UI)
# ==============================================================
st.title("📚 Asistente de Normativa Educativa - CyL")

etapa_seleccionada = st.selectbox(
    "Selecciona la Etapa Educativa:",
    ["Infantil y Primaria", "ESO y Bachillerato", "Formación Profesional"]
)

# 🌟 NUEVO: Disclaimer añadido
st.warning("⚠️ **Nota importante:** Este asistente utiliza Inteligencia Artificial para buscar y resumir la normativa educativa de Castilla y León. Aunque está diseñado para ser riguroso, la IA puede cometer errores, omitir matices o no reflejar la interpretación jurídica exacta. Utiliza esta herramienta como una guía de apoyo y contrasta siempre la información final con los documentos oficiales.")

st.divider()

index, metadata = load_faiss_and_meta(etapa_seleccionada)

if index is None or metadata is None:
    st.error(f"⚠️ Faltan los archivos de la etapa seleccionada. Revisa que los archivos .bin y .json estén en GitHub.")
    st.stop()

# ==============================================================
# 5. GESTIÓN DE LA MEMORIA Y CHAT
# ==============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": f"¡Hola! Soy tu asistente de normativa. He cargado las leyes de **{etapa_seleccionada}**. ¿En qué puedo ayudarte?"})

if "current_etapa" not in st.session_state:
    st.session_state.current_etapa = etapa_seleccionada

if st.session_state.current_etapa != etapa_seleccionada:
    st.session_state.messages = [{"role": "assistant", "content": f"He cambiado mi base de datos a **{etapa_seleccionada}**. ¿Qué necesitas saber?"}]
    st.session_state.current_etapa = etapa_seleccionada

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and i > 0: 
            msg_usuario = st.session_state.messages[i-1] if i>0 and st.session_state.messages[i-1]["role"] == "user" else {"role": "user", "content": "Consulta general"}
            
            mensajes_pdf_individual = [msg_usuario, msg]
            pdf_individual = generar_pdf(mensajes_pdf_individual, "Consulta Normativa")
            pdf_historial = generar_pdf(st.session_state.messages, "Historial de Consultas Normativas")
            
            col_espacio, col_btn_resp, col_btn_conv = st.columns([4, 3, 3])
            
            with col_btn_resp:
                st.download_button(
                    label="📥 Guardar respuesta",
                    data=pdf_individual,
                    file_name=f"consulta_normativa_{i}.pdf",
                    mime="application/pdf",
                    key=f"dl_resp_{i}",
                    use_container_width=True
                )
                
            with col_btn_conv:
                st.download_button(
                    label="📄 Guardar conversación",
                    data=pdf_historial,
                    file_name="historial_normativa.pdf",
                    mime="application/pdf",
                    key=f"dl_conv_{i}",
                    use_container_width=True
                )

# ==============================================================
# 6. LÓGICA DEL RAG (BÚSQUEDA Y CITAS)
# ==============================================================
def buscar_contexto(pregunta):
    vector = embed_model.encode([pregunta], convert_to_numpy=True).astype('float32')
    distancias, indices = index.search(vector, FETCH_CHUNKS)

    contexto_textos = []
    documentos_citados = set()

    for idx in indices[0]:
        if idx == -1 or idx >= len(metadata):
            continue
            
        meta = metadata[idx]
        texto = meta["chunk_text"]
        doc_name = meta["doc_name"]
        page = meta["page_num"]

        nombre_real = doc_name.replace(".pdf", "").replace("_", " ")

        url_oficial = diccionario_enlaces.get(doc_name)
        if url_oficial:
            cita_formateada = f"- [{nombre_real}]({url_oficial}) (Pág. {page})"
        else:
            cita_formateada = f"- {nombre_real} (Pág. {page})"

        fragmento = f"--- [Documento: {nombre_real} | Página: {page}] ---\n{texto}\n"

        if fragmento not in contexto_textos:
            contexto_textos.append(fragmento)
            documentos_citados.add(cita_formateada)

        if len(contexto_textos) >= MAX_CHUNKS_TO_LLM:
            break

    return "\n".join(contexto_textos), list(documentos_citados)

# ==============================================================
# 7. INTERACCIÓN DEL USUARIO
# ==============================================================
if prompt := st.chat_input("Escribe tu pregunta sobre normativa..."):
    
    # Añadimos la pregunta del usuario a la memoria visual
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la normativa..."):
            contexto_str, citas = buscar_contexto(prompt)
            
        mensajes_api = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=contexto_str)}
        ]

        # 🌟 ARREGLO: Limpiamos las fuentes del historial enviado a la IA (los últimos 4 mensajes)
        # Esto asegura que la IA nunca lea las fuentes automáticas y no intente imitarlas.
        historial_previo = st.session_state.messages[:-1][-4:] 
        for m in historial_previo:
            contenido = m["content"]
            if m["role"] == "assistant" and "**📚 Fuentes consultadas:**" in contenido:
                contenido = contenido.split("\n\n---")[0].strip()
                
            mensajes_api.append({"role": m["role"], "content": contenido})

        # Añadimos la nueva pregunta del usuario
        mensajes_api.append({"role": "user", "content": prompt})

        respuesta_placeholder = st.empty()
        respuesta_completa = ""

        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=mensajes_api,
                temperature=0.1, 
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    respuesta_completa += chunk.choices[0].delta.content
                    respuesta_placeholder.markdown(respuesta_completa + "▌")

            # 🌟 ARREGLO: Solo se ocultan las citas si dice la frase EXACTA de error.
            if citas and "no encuentro esa información exacta" not in respuesta_completa.lower():
                citas_mostrar = citas[:4] 
                pie_fuentes = "\n\n---\n**📚 Fuentes consultadas:**\n" + "\n".join(citas_mostrar)
                respuesta_completa += pie_fuentes

            respuesta_placeholder.markdown(respuesta_completa)

        except Exception as e:
            respuesta_completa = f"⚠️ Ocurrió un error al contactar con la IA: {e}"
            respuesta_placeholder.markdown(respuesta_completa)

        # Guardamos la respuesta final en la memoria visual
        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})
        
        st.rerun()