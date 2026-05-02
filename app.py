import streamlit as st
import faiss
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from groq import Groq

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
3. Cita SIEMPRE el documento y la página de donde sacas la información al final de tu explicación.
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
        "Primaria": ("faiss_primaria.bin", "meta_primaria.json"),
        "Secundaria": ("faiss_secundaria.bin", "meta_secundaria.json"),
        "FP": ("faiss_fp.bin", "meta_fp.json")
    }
    bin_file, json_file = archivos.get(etapa, (None, None))

    if not bin_file or not os.path.exists(bin_file) or not os.path.exists(json_file):
        return None, None

    # faiss.read_index lee automáticamente los índices de 8-bits sin problemas
    index = faiss.read_index(bin_file)
    with open(json_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

embed_model = load_embedding_model()
client = load_groq_client()

# ==============================================================
# 3. INTERFAZ DE USUARIO (UI)
# ==============================================================
st.title("📚 Asistente de Normativa Educativa - CyL")

with st.sidebar:
    st.header("⚙️ Configuración")
    etapa_seleccionada = st.selectbox(
        "Selecciona la Etapa Educativa:",
        ["Primaria", "Secundaria", "FP"]
    )
    st.info("💡 Consejo: Selecciona la etapa correspondiente a tu pregunta para que la IA busque en las leyes correctas.")

index, metadata = load_faiss_and_meta(etapa_seleccionada)

if index is None or metadata is None:
    st.error(f"⚠️ Faltan los archivos de la etapa {etapa_seleccionada}. Revisa que los archivos .bin y .json estén en GitHub.")
    st.stop()

# ==============================================================
# 4. GESTIÓN DE LA MEMORIA
# ==============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": f"¡Hola! Soy tu asistente de normativa. He cargado las leyes de **{etapa_seleccionada}**. ¿En qué puedo ayudarte?"})

if "current_etapa" not in st.session_state:
    st.session_state.current_etapa = etapa_seleccionada

# Si el usuario cambia de etapa en el menú lateral, avisamos en el chat
if st.session_state.current_etapa != etapa_seleccionada:
    st.session_state.messages = [{"role": "assistant", "content": f"He cambiado mi base de datos a **{etapa_seleccionada}**. ¿Qué necesitas saber?"}]
    st.session_state.current_etapa = etapa_seleccionada

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================================================
# 5. LÓGICA DEL RAG (BÚSQUEDA)
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

        # Convertimos los guiones bajos en espacios para que la IA lo lea de forma más natural
        nombre_real = doc_name.replace(".pdf", "").replace("_", " ")

        fragmento = f"--- [Documento: {nombre_real} | Página: {page}] ---\n{texto}\n"

        if fragmento not in contexto_textos:
            contexto_textos.append(fragmento)
            documentos_citados.add(f"- {nombre_real} (Pág. {page})")

        if len(contexto_textos) >= MAX_CHUNKS_TO_LLM:
            break

    return "\n".join(contexto_textos), list(documentos_citados)

# ==============================================================
# 6. INTERACCIÓN DEL USUARIO
# ==============================================================
if prompt := st.chat_input("Escribe tu pregunta sobre normativa..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en la normativa..."):
            contexto_str, citas = buscar_contexto(prompt)
            
        mensajes_api = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=contexto_str)}
        ]

        # Pasamos un poco de memoria de la conversación reciente
        for m in st.session_state.messages[-3:-1]:
            mensajes_api.append({"role": m["role"], "content": m["content"]})

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

            # Añadir las citas al final de la respuesta si encontró información
            if citas and "no encuentro" not in respuesta_completa.lower() and "lo siento" not in respuesta_completa.lower():
                citas_mostrar = citas[:4] # Mostramos un máximo de 4 fuentes para no saturar visualmente
                pie_fuentes = "\n\n---\n**📚 Fuentes consultadas:**\n" + "\n".join(citas_mostrar)
                respuesta_completa += pie_fuentes

            respuesta_placeholder.markdown(respuesta_completa)

        except Exception as e:
            respuesta_completa = f"⚠️ Ocurrió un error al contactar con la IA: {e}"
            respuesta_placeholder.markdown(respuesta_completa)

        st.session_state.messages.append({"role": "assistant", "content": respuesta_completa})