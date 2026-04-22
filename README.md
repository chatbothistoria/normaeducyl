# 📚 Buscador de Normativa Educativa (100% Semántico y Automatizado)

Aplicación web desarrollada con **Streamlit** para consultar **38 documentos oficiales de normativa educativa** de Castilla y León mediante búsqueda verdaderamente semántica. 

Las respuestas se generan **exclusivamente** a partir del contenido de los PDF, usando **FAISS** para la recuperación de vectores (significados) y el modelo **llama-3.3-70b-versatile** de **Groq** para la redacción de la respuesta final.

---

## ✨ Características Principales
1. **Auto-Configuración Inteligente:** No necesitas preprocesar nada a mano. La primera vez que se ejecuta la app, detecta si falta el índice, lee los PDFs usando `pdfplumber`, extrae los fragmentos y crea la base de datos vectorial automáticamente.
2. **Búsqueda por Significado:** Usa el modelo `paraphrase-multilingual-mpnet-base-v2` (SentenceTransformers) para entender lo que preguntas, no solo para buscar palabras exactas.
3. **Respuesta RAG (Retrieval-Augmented Generation):** Groq recibe el contexto exacto del BOE/BOCYL y redacta una respuesta detallada sin inventar datos (alucinaciones minimizadas).
4. **Exportación a PDF:** Puedes descargar la respuesta generada junto con sus fuentes en un documento PDF bien formateado.

---

## 🚀 Puesta en marcha (Streamlit Cloud / Local)

### 1. Estructura de archivos necesaria
Asegúrate de que tu repositorio en GitHub (o tu carpeta local) tenga exactamente esta estructura:

```text
📁 tu-repositorio/
├── app.py                  
├── requirements.txt        
├── README.md               
└── Normativa_Oficial/      ← CARPETA: Pon aquí tus 38 PDFs
    ├── 01_Ley_Organica_2_2006_LOE_consolidada.pdf
    └── ...