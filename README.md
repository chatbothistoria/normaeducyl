# 📚 Buscador de Normativa Educativa (100% Semántico)

Aplicación Streamlit para consultar **38 documentos oficiales de normativa educativa** de Castilla y León mediante búsqueda verdaderamente semántica. Las respuestas se generan **exclusivamente** a partir del contenido de los PDF, usando **FAISS** para la recuperación de vectores y **llama-3.3-70b-versatile** de Groq para la respuesta.

---

## ⚙️ Puesta en marcha

### 1. Coloca los PDFs
Copia los 38 archivos PDF en la carpeta `Normativa_Oficial/`.

### 2. Genera el índice semántico localmente (Solo una vez)
Asegúrate de haber instalado los requerimientos (`pip install -r requirements.txt`). Luego ejecuta:
```bash
python preprocess.py