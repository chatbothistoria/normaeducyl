import streamlit as st
from sentence_transformers import SentenceTransformer
import requests as _requests
import csv, os, json, textwrap, time, requests, re, unicodedata, difflib
import numpy as np
import hmac
from urllib.parse import urlparse
from fpdf import FPDF

# =============================================================================
# CONFIGURACIÓN CENTRAL
# =============================================================================
# Proveedor de IA oculto al usuario: URL, modelo y clave se configuran en Secrets.
# No fijamos aquí el proveedor para no exponerlo en la interfaz ni en el repositorio.
IA_MODEL   = ""  # se lee más abajo desde Secrets
IA_API_URL = ""  # se lee más abajo desde Secrets
# Modo coste cero: límites duros para evitar consumo excesivo de free tiers.
MAX_TOKENS_RESPUESTA  = 900
MAX_TOKENS_RAPIDO     = 300
IA_REINTENTOS_TEMPORALES = 2      # reintentos ante límite/servicio temporal de IA
IA_REINTENTO_SEGUNDOS    = 20     # espera base antes del reintento automático
IA_REINTENTO_BACKOFF     = 2      # backoff progresivo: 20s, 40s
MAX_CHARS_PREGUNTA    = 500
MAX_CHARS_CONTEXTO    = 18000
MAX_PREGUNTAS_SESION  = 10
MATCH_THRESHOLD_ALTO  = 0.40
MATCH_THRESHOLD_BAJO  = 0.25
MATCH_COUNT           = 8      # fragmentos finales enviados al LLM
MATCH_COUNT_RETRIEVAL = 40     # candidatos recuperados antes de reordenar
HISTORIAL_TURNOS      = 0      # seguridad jurídica: no arrastrar respuestas previas al LLM
MAX_HISTORIAL_LOCAL   = 10
COLLECTION_NAME       = "normativa"
FAQ_FILE              = "faq_normativa.json"
FAQ_MATCH_MIN_RATIO   = 0.88
FAQ_MATCH_MIN_COVER   = 0.78

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(page_title="NormaEdu 2", page_icon="📚", layout="centered")

def _ocultar_navegacion_multipagina():
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        section[data-testid="stSidebar"] nav {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

_ocultar_navegacion_multipagina()


# =============================================================================
# SESSION STATE
# =============================================================================
_DEFAULTS = {
    "historial_completo": [], "ultima_pregunta": None,
    "ultima_respuesta": None, "ultimas_fuentes": [],
    "confirmar_borrar": False,
    "feedback_pendiente": False, "feedback_pregunta": None,
    "feedback_respuesta": None, "pregunta_actual": "",
    "consultas_sesion": 0,
    "ultimo_diagnostico": None,
    "ultima_trazabilidad": None,
    "admin_diagnostico_ok": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =============================================================================
# PDF
# =============================================================================
_UNICODE_FIX = {
    "\u2018":"'", "\u2019":"'", "\u201C":'"', "\u201D":'"',
    "\u2013":"-", "\u2014":"-", "\u2022":"-", "\u00B7":"-",
    "\u2026":"...", "\u00A0":" ", "\u00AD":"-",
}

def _limpiar(texto):
    for orig, repl in _UNICODE_FIX.items():
        texto = texto.replace(orig, repl)
    return texto.encode("latin-1", "replace").decode("latin-1")

def generar_pdf(lista_interacciones, titulo="Normativa Educativa"):
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

        traza = item.get("trazabilidad")
        if traza:
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 10)
            for linea in formatear_trazabilidad_bloque(traza).splitlines():
                for sublinea in textwrap.wrap(_limpiar(linea), 90):
                    pdf.cell(0, 5, sublinea, ln=True)

        pdf.ln(8)
    return bytes(pdf.output())

# =============================================================================
# CLAVES Y SERVICIOS
# =============================================================================
def _leer_secreto(nombre: str) -> str:
    """Lee secretos desde Streamlit Cloud o variables de entorno, sin romper la app.

    En Streamlit Cloud se configuran en Manage app > Settings > Secrets.
    En local pueden definirse como variables de entorno.
    """
    try:
        valor = st.secrets.get(nombre, "")
    except Exception:
        valor = ""
    if not valor:
        valor = os.environ.get(nombre, "")
    return str(valor).strip()


IA_API_KEY = _leer_secreto("IA_API_KEY")
IA_API_URL = _leer_secreto("IA_API_URL").rstrip("/")
IA_MODEL = _leer_secreto("IA_MODEL")
QDRANT_URL = _leer_secreto("QDRANT_URL").rstrip("/")
QDRANT_API_KEY = _leer_secreto("QDRANT_API_KEY")
ADMIN_DIAGNOSTIC_KEY = _leer_secreto("ADMIN_DIAGNOSTIC_KEY")

_secretos_faltantes = [
    nombre for nombre, valor in {
        "IA_API_KEY": IA_API_KEY,
        "IA_API_URL": IA_API_URL,
        "IA_MODEL": IA_MODEL,
        "QDRANT_URL": QDRANT_URL,
        "QDRANT_API_KEY": QDRANT_API_KEY,
    }.items()
    if not valor
]

def _url_http_valida(valor: str) -> bool:
    """Valida que una URL de configuración tenga esquema HTTP/HTTPS.

    No comprueba credenciales ni disponibilidad del servicio; solo evita errores
    tipo MissingSchema cuando se pega por error una clave en un campo URL.
    """
    valor = str(valor or "").strip()
    return bool(re.match(r"^https?://[^\s]+$", valor))

_secretos_mal_formados = []
if IA_API_URL and not _url_http_valida(IA_API_URL):
    _secretos_mal_formados.append("IA_API_URL debe ser un endpoint completo que empiece por https://")
if QDRANT_URL and not _url_http_valida(QDRANT_URL):
    _secretos_mal_formados.append("QDRANT_URL debe ser una URL completa que empiece por https://")

if _secretos_faltantes or _secretos_mal_formados:
    st.title("📚 NormaEdu 2")
    st.error("Faltan claves obligatorias en los Secrets de Streamlit.")
    st.write("Añade estas claves en Streamlit Cloud: **Manage app → Settings → Secrets**.")
    st.code(
        '''IA_API_KEY = "pega_aqui_tu_clave_de_ia"
IA_API_URL = "https://endpoint-del-proveedor/v1/chat/completions"
IA_MODEL = "nombre-del-modelo"
QDRANT_URL = "https://tu-cluster.cloud.qdrant.io"
QDRANT_API_KEY = "pega_aqui_tu_clave_de_qdrant"''',
        language="toml",
    )
    st.warning(
        "No pegues estas claves en GitHub. Deben ir solo en los Secrets de Streamlit "
        "o en variables de entorno locales."
    )
    if _secretos_faltantes:
        st.caption("Faltan: " + ", ".join(_secretos_faltantes))
    if _secretos_mal_formados:
        st.caption("Configuración mal formada: " + " · ".join(_secretos_mal_formados))
    st.stop()


@st.cache_resource
def load_model():
    # ⚠️ NO cambiar sin re-vectorizar los documentos en Qdrant.
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_data
def cargar_enlaces():
    """Carga el diccionario nombre_archivo → URL desde enlaces.csv.
    Usa utf-8-sig para manejar el BOM que tiene el fichero.
    Busca el fichero en varias ubicaciones posibles.
    """
    enlaces = {}
    rutas_posibles = ["enlaces.csv", "normativa_educativa/enlaces.csv", "/app/enlaces.csv"]
    ruta_encontrada = None
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            ruta_encontrada = ruta
            break
    if ruta_encontrada is None:
        return enlaces
    try:
        with open(ruta_encontrada, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for fila in reader:
                nombre = (fila.get("nombre_archivo") or "").strip()
                url    = (fila.get("url_oficial_verificada") or "").strip()
                if nombre and url:
                    enlaces[nombre] = url
    except Exception:
        # Fallback: lectura posicional
        try:
            with open(ruta_encontrada, encoding="utf-8-sig") as f:
                for i, fila in enumerate(csv.reader(f)):
                    if i == 0:
                        continue
                    if len(fila) >= 2 and fila[0].strip() and fila[1].strip():
                        enlaces[fila[0].strip()] = fila[1].strip()
        except Exception:
            pass

    return enlaces

@st.cache_data
def cargar_faq_normativa():
    """Carga la base local de FAQ verificadas.

    Es coste cero: no llama a Qdrant, IA ni servicios externos.
    """
    rutas_posibles = [FAQ_FILE, os.path.join(os.path.dirname(__file__), FAQ_FILE)]
    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            try:
                with open(ruta, encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("faqs", [])
            except Exception:
                return []
    return []


def _normalizar_faq(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _tokens_faq(texto: str) -> set:
    return {t for t in _normalizar_faq(texto).split() if len(t) > 2}


_FAQ_SINONIMOS = {
    "fp": ["fp", "formacion profesional"],
    "evaluacion": ["evaluacion", "evalua", "evaluar", "calificacion", "calificaciones"],
    "bachillerato": ["bachillerato", "bachiller"],
    "titulo": ["titulo", "titular", "obtener"],
    "castilla": ["castilla", "cyl", "castilla leon", "castilla y leon"],
    "leon": ["leon", "cyl", "castilla leon", "castilla y leon"],
    "localidad": ["localidad", "provincia", "otra provincia", "otra ciudad", "distinta localidad"],
    "permiso": ["permiso", "dias", "dia", "baja"],
    "areas": ["areas", "materias", "asignaturas", "ambitos"],
    "cursos": ["cursos", "anos", "duracion"],
    "reclamacion": ["reclamacion", "reclamar", "reclamaciones", "reclamo"],
    "calificaciones": ["calificaciones", "calificacion", "califica", "califican", "notas", "nota", "in", "su", "bi", "nt", "sb"],
    "permanencia": ["permanencia", "permanecer", "repetir", "repeticion"],
    "duracion": ["duracion", "dura", "duran", "cuanto", "cuantos"],
    "acceso": ["acceso", "acceder", "entrar", "requisitos"],
    "tutor": ["tutor", "tutora", "supervisa", "supervision", "seguimiento"],
    "repetir": ["repetir", "repite", "repiten", "repeticion", "permanecer", "permanencia"],
    "empresa": ["empresa", "empresas", "organismo equiparado", "centro empresa"],
    "regimen": ["regimen", "régimen", "general", "intensivo"],
    "norma": ["norma", "regula", "real decreto", "decreto", "ordenacion", "ordenación"],
    "religion": ["religion", "religión"],
    "fuente": ["fuente", "fuentes", "boe", "bocyl", "enlace", "oficial"],
}


def _term_faq_presente(termino: str, pregunta_n: str) -> bool:
    termino_n = _normalizar_faq(termino)
    opciones = _FAQ_SINONIMOS.get(termino_n, [termino_n])
    tokens_pregunta = set(pregunta_n.split())

    for op in opciones:
        op = _normalizar_faq(op)
        if not op:
            continue
        # Evita falsos positivos con términos de una sola letra, por ejemplo
        # required_terms=["d"] activando cualquier pregunta que contenga "de".
        if re.fullmatch(r"[a-z]", op):
            if op in tokens_pregunta:
                return True
            continue
        if op in pregunta_n:
            return True
    return False


def _bloque_faq_compatible(faq_bloque: str, bloque_elegido: str) -> bool:
    if not faq_bloque or faq_bloque == "general":
        return True
    if bloque_elegido == "general":
        # En modo General permitimos respuestas FAQ de cualquier etapa solo si
        # la pregunta coincide de forma muy clara por términos.
        return True
    return faq_bloque == bloque_elegido


def _faq_tiene_alguno(pregunta_n: str, opciones) -> bool:
    return any(_normalizar_faq(op) in pregunta_n for op in opciones if op)


def _faq_tiene_todos(pregunta_n: str, grupos) -> bool:
    """Cada grupo puede contener varios sinónimos; debe aparecer al menos uno por grupo."""
    return all(_faq_tiene_alguno(pregunta_n, grupo) for grupo in grupos)


def _faq_contiene_contexto_familiar_no_profesional(pregunta_n: str) -> bool:
    """Evita que la FAQ de familias profesionales salte ante preguntas sobre familias de alumnos.

    Ejemplos que NO deben activar la FAQ de número de familias profesionales:
    - familias de alumnos en FP
    - padres divorciados
    - reuniones con familias
    """
    terminos_contexto = [
        "familias de alumnos", "familias del alumnado", "padres", "madres",
        "padre", "madre", "tutor", "tutores", "reunion", "reuniones",
        "divorciado", "divorciados", "custodia", "familia del alumno",
        "familias pueden", "familias participan", "participan", "participar",
        "participacion de familias", "familias en un centro", "en un centro",
        "centro educativo", "centro de fp",
    ]
    return any(t in pregunta_n for t in terminos_contexto)


def _faq_contiene_patron_identificativo(pregunta: str, pregunta_n: str) -> bool:
    """Detecta consultas con datos potencialmente identificativos.

    Criterio v0.4.6:
    - Bloquea nombres propios, DNI/NIE y preguntas explícitas sobre introducir datos
      personales en la app.
    - No bloquea preguntas normativas abstractas ni casos formulados expresamente
      como anonimizados/sin nombres, salvo que incluyan un identificador real o el
      usuario pregunte por introducir datos sensibles concretos.
    """
    pregunta_original = pregunta or ""

    patron_nombre = re.compile(
        r"\b(alumno|alumna|menor|docente|profesor|profesora)\s+"
        r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+"
    )
    hay_nombre_propio = bool(patron_nombre.search(pregunta_original))

    # DNI/NIE con formato real aproximado. No bloquea preguntas abstractas sobre DNI.
    patron_documento = re.compile(r"\b(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b", re.IGNORECASE)
    hay_documento_real = bool(patron_documento.search(pregunta_original))

    if hay_nombre_propio or hay_documento_real:
        return True

    contexto_anonimo = _faq_tiene_alguno(pregunta_n, [
        "sin nombre", "sin nombres", "sin datos", "anonimo", "anonima",
        "anonimizado", "anonimizada", "anonimizar", "sin identificar",
    ])

    habla_app_o_entrada = _faq_tiene_alguno(pregunta_n, [
        "app", "aplicacion", "chat", "meter", "poner", "introducir",
        "subir", "pegar", "enviar", "escribir", "aqui", "caso concreto",
        "que hago con", "que hacer con",
    ])
    habla_datos = _faq_tiene_alguno(pregunta_n, [
        "nombre", "nombres", "apellidos", "dni", "nie", "nif", "datos personales",
        "datos de alumnos", "expediente", "informe medico", "diagnostico",
        "diagnostico medico", "tdah", "tea", "salud", "medico", "medica",
    ])
    habla_persona = _faq_tiene_alguno(pregunta_n, [
        "alumno", "alumnos", "alumna", "alumnas", "menor", "menores",
        "familia", "familias", "docente", "profesor", "profesora",
    ])

    accion_introducir_en_app = _faq_tiene_alguno(pregunta_n, [
        "app", "aplicacion", "chat", "meter", "poner", "introducir",
        "subir", "pegar", "enviar", "escribir", "aqui",
    ])
    pregunta_explicitamente_introducir_datos = habla_app_o_entrada and habla_datos and habla_persona

    # Regla de seguridad adicional v0.4.8:
    # si el usuario pregunta por introducir/subir/pegar en la app documentación o
    # datos sensibles de alumnado, bloqueamos con la FAQ de privacidad incluso si
    # dice que están anonimizados. La anonimización real no puede verificarse aquí.
    dato_o_documento_sensible = _faq_tiene_alguno(pregunta_n, [
        "datos personales", "datos de alumnos", "dato personal", "dni", "nie", "nif",
        "nombre", "nombres", "apellidos", "expediente", "expediente disciplinario",
        "expediente medico", "informe medico", "informe psicopedagogico",
        "diagnostico", "diagnostico medico", "historial medico", "salud",
    ])
    accion_de_entrada = _faq_tiene_alguno(pregunta_n, [
        "app", "aplicacion", "chat", "meter", "poner", "introducir",
        "subir", "pegar", "enviar", "escribir", "aqui",
    ])
    if accion_de_entrada and dato_o_documento_sensible:
        return True

    # Si el usuario ha anonimizado explícitamente y no pregunta por introducir datos
    # o documentación sensible en la app, dejamos que la consulta normativa vaya al RAG.
    # Expresiones como “qué hago con...” no bastan para bloquear cuando la consulta dice “sin nombres”.
    if contexto_anonimo and not (accion_introducir_en_app and habla_datos and habla_persona):
        return False

    if pregunta_explicitamente_introducir_datos:
        return True

    # Expresiones explícitas de identificación aunque no haya nombre propio escrito.
    if _faq_tiene_alguno(pregunta_n, ["nombres de alumnos", "nombre de un alumno", "apellidos del alumno"]):
        return True

    if _faq_tiene_todos(pregunta_n, [
        ["alumno", "alumna", "menor", "docente", "profesor", "profesora"],
        ["llamado", "llamada", "se llama", "nombre", "apellidos"],
    ]):
        return True

    return False


# ============================================================
# v073b post-validación - prioridades FAQ para variantes duplicadas
# ============================================================
# La validación sintética post-v073b detectó variantes exactas presentes en dos
# FAQ verificadas. Para no depender del orden del JSON, se fija aquí la FAQ
# preferente solo para esas formulaciones exactas. No rebaja umbrales ni amplía
# el matching difuso; únicamente resuelve solapamientos controlados.
_FAQ_VARIANTES_PRIORITARIAS_POSTVALIDACION = {
    _normalizar_faq("consejo orientador eso"): "eso_consejo_orientador",
    _normalizar_faq("modalidades bachillerato"): "bachillerato_modalidades",
    _normalizar_faq("cuáles son las modalidades de bachillerato"): "bachillerato_modalidades",
    _normalizar_faq("que modalidades de bachillerato existen"): "bachillerato_modalidades_basica",
    _normalizar_faq("qué modalidades hay en bachillerato"): "bachillerato_modalidades",
    # r7 mínima: variante abreviada detectada en piloto exhaustivo.
    _normalizar_faq("modalidades bach"): "bachillerato_modalidades",
    _normalizar_faq("modalidades bachiller"): "bachillerato_modalidades",
    _normalizar_faq("con cuántas materias se promociona de 1º a 2º de bachillerato"): "bachillerato_promocion_dos_materias",
    _normalizar_faq("con cuántas materias se promociona de 1.º a 2.º de bachillerato"): "bachillerato_promocion_dos_materias",
    _normalizar_faq("con dos materias suspensas se puede pasar de 1º a 2º de bachillerato"): "bachillerato_promocion_dos_materias",
    _normalizar_faq("con dos materias suspensas se puede pasar de 1.º a 2.º de bachillerato"): "bachillerato_promocion_dos_materias",
    _normalizar_faq("se puede pasar de primero a segundo de bachillerato con dos materias suspensas"): "bachillerato_promocion_dos_materias",
    _normalizar_faq("se puede repetir 1º de bachillerato"): "bachillerato_permanencia_cuatro_anos",
    _normalizar_faq("se puede repetir 1.º de bachillerato"): "bachillerato_permanencia_cuatro_anos",
    _normalizar_faq("se puede repetir 2º de bachillerato"): "bachillerato_permanencia_cuatro_anos",
    _normalizar_faq("se puede repetir 2.º de bachillerato"): "bachillerato_permanencia_cuatro_anos",
    _normalizar_faq("seguimiento alumnado empresa fp"): "fp_tutor_empresa_seguimiento_contacto",
    _normalizar_faq("cuántos días de permiso tiene un docente por hospitalización de su padre"): "permiso_hospitalizacion_padre",
    _normalizar_faq("cuantos dias de permiso tiene un docente por hospitalizacion de su padre"): "permiso_hospitalizacion_padre",
    _normalizar_faq("cuáles son las áreas de infantil"): "infantil_areas",
    _normalizar_faq("cuales son las areas de infantil"): "infantil_areas",
    _normalizar_faq("qué significan in su bi nt y sb en las calificaciones"): "primaria_calificaciones_siglas_in_su_bi_nt_sb",
    _normalizar_faq("que significan in su bi nt y sb en las calificaciones"): "primaria_calificaciones_siglas_in_su_bi_nt_sb",
    # r6 mínima: incidencias menores detectadas en piloto interno r5.
    _normalizar_faq("cuándo disfrutan las vacaciones los docentes"): "vacaciones_docentes_agosto",
    _normalizar_faq("cuando disfrutan las vacaciones los docentes"): "vacaciones_docentes_agosto",
    _normalizar_faq("qué hace el alumnado que no elige religión en primaria"): "primaria_atencion_educativa_no_religion",
    _normalizar_faq("que hace el alumnado que no elige religion en primaria"): "primaria_atencion_educativa_no_religion",
    _normalizar_faq("qué hace el alumnado que no cursa religión en primaria"): "primaria_atencion_educativa_no_religion",
    _normalizar_faq("que hace el alumnado que no cursa religion en primaria"): "primaria_atencion_educativa_no_religion",
    _normalizar_faq("la fp puede impartirse en modalidad virtual"): "fp_modalidades_presencial_semipresencial_virtual",
    _normalizar_faq("puede impartirse la fp en modalidad virtual"): "fp_modalidades_presencial_semipresencial_virtual",
    _normalizar_faq("se puede cursar fp virtual"): "fp_modalidades_presencial_semipresencial_virtual",
}

def _faq_match_exacta(pregunta_n: str, bloque_elegido: str):
    """Coincidencia exacta contra pregunta canónica o variantes normalizadas.

    Esto garantiza que las FAQ verificadas respondan a sus propias variantes
    previstas, sin rebajar el umbral global ni aumentar falsos positivos.
    """
    if not pregunta_n:
        return None

    # v073b post-validación: si una variante exacta está duplicada en dos FAQ,
    # se usa la prioridad explícita definida arriba. Esto evita que el resultado
    # dependa del orden físico de las FAQ en faq_normativa.json.
    faq_id_prioritaria = _FAQ_VARIANTES_PRIORITARIAS_POSTVALIDACION.get(pregunta_n)
    if faq_id_prioritaria:
        faq_prioritaria = _buscar_faq_por_id(faq_id_prioritaria)
        if _faq_bloque_intencion_ok(faq_prioritaria, bloque_elegido):
            return faq_prioritaria

    for faq in faq_normativa:
        if not _bloque_faq_compatible(faq.get("bloque", ""), bloque_elegido):
            continue
        textos = [faq.get("pregunta_canonica", "")] + faq.get("variantes", [])
        for texto in textos:
            if _normalizar_faq(texto) == pregunta_n:
                return faq
    return None

def _buscar_faq_por_id(faq_id: str):
    for faq in faq_normativa:
        if faq.get("id") == faq_id:
            return faq
    return None


def _faq_bloque_intencion_ok(faq: dict, bloque_elegido: str) -> bool:
    return bool(faq) and _bloque_faq_compatible(faq.get("bloque", ""), bloque_elegido)


def _faq_pregunta_fp_especifica_debe_ir_a_rag(pregunta_n: str) -> bool:
    """v072: evita que una FAQ genérica sobre el RD 659/2023 capture preguntas
    específicas sobre FCT, formación en centros de trabajo o requisitos.

    La FAQ `fp_norma_estatal_rd659` debe responder a "qué norma regula la FP",
    no a preguntas que piden requisitos o contenido concreto.
    """
    if not pregunta_n:
        return False

    menciona_fct_o_formacion = _faq_tiene_alguno(pregunta_n, [
        "fct",
        "formacion en centros de trabajo",
        "centros de trabajo",
        "formacion en centro de trabajo",
        "formacion en empresa",
        "empresa u organismo equiparado",
        "modulo de fct",
    ])

    pide_detalle = _faq_tiene_alguno(pregunta_n, [
        "requisito",
        "requisitos",
        "condiciones",
        "que debe contener",
        "que contiene",
        "contenido",
        "menciona",
        "como se realiza",
        "como debe realizarse",
        "duracion",
        "periodo",
        "horas",
        "evaluacion",
    ])

    # Si la pregunta pide expresamente la norma, dejamos actuar a la FAQ normativa.
    pide_norma = _faq_tiene_todos(pregunta_n, [
        ["norma", "real decreto", "regula"],
        ["fp", "formacion profesional"],
    ]) and not pide_detalle

    return menciona_fct_o_formacion and pide_detalle and not pide_norma


def _faq_pide_garantias_procedimiento_convivencia(pregunta_n: str) -> bool:
    """r7 mínima: preguntas sobre audiencia/defensa en sanciones de convivencia
    no deben resolverse con una FAQ genérica de sanciones. Van a RAG/prudencia
    para evitar responder con una lista de sanciones cuando se piden garantías
    procedimentales.
    """
    if not pregunta_n:
        return False
    contexto = _faq_tiene_alguno(pregunta_n, [
        "sancion", "sanciones", "procedimiento", "corrector", "correccion", "convivencia",
        "expediente", "conducta", "falta", "disciplinario", "disciplina",
    ])
    garantias = _faq_tiene_alguno(pregunta_n, [
        "sin oir", "ser oido", "ser oida", "oir al alumno", "oir a la familia",
        "audiencia", "alegaciones", "defensa", "derecho a la defensa", "prueba",
        "recurso", "impugnacion", "motivacion", "familia", "padres", "tutores",
        "tutor legal", "tutores legales", "garantias", "procedimentales", "procesales",
    ])
    return contexto and garantias


def _faq_match_reglas_intencion(pregunta: str, bloque_elegido: str):
    """Reglas conservadoras para preguntas frecuentes formuladas de forma liosa.

    No sustituyen al matcher general: solo cubren casos muy acotados donde
    exigir similitud textual alta genera falsos negativos, pero los términos
    presentes dejan clara la intención del usuario.
    """
    p = _normalizar_faq(pregunta)

    # r5 mínima: refuerzos FAQ detectados en piloto de 25 preguntas reales.
    # Son reglas cerradas para evitar consumo innecesario de RAG/IA en preguntas FAQ claras.
    # r6 mínima: refuerzos finos detectados en piloto interno r5.
    if _faq_tiene_todos(p, [["vacaciones"], ["docente", "docentes", "profesor", "profesores", "profesorado"]]):
        faq = _buscar_faq_por_id("vacaciones_docentes_agosto")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["religion"], ["no elige", "no cursa", "no da", "no recibe", "alternativa", "atencion educativa"], ["primaria", "educacion primaria", "alumnado", "alumno", "alumnos"]]):
        faq = _buscar_faq_por_id("primaria_atencion_educativa_no_religion")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["fp", "formacion profesional"], ["virtual", "online", "semipresencial", "modalidad", "modalidades"]]):
        faq = _buscar_faq_por_id("fp_modalidades_presencial_semipresencial_virtual")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["areas", "area"], ["infantil", "educacion infantil"]]):
        faq = _buscar_faq_por_id("infantil_areas")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["in", "su", "bi", "nt", "sb"], ["calificaciones", "notas"]]) and bloque_elegido == "infantil_primaria":
        faq = _buscar_faq_por_id("primaria_calificaciones_siglas_in_su_bi_nt_sb")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["permiso", "dias", "dia"], ["hospitalizacion", "hospitalizado", "ingresado", "hospital"], ["padre", "madre", "progenitor"]]):
        faq = _buscar_faq_por_id("permiso_hospitalizacion_padre")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["bachillerato", "bachiller", "bach"], ["modalidades", "modalidad", "tipos"]]):
        faq = _buscar_faq_por_id("bachillerato_modalidades")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["bachillerato", "bachiller", "bach"], ["pasar", "promocionar", "promocion"], ["dos materias", "2 materias", "dos suspensas", "2 suspensas", "materias suspensas"]]):
        faq = _buscar_faq_por_id("bachillerato_promocion_dos_materias")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    # v073b: las preguntas sobre "prueba objetiva tipo test" y
    # "evaluación objetiva" deben ir a la FAQ verificada de derecho a
    # evaluación objetiva, no depender de Qdrant.
    if _faq_tiene_alguno(p, [
        "evaluacion objetiva",
        "prueba objetiva",
        "tipo test",
        "test cuenta como evaluacion objetiva",
        "examen tipo test",
    ]) and _faq_tiene_alguno(p, [
        "cuenta como",
        "es evaluacion objetiva",
        "evaluacion objetiva",
        "criterios",
        "calificacion",
        "reclamacion",
        "revisar",
    ]):
        faq = _buscar_faq_por_id("alumnado_derecho_evaluacion_objetiva")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    # v069: variantes sintéticas prioritarias, muy acotadas, antes del matcher general.
    if _faq_tiene_alguno(p, ["que areas hay en educacion infantil", "infantil areas nombres", "nombres de las areas de infantil"]):
        faq = _buscar_faq_por_id("infantil_areas")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["sexto", "6"], ["primaria"], ["valores"]]):
        faq = _buscar_faq_por_id("primaria_asignaturas_sexto_valores")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["promocionar", "promocion"], ["primaria"], ["final de ciclo", "ciclo"]]):
        faq = _buscar_faq_por_id("primaria_promocion_final_ciclo_cyl")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["repite", "repetir", "repeticion"], ["primaria"], ["final de ciclo", "ciclo"]]):
        faq = _buscar_faq_por_id("primaria_repetir_cuando_condiciones_cyl")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["el consejo orientador que es", "que es el consejo orientador", "consejo orientador"]):
        faq = _buscar_faq_por_id("eso_consejo_orientador")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["evaluacion"], ["objetiva"], ["eso", "secundaria"]]):
        faq = _buscar_faq_por_id("eso_evaluacion_caracter_objetiva_rd")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["calificaciones", "notas"], ["cualitativas", "insuficiente", "suficiente", "bien", "notable", "sobresaliente"], ["eso"]]):
        faq = _buscar_faq_por_id("eso_calificaciones_cualitativas_cyl")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["bachiller", "bachillerato"], ["fp", "formacion profesional", "tecnico"]]):
        if _faq_tiene_alguno(p, ["desde fp", "desde formacion profesional", "titulo de bachiller", "obtener bachiller", "bachiller desde"]):
            faq = _buscar_faq_por_id("bachillerato_titulo_desde_fp")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    if _faq_tiene_todos(p, [["bachiller", "bachillerato"], ["promocion", "promocionar"], ["dos materias", "2 materias", "dos suspensas", "2 suspensas"]]):
        faq = _buscar_faq_por_id("bachillerato_promocion_dos_materias")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["titulo", "titulacion"], ["bachiller", "bachillerato"], ["regla general", "todas las materias"]]):
        faq = _buscar_faq_por_id("bachillerato_titulo_regla_general_todas_materias")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["certificado de competencia"], ["fp", "formacion profesional"]]):
        faq = _buscar_faq_por_id("fp_grado_b_certificado_competencia")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["certificado profesional"], ["fp", "formacion profesional"]]):
        faq = _buscar_faq_por_id("fp_grado_c_certificado_profesional")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["cursos de especializacion", "curso de especializacion"], ["fp", "formacion profesional"]]):
        faq = _buscar_faq_por_id("fp_grado_e")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["porcentaje"], ["empresa"], ["intensivo", "intensiva"], ["fp", "formacion profesional"]]):
        faq = _buscar_faq_por_id("fp_formacion_empresa_regimen_intensivo_porcentaje")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["ofertas modulares", "oferta modular"], ["fp", "formacion profesional"]]):
        faq = _buscar_faq_por_id("fp_ofertas_modulares")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["ciclos"], ["basico", "medio", "superior"], ["definicion", "que son"]]):
        faq = _buscar_faq_por_id("fp_ciclos_basico_medio_superior_definicion")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["sanciones"], ["faltas graves", "gravemente perjudiciales", "articulo 49"]]):
        faq = _buscar_faq_por_id("cyl_sanciones_faltas_graves_art49")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    # v072: "medidas inmediatas" debe tratarse como equivalente a
    # "actuaciones inmediatas" de convivencia.
    if _faq_tiene_todos(p, [
        ["medidas inmediatas", "actuaciones inmediatas"],
        ["conducta", "convivencia", "perturba", "perturbadora"],
    ]):
        faq = _buscar_faq_por_id("cyl_actuaciones_inmediatas_convivencia")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["mediacion"], ["escolar", "convivencia"]]):
        faq = _buscar_faq_por_id("cyl_mediacion_escolar_convivencia")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["permiso"], ["examen", "examenes", "oficiales"]]):
        faq = _buscar_faq_por_id("permiso_examenes")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["mi hijo repite primaria", "hijo repite primaria", "hija repite primaria"]):
        faq = _buscar_faq_por_id("primaria_repeticion")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["con dos suspensas paso bach", "con 2 suspensas paso bach", "dos suspensas paso bachillerato"]):
        faq = _buscar_faq_por_id("bachillerato_repeticion_promocion_basica")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["nino pega en clase que hago", "niño pega en clase que hago", "alumno pega en clase"]):
        faq = _buscar_faq_por_id("cyl_actuaciones_inmediatas_convivencia")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["puedo pegar informe medico alumno", "puedo pegar un informe medico de un alumno", "datos medicos concretos"]):
        faq = _buscar_faq_por_id("privacidad_no_datos_personales_app")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    # v065: defensiva FP y refuerzos cortos prioritarios.
    if _faq_tiene_todos(p, [["ciclo", "ciclos"], ["familia profesional", "familias profesionales"]]):
        if _faq_tiene_alguno(p, ["exactamente", "dentro de cada", "por familia", "cada familia", "numero", "número", "cuantos", "cuántos"]):
            faq = _buscar_faq_por_id("fp_ciclos_por_familia_no_estable")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    if _faq_tiene_alguno(p, ["seguimiento alumnado empresa fp", "tutor empresa fp", "contacto centro empresa fp", "centro empresa fp"]):
        faq = _buscar_faq_por_id("fp_tutor_empresa_seguimiento_contacto")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["alumno no deja dar clase", "alumno molesta en clase", "alumno interrumpe clase", "alumno impide la clase"]):
        faq = _buscar_faq_por_id("cyl_actuaciones_inmediatas_convivencia")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["como se organiza primaria", "organizacion primaria", "ciclos primaria", "primaria ciclos"]):
        faq = _buscar_faq_por_id("primaria_ciclos")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["asignaturas educacion infantil", "asignaturas de infantil", "materias infantil", "ambitos infantil"]):
        faq = _buscar_faq_por_id("infantil_areas")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["finalidad educacion infantil", "para que sirve la educacion infantil", "fines infantil", "fines de infantil"]):
        faq = _buscar_faq_por_id("infantil_finalidad")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["evaluacion educacion infantil", "evaluacion infantil", "como se evalua en infantil"]):
        faq = _buscar_faq_por_id("infantil_evaluacion")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0


    # v064: variantes coloquiales detectadas en pruebas reales.
    # Regla defensiva: trabajar en un hospital no equivale a hospitalización.
    if _faq_tiene_todos(p, [["trabaja", "trabajan", "trabajar"], ["hospital", "hospitales"]]):
        faq = _buscar_faq_por_id("permiso_no_por_trabajar_en_hospital")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    # ESO: "con suspensos se puede pasar de curso".
    if _faq_tiene_alguno(p, ["eso", "educacion secundaria obligatoria"]):
        if _faq_tiene_todos(p, [["suspenso", "suspensos", "suspensa", "suspensas", "suspendidas", "materias no superadas"], ["pasar de curso", "pasar", "promocionar", "promocion"]]):
            faq = _buscar_faq_por_id("eso_promocion_repeticion_basica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # Bachillerato coloquial: "qué bachilleratos existen" y "pasar con suspensas".
    if _faq_tiene_alguno(p, ["bachillerato", "bachilleratos", "bachiller"]):
        if _faq_tiene_alguno(p, ["que bachilleratos existen", "que bachilleratos hay", "tipos de bachillerato", "bachilleratos existen", "bachilleratos hay"]):
            faq = _buscar_faq_por_id("bachillerato_modalidades_basica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["suspensa", "suspensas", "suspenso", "suspensos", "materias no superadas"], ["pasar", "promocionar", "promocion", "primero", "segundo", "1", "2"]]):
            faq = _buscar_faq_por_id("bachillerato_repeticion_promocion_basica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # FP coloquial sin mencionar siempre "formación profesional".
    if _faq_tiene_todos(p, [["grado medio"], ["titulo", "título", "da", "obtiene", "exactamente"]]):
        faq = _buscar_faq_por_id("fp_titulos_basico_medio_superior")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_todos(p, [["tecnico", "técnico"], ["bachillerato", "bachiller"]]):
        faq = _buscar_faq_por_id("fp_titulo_tecnico_acceso_bachillerato")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    if _faq_tiene_alguno(p, ["fp general e intensiva", "fp general intensiva", "fp intensiva", "fp general"]):
        if _faq_tiene_alguno(p, ["diferencia", "diferencias", "que es", "son lo mismo", "intensiva", "general"]):
            faq = _buscar_faq_por_id("fp_regimen_general_intensivo")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # Permiso por fallecimiento en lenguaje cotidiano.
    if _faq_tiene_todos(p, [["fallece", "fallecimiento", "muere", "muerte"], ["padre", "madre", "familiar", "primer grado"]]):
        faq = _buscar_faq_por_id("permiso_fallecimiento_primer_grado")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    # Convivencia: alumno que no deja dar clase.
    if _faq_tiene_todos(p, [["alumno", "alumna"], ["no deja dar clase", "interrumpe", "impide", "molesta", "dar clase"]]):
        faq = _buscar_faq_por_id("cyl_actuaciones_inmediatas_convivencia")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0


    # v062: FAQ básicas de cobertura general.
    if _faq_tiene_alguno(p, ["primaria", "educacion primaria"]):
        if _faq_tiene_alguno(p, ["asignaturas", "materias", "areas"]):
            if _faq_tiene_alguno(p, ["6", "6o", "sexto", "valores civicos", "valores civicos y eticos"]):
                faq = _buscar_faq_por_id("primaria_asignaturas_sexto_valores")
            else:
                faq = _buscar_faq_por_id("primaria_asignaturas_primero_quinto")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["religion", "religión"]):
            if _faq_tiene_alguno(p, ["no cursa", "no recibe", "alternativa", "atencion educativa", "atención educativa"]):
                faq = _buscar_faq_por_id("primaria_atencion_educativa_no_religion")
            else:
                faq = _buscar_faq_por_id("primaria_religion_oferta_voluntaria")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["in su bi nt sb", "calificaciones", "notas"]):
            faq = _buscar_faq_por_id("primaria_calificaciones_siglas_in_su_bi_nt_sb")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["evaluacion"], ["finalidad", "continua", "global", "formativa"]]):
            faq = _buscar_faq_por_id("primaria_evaluacion_continua_finalidad")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["dificultades", "medidas de refuerzo", "refuerzo"]):
            faq = _buscar_faq_por_id("primaria_refuerzo_dificultades_aprendizaje")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["norma", "real decreto", "regula"], ["estatal", "ordenacion", "ordenación"]]):
            faq = _buscar_faq_por_id("primaria_norma_estatal_rd157")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    if _faq_tiene_alguno(p, ["eso", "educacion secundaria obligatoria"]):
        if _faq_tiene_alguno(p, ["edades", "edad"]):
            faq = _buscar_faq_por_id("eso_edades_comprende")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["materias", "asignaturas"]):
            if _faq_tiene_alguno(p, ["4", "4o", "cuarto"]):
                faq = _buscar_faq_por_id("eso_materias_cuarto")
            else:
                faq = _buscar_faq_por_id("eso_materias_primero_tercero")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["repetir", "repeticion", "cuantas veces", "promocionar", "promocion", "materias suspensas", "quien decide"]):
            faq = _buscar_faq_por_id("eso_promocion_repeticion_basica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["titulo", "titular", "graduado", "materias suspensas"]):
            faq = _buscar_faq_por_id("eso_titulacion_materias_suspensas")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["documentos oficiales", "consejo orientador", "diagnostico"]):
            faq = _buscar_faq_por_id("eso_documentos_consejo_diagnostico")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    if _faq_tiene_alguno(p, ["bachillerato", "bachiller", "bach"]):
        if _faq_tiene_alguno(p, ["modalidades", "modalidad"]):
            faq = _buscar_faq_por_id("bachillerato_modalidades_basica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["materias", "asignaturas"]):
            faq = _buscar_faq_por_id("bachillerato_materias_comunes_primero_segundo")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["promociona", "promocion", "repetir", "repite", "permanecer"]):
            faq = _buscar_faq_por_id("bachillerato_repeticion_promocion_basica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["norma", "real decreto", "regula"], ["estatal", "bachillerato"]]):
            faq = _buscar_faq_por_id("bachillerato_norma_estatal_rd243")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    if _faq_tiene_alguno(p, ["fp", "formacion profesional"]):
        if _faq_tiene_alguno(p, ["familia profesional", "familias profesionales", "informatica y comunicaciones"]):
            faq = _buscar_faq_por_id("fp_familia_profesional_definicion_basica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["ciclo formativo de grado basico", "ciclo formativo de grado medio", "ciclo formativo de grado superior"]):
            faq = _buscar_faq_por_id("fp_ciclos_basico_medio_superior_definicion")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["titulo"], ["grado basico", "grado medio", "grado superior", "tecnico", "tecnico superior"]]):
            faq = _buscar_faq_por_id("fp_titulos_basico_medio_superior")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["tecnico"], ["bachillerato", "bachiller"]]):
            faq = _buscar_faq_por_id("fp_titulo_tecnico_acceso_bachillerato")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["grados a b c d e", "grados a b c d y e"]):
            faq = _buscar_faq_por_id("fp_grados_abcde_conjunto")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["empresa", "empresas", "organismo equiparado", "centro y la empresa"], ["formacion", "seguimiento", "tutor", "contacto", "plan"]]):
            if _faq_tiene_alguno(p, ["regimen general", "regimen intensivo", "régimen general", "régimen intensivo"]):
                faq = _buscar_faq_por_id("fp_regimen_general_intensivo")
            elif _faq_tiene_alguno(p, ["seguimiento", "tutor de empresa", "contacto"]):
                faq = _buscar_faq_por_id("fp_tutor_empresa_seguimiento_contacto")
            else:
                faq = _buscar_faq_por_id("fp_formacion_empresa_obligatoria_plan")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["norma", "real decreto", "regula"], ["fp", "formacion profesional"]]):
            if not _faq_pregunta_fp_especifica_debe_ir_a_rag(p):
                faq = _buscar_faq_por_id("fp_norma_estatal_rd659")
                if _faq_bloque_intencion_ok(faq, bloque_elegido):
                    return faq, 1.0

    if _faq_tiene_alguno(p, ["fuente oficial", "fuentes consultadas", "boe", "bocyl"]):
        faq = _buscar_faq_por_id("uso_app_fuentes_oficiales")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0


    # 0) Reglas directas añadidas v0.5.0 para cubrir fallos reales de recuperación.
    # Convivencia, derechos/deberes y sanciones del alumnado en Castilla y León.
    if _faq_tiene_alguno(p, ["castilla y leon", "castilla leon", "cyl", "centros educativos", "alumnado", "alumno", "alumnos"]):
        if _faq_tiene_todos(p, [["convivencia", "disciplina"], ["norma", "regula", "decreto"]]):
            faq = _buscar_faq_por_id("cyl_convivencia_norma_decreto51")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["derechos", "deberes"], ["sanciones", "convivencia", "disciplina"]]):
            faq = _buscar_faq_por_id("cyl_derechos_deberes_sanciones_norma")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["sancion", "sanciones", "faltas graves", "faltas"], ["alumnado", "alumno", "convivencia"]]):
            faq = _buscar_faq_por_id("cyl_sanciones_faltas_graves_art49")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["conductas contrarias", "conducta contraria", "normas de convivencia"]):
            faq = _buscar_faq_por_id("cyl_conductas_contrarias_convivencia")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["gravemente perjudicial", "faltas graves", "conductas graves"]):
            faq = _buscar_faq_por_id("cyl_conductas_gravemente_perjudiciales")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["actuaciones inmediatas", "amonestacion", "peticion de disculpas", "trabajos especificos"]):
            faq = _buscar_faq_por_id("cyl_actuaciones_inmediatas_convivencia")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["mediacion escolar", "mediacion", "mediador"]):
            faq = _buscar_faq_por_id("cyl_mediacion_escolar_convivencia")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["acuerdo reeducativo", "acuerdos reeducativos"]):
            faq = _buscar_faq_por_id("cyl_acuerdo_reeducativo_convivencia")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["criterios", "proporcionalidad", "dignidad", "derecho a la educacion"], ["correcciones", "sanciones", "disciplinarias"]]):
            faq = _buscar_faq_por_id("cyl_criterios_correcciones_alumnado")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # Primaria: repetición, promoción y plan de refuerzo.
    if _faq_tiene_alguno(p, ["primaria", "educacion primaria"]):
        if _faq_tiene_alguno(p, ["primero tercero quinto", "1 3 5", "promocion automatica", "promociona automaticamente"]):
            faq = _buscar_faq_por_id("primaria_promocion_fin_ciclo_automatica")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["repetir", "permanece", "permanecer", "repeticion"], ["cuantas veces", "una vez", "veces", "etapa"]]):
            faq = _buscar_faq_por_id("primaria_repetir_una_vez_etapa_condicion")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["no promociona", "repite", "repetir", "permanece"], ["plan de refuerzo", "plan especifico", "plan especifico de refuerzo", "refuerzo", "apoyos", "padres", "tutores"]]) and not _faq_tiene_alguno(p, ["sin repetir", "sin repeticion", "sin que repita"]):
            faq = _buscar_faq_por_id("primaria_no_promocion_plan_refuerzo")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["curso", "cursos", "segundo", "cuarto", "sexto", "2", "4", "6", "final de ciclo"], ["repeticion", "repetir", "permanencia", "decidirse", "decidir"]]):
            faq = _buscar_faq_por_id("primaria_repetir_cuando_condiciones_cyl")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["cuando se puede repetir", "cuando puede repetir", "cuando se repite", "permanecer un ano mas", "repeticion primaria"]):
            faq = _buscar_faq_por_id("primaria_repetir_cuando_condiciones_cyl")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # Derecho transversal a evaluación objetiva.
    # Cubre formulaciones generales como “qué garantías tiene un alumno para que su evaluación sea objetiva”,
    # que antes podían no activar FAQ por no mencionar explícitamente “derecho” o “Bachillerato”.
    # Evita falsos positivos de “prueba objetiva” exigiendo contexto de garantías/derechos/alumnado.
    if _faq_tiene_alguno(p, ["evaluacion objetiva", "evaluacion sea objetiva", "objetividad en la evaluacion", "objetividad de la evaluacion", "valorados con objetividad"]):
        if _faq_tiene_alguno(p, ["garantia", "garantias", "derecho", "alumno", "alumnado", "estudiante", "estudiantes", "esfuerzo", "rendimiento", "valorados", "reclamacion", "aclaraciones"]):
            if _faq_tiene_alguno(p, ["bachillerato", "bachiller", "bach"]):
                faq = _buscar_faq_por_id("bachillerato_evaluacion_objetiva_art11")
            else:
                faq = _buscar_faq_por_id("alumnado_derecho_evaluacion_objetiva")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # Bachillerato: evaluación objetiva, aclaraciones, reclamaciones y documentos.
    if _faq_tiene_alguno(p, ["bachillerato", "bachiller", "bach"]):
        if _faq_tiene_todos(p, [["evaluacion objetiva", "objetiva", "objetividad"], ["derecho", "valorados", "esfuerzo", "rendimiento"]]):
            faq = _buscar_faq_por_id("bachillerato_evaluacion_objetiva_art11")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["aclaraciones", "pedir aclaraciones", "solicitar aclaraciones"]):
            faq = _buscar_faq_por_id("bachillerato_aclaraciones_evaluacion")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["reclamar", "reclamacion", "reclama"], ["nota", "calificacion", "evaluacion", "promocion", "titulacion"]]):
            if _faq_tiene_alguno(p, ["plazo", "dias", "dos dias", "2 dias", "hasta cuando"]):
                faq = _buscar_faq_por_id("bachillerato_plazo_reclamacion_dos_dias")
            else:
                faq = _buscar_faq_por_id("bachillerato_reclamacion_calificaciones_centro")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["documentos oficiales", "actas", "expediente", "historial"], ["evaluacion", "documentos", "papeles"]]):
            faq = _buscar_faq_por_id("bachillerato_documentos_oficiales_rd243")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # FP: seguimiento en empresa, plan formativo y tutor dual.
    if _faq_tiene_alguno(p, ["fp", "formacion profesional"]):
        if _faq_tiene_todos(p, [["empresa", "formacion en empresa"], ["seguimiento", "supervision", "supervisa", "quien hace", "quien realiza"]]):
            faq = _buscar_faq_por_id("fp_tutor_dual_empresa")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["contacto continuo", "coordinacion", "relacion"], ["centro", "empresa"]]):
            faq = _buscar_faq_por_id("fp_contacto_continuo_centro_empresa")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["plan formativo", "plan de formacion"], ["firma", "firmado", "quien firma"]]):
            faq = _buscar_faq_por_id("fp_plan_formacion_empresa_firmas")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_todos(p, [["tutor dual", "tutora dual"], ["centro", "funciones", "que hace"]]):
            faq = _buscar_faq_por_id("fp_tutor_dual_centro_funciones")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # 1) Número de familias profesionales de FP.
    # Debe ser muy claro que "familias" significa familias profesionales, no familias del alumnado.
    if _faq_tiene_alguno(p, ["fp", "formacion profesional"]) or bloque_elegido == "fp":
        # Reglas de acceso: deben ir antes de las reglas genéricas de “grado”,
        # porque “acceder a grado medio/superior” pregunta por requisitos,
        # no por la definición del grado.
        habla_acceso = _faq_tiene_alguno(p, ["acceder", "acceso", "entrar", "requisitos", "necesito", "puedo entrar"])
        if habla_acceso and _faq_tiene_alguno(p, ["grado medio", "ciclo formativo de grado medio"]):
            faq = _buscar_faq_por_id("fp_acceso_grado_medio_requisitos")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if habla_acceso and _faq_tiene_alguno(p, ["grado superior", "ciclo formativo de grado superior"]):
            faq = _buscar_faq_por_id("fp_acceso_grado_superior_requisitos")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

        # Duración del Grado E: debe ganar frente a la FAQ general de “qué es Grado E”.
        if _faq_tiene_alguno(p, ["grado e"]) and _faq_tiene_alguno(p, ["dura", "duracion", "cuanto dura", "cuantas horas", "horas"]):
            faq = _buscar_faq_por_id("fp_grado_e_duracion_cursos_especializacion")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

        # Reglas directas para Grados A, B, C, D y E. Se evalúan antes de otras
        # reglas de FP para evitar que una letra suelta active el Grado D/E.
        grados_directos = [
            ("grado a", "fp_grado_a_microacreditacion"),
            ("grado b", "fp_grado_b_certificado_competencia"),
            ("grado c", "fp_grado_c_certificado_profesional"),
            ("grado d", "fp_grado_d"),
            ("grado e", "fp_grado_e"),
        ]
        p_pad_fp = f" {p} "
        for frase_grado, faq_id in grados_directos:
            # Debe aparecer la expresión exacta "grado a/b/c/d/e" como palabras.
            # Evitamos falsos positivos como "grado de satisfacción" y
            # "grado a distancia" (a = preposición, no Grado A).
            if f" {frase_grado} " in p_pad_fp:
                if frase_grado == "grado a" and _faq_tiene_alguno(p, ["grado a distancia", "grado a partir"]):
                    continue
                faq = _buscar_faq_por_id(faq_id)
                if _faq_bloque_intencion_ok(faq, bloque_elegido):
                    return faq, 1.0

        pide_numero = _faq_tiene_alguno(p, ["cuantas", "cuantos", "numero", "13", "26", "hay", "existen", "me dices cuantas"])
        menciona_familias_profesionales = _faq_tiene_alguno(p, ["familias profesionales", "familia profesional"])
        menciona_familias_en_fp = _faq_tiene_alguno(p, ["familias"]) and not _faq_contiene_contexto_familiar_no_profesional(p)
        pregunta_sobre_ciclos_por_familia = _faq_tiene_todos(p, [
            ["ciclos", "ciclo"],
            ["cada familia", "por familia", "de cada familia"],
        ])
        if pide_numero and not pregunta_sobre_ciclos_por_familia and (menciona_familias_profesionales or menciona_familias_en_fp):
            faq = _buscar_faq_por_id("fp_numero_familias")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

        # Duración de ciclos de grado medio/superior. Regla separada para evitar
        # que preguntas naturales tipo "cuántos años dura un grado medio" se pierdan,
        # sin activar respuestas sobre requisitos o familias profesionales.
        habla_grado_ms = _faq_tiene_alguno(p, ["grado medio", "grado superior", "ciclo formativo", "ciclos formativos"])
        pide_duracion = _faq_tiene_alguno(p, ["dura", "duran", "duracion", "cuanto tiempo", "cuantos anos", "cuantas horas", "anos dura"])
        if habla_grado_ms and pide_duracion:
            faq = _buscar_faq_por_id("fp_grado_medio_superior_duracion")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # 2) Calificaciones cualitativas de Primaria en Castilla y León.
    if _faq_tiene_alguno(p, ["primaria", "educacion primaria"]):
        if _faq_tiene_alguno(p, ["documentos oficiales", "actas", "expediente academico", "historial academico", "informe final"]):
            faq = _buscar_faq_por_id("primaria_documentos_evaluacion")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

        p_pad = f" {p} "
        menciona_codigos = (
            "in su bi nt sb" in p
            or "insuficiente suficiente bien notable sobresaliente" in p
            or all(f" {cod} " in p_pad for cod in ["in", "su", "bi", "nt", "sb"])
        )
        habla_notas = _faq_tiene_alguno(p, ["calificaciones", "calificacion", "notas", "evaluar", "evaluacion", "de donde sale"])
        if menciona_codigos and habla_notas:
            faq = _buscar_faq_por_id("primaria_calificaciones_cualitativas_cyl")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # 2) Evaluación de Bachillerato en Castilla y León.
    # Permitimos que "Orden EDU/425/2024" active la FAQ aunque el usuario no escriba CYL.
    if _faq_tiene_alguno(p, ["bachillerato", "bachiller", "bach"]):
        if _faq_tiene_alguno(p, ["documentos oficiales", "actas", "expediente academico", "historial academico", "informe personal"]):
            faq = _buscar_faq_por_id("bachillerato_documentos_evaluacion")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

        # Título de Bachiller desde Técnico de FP.
        if _faq_tiene_alguno(p, ["tecnico", "titulado tecnico", "titulo de tecnico"]) and _faq_tiene_alguno(p, ["fp", "formacion profesional"]):
            faq = _buscar_faq_por_id("bachillerato_titulo_desde_fp")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

        # Reglas específicas para evitar que una FAQ general de promoción secuestre
        # preguntas sobre dos materias pendientes o titulación con una materia.
        if _faq_tiene_alguno(p, ["titular", "titulo", "obtener titulo", "obtener bachillerato"]) and _faq_tiene_alguno(p, ["una materia", "1 materia", "materia suspensa", "una suspensa", "una asignatura", "asignatura suspensa", "asignatura pendiente"]):
            faq = _buscar_faq_por_id("bachillerato_titulo_una_materia_condiciones")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if _faq_tiene_alguno(p, ["promocionar", "promocion", "pasar de curso", "pasar a segundo", "pasar  segundo"]) and _faq_tiene_alguno(p, ["dos materias", "2 materias", "dos suspensas", "pendientes"]):
            faq = _buscar_faq_por_id("bachillerato_promocion_dos_materias")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

        habla_evaluacion = _faq_tiene_alguno(p, ["evaluacion", "evaluar", "calificacion", "calificaciones"])
        habla_cyl = _faq_tiene_alguno(p, ["castilla y leon", "castilla leon", "cyl", "castilla"])
        habla_orden_425 = _faq_tiene_alguno(p, ["orden 425 2024", "edu 425 2024", "orden edu 425", "425 2024"])
        compara_estatal_autonomica = _faq_tiene_alguno(p, ["real decreto", "estatal", "rd", "orden", "mayo", "2024"])
        if habla_evaluacion and (habla_cyl or habla_orden_425) and compara_estatal_autonomica:
            faq = _buscar_faq_por_id("cyl_evaluacion_bach_norma")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # 2.b) Permanencia máxima en Bachillerato ordinario.
    if _faq_tiene_alguno(p, ["bachillerato", "bachiller", "bach"]):
        habla_permanencia = _faq_tiene_alguno(p, ["permanecer", "permanencia", "maximo", "maxima", "cuanto tiempo", "cuantos anos", "estar", "dura", "duracion"])
        habla_regimen_ordinario = _faq_tiene_alguno(p, ["ordinario", "regimen ordinario", "anos", "cuatro", "4"])
        if habla_permanencia and habla_regimen_ordinario:
            faq = _buscar_faq_por_id("bachillerato_permanencia_cuatro_anos")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # 3) Permiso por hospitalización/enfermedad grave de familiar de primer grado.
    # Evitamos falsos positivos tipo "mi padre trabaja en un hospital" y
    # distinguimos misma localidad frente a distinta localidad.
    if _faq_tiene_alguno(p, ["padre", "madre", "familiar de primer grado", "primer grado"]):
        hospitalizacion_real = _faq_tiene_alguno(p, ["hospitalizado", "hospitalizada", "hospitalizacion", "ingresado", "ingresada", "ingreso hospitalario", "enfermedad grave"])
        tipo_permiso = _faq_tiene_alguno(p, ["permiso", "baja", "licencia", "vacaciones", "dias", "dia"])
        misma_localidad = _faq_tiene_alguno(p, [
            "misma localidad", "mismo municipio", "misma ciudad",
            "en mi ciudad", "en mi localidad", "en el mismo municipio",
        ])
        # Debe ser una referencia territorial clara. No se aceptan palabras sueltas
        # como "otra", "distinta" o "fuera", porque generan falsos positivos:
        # "otra cosa", "otra planta del hospital", "fuera del quirófano", etc.
        distinta_localidad = _faq_tiene_alguno(p, [
            "distinta localidad", "distinto municipio", "otra localidad",
            "otra ciudad", "otra provincia", "otro municipio",
            "fuera de mi provincia", "fuera de la provincia",
            "fuera de mi localidad", "fuera de mi municipio",
            "fuera de mi ciudad", "en distinta localidad",
            "en otro municipio", "en otra localidad", "en otra ciudad",
            "en otra provincia",
        ])
        if hospitalizacion_real and tipo_permiso and misma_localidad:
            faq = _buscar_faq_por_id("permiso_primer_grado_misma_localidad")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0
        if hospitalizacion_real and tipo_permiso and distinta_localidad:
            faq = _buscar_faq_por_id("permiso_hospitalizacion_padre")
            if _faq_bloque_intencion_ok(faq, bloque_elegido):
                return faq, 1.0

    # 4) Derecho a confidencialidad de datos del alumnado.
    # Debe ganar frente a la FAQ de advertencia de privacidad cuando la pregunta
    # es abstracta/normativa y no introduce un caso identificable.
    if _faq_tiene_todos(p, [
        ["derecho", "tiene derecho", "confidencialidad"],
        ["confidencialidad", "datos personales"],
        ["alumnado", "alumno", "alumnos"],
    ]):
        faq = _buscar_faq_por_id("alumnado_derecho_confidencialidad_datos")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    # 5) Privacidad y datos personales.
    contexto_anonimo = _faq_tiene_alguno(p, [
        "sin nombre", "sin nombres", "sin datos", "anonimo", "anonima",
        "anonimizado", "anonimizada", "anonimizar", "sin identificar",
    ])
    combo_privacidad = _faq_tiene_todos(p, [
        ["nombre", "nombres", "dni", "datos", "expediente", "medico", "medica", "salud", "diagnostico", "tdah"],
        ["alumno", "alumnos", "alumna", "alumnas", "menor", "menores", "familia", "familias", "docente", "profesor", "profesora"],
        ["app", "aplicacion", "meter", "poner", "introducir", "subir", "caso concreto", "que hacer", "que hago", "aqui"],
    ])
    accion_introducir_en_app = _faq_tiene_alguno(p, [
        "app", "aplicacion", "chat", "meter", "poner", "introducir",
        "subir", "pegar", "enviar", "escribir", "aqui",
    ])
    if _faq_contiene_patron_identificativo(pregunta, p) or (combo_privacidad and (not contexto_anonimo or accion_introducir_en_app)):
        faq = _buscar_faq_por_id("privacidad_no_datos_personales_app")
        if _faq_bloque_intencion_ok(faq, bloque_elegido):
            return faq, 1.0

    return None, 0.0


def buscar_faq_verificada(pregunta: str, bloque_elegido: str):
    """Busca una FAQ verificada con criterio conservador.

    Devuelve (faq, score) o (None, 0). No usa IA ni APIs.
    """
    pregunta_n = _normalizar_faq(pregunta)
    pregunta_tokens = _tokens_faq(pregunta)
    if not pregunta_n or not pregunta_tokens:
        return None, 0.0

    # r7 mínima: preguntas de garantías/audiencia en procedimientos de convivencia
    # se dejan pasar a RAG para evitar capturas por FAQ genéricas de sanciones.
    if _faq_pide_garantias_procedimiento_convivencia(pregunta_n):
        return None, 0.0

    faq_exacta = _faq_match_exacta(pregunta_n, bloque_elegido)
    if faq_exacta:
        return faq_exacta, 1.0

    faq_regla, score_regla = _faq_match_reglas_intencion(pregunta, bloque_elegido)
    if faq_regla:
        return faq_regla, score_regla

    mejor = None
    mejor_score = 0.0
    for faq in faq_normativa:
        if not _bloque_faq_compatible(faq.get("bloque", ""), bloque_elegido):
            continue

        required = [x for x in faq.get("required_terms", []) if x]
        required_ok = all(_term_faq_presente(req, pregunta_n) for req in required) if required else True
        if not required_ok:
            continue

        # v072: protección de enrutamiento FAQ/RAG.
        # La FAQ genérica sobre el RD 659/2023 no debe secuestrar preguntas
        # específicas sobre FCT, requisitos o formación en centros de trabajo.
        if faq.get("id") == "fp_norma_estatal_rd659":
            if _faq_pregunta_fp_especifica_debe_ir_a_rag(pregunta_n):
                continue

        # Protección adicional: la FAQ de número de familias profesionales no debe
        # saltar por similitud textual ante preguntas sobre familias del alumnado.
        if faq.get("id") == "fp_numero_familias":
            menciona_profesional = _faq_tiene_alguno(pregunta_n, ["familias profesionales", "familia profesional"])
            if _faq_contiene_contexto_familiar_no_profesional(pregunta_n) and not menciona_profesional:
                continue

        # Evita que “cursos de formación” o “cursos docentes” activen las FAQ
        # que explican cuántos cursos componen ESO/Bachillerato.
        if faq.get("id") in {"eso_cursos", "bachillerato_cursos"}:
            if _faq_tiene_alguno(pregunta_n, ["cursos de formacion", "curso de formacion", "formacion docente", "cursos docentes"]):
                continue

        variantes = [faq.get("pregunta_canonica", "")] + faq.get("variantes", [])
        for variante in variantes:
            var_n = _normalizar_faq(variante)
            if not var_n:
                continue
            var_tokens = _tokens_faq(variante)
            if not var_tokens:
                continue

            ratio = difflib.SequenceMatcher(None, pregunta_n, var_n).ratio()
            coverage = len(pregunta_tokens & var_tokens) / max(1, len(var_tokens))
            jaccard = len(pregunta_tokens & var_tokens) / max(1, len(pregunta_tokens | var_tokens))
            substring_bonus = 0.10 if (len(var_n) >= 14 and (var_n in pregunta_n or pregunta_n in var_n)) else 0.0

            # Evita que variantes muy cortas tipo "evaluación ESO" secuestren
            # preguntas más específicas sobre otra FAQ. La cobertura por tokens
            # solo pesa fuerte si la variante tiene al menos 4 tokens.
            if len(var_tokens) < 4:
                token_score = jaccard * 0.80
            else:
                token_score = coverage * 0.78 + jaccard * 0.22
            score = max(ratio, token_score) + substring_bonus

            if score > mejor_score:
                mejor = faq
                mejor_score = score

    if mejor and mejor_score >= FAQ_MATCH_MIN_RATIO:
        return mejor, min(mejor_score, 1.0)
    return None, mejor_score


def construir_respuesta_faq(faq: dict, score: float):
    fuente = (faq.get("fuentes") or [{}])[0]
    pagina = fuente.get("pagina")
    ref_pagina = f", página {pagina}" if pagina else ""
    documento = fuente.get("documento", "Fuente oficial")
    frag = fuente.get("fragmento_verificado", "")

    texto = (
        "## Respuesta verificada\n"
        f"{faq.get('respuesta', '')}\n\n"
        "## Base normativa encontrada\n"
        "| Fuente | Qué acredita |\n"
        "|---|---|\n"
        f"| {documento}{ref_pagina} | {frag} |\n\n"
        "## Límites de la respuesta\n"
        "Esta respuesta procede de la base local de FAQ verificadas. No resuelve casos individualizados ni sustituye la consulta de la fuente oficial o del órgano competente.\n\n"
        f"_FAQ verificada: `{faq.get('id', '')}` · coincidencia: {score:.2f}_"
    )

    url = fuente.get("url", "")
    if url:
        fuente_screen = f"[FAQ] [{documento}{ref_pagina}]({url})"
        fuente_pdf = f"[FAQ] {documento}{ref_pagina} — {url}"
    else:
        fuente_screen = f"[FAQ] {documento}{ref_pagina}"
        fuente_pdf = fuente_screen
    return texto, [fuente_screen], [fuente_pdf]

faq_normativa = cargar_faq_normativa()

model        = load_model()
# IA usa requests directamente — sin cliente especial
enlaces      = cargar_enlaces()
if not enlaces:
    st.sidebar.warning("⚠️ enlaces.csv no encontrado — las fuentes no tendrán enlace.")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _parse_json(text, default):
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

def validar_input(pregunta):
    if not pregunta or not pregunta.strip():
        return False, "La pregunta no puede estar vacía."
    if len(pregunta) > MAX_CHARS_PREGUNTA:
        return False, f"Pregunta demasiado larga (máximo {MAX_CHARS_PREGUNTA} caracteres)."
    patrones = ["ignore previous", "ignora las instrucciones", "system:", "</s>", "[inst]", "###"]
    if any(p in pregunta.lower() for p in patrones):
        return False, "La pregunta contiene contenido no válido."
    return True, ""


# =============================================================================
# FILTRO DE DOMINIO v067-v069
# =============================================================================
# Objetivo: evitar que preguntas claramente ajenas al ámbito educativo/docente
# entren en Qdrant/RAG y recuperen normativa laboral, contractual o administrativa
# que pueda inducir una respuesta fuera de alcance.
# v069: amplía supuestos de derecho privado, consumo, tráfico, vivienda,
# propiedad industrial, fiscalidad y ámbitos municipales/agrarios.

_DOMINIO_EDUCATIVO_TERMS = [
    "educacion", "educativo", "educativa", "ensenanza", "ensenanzas",
    "alumno", "alumna", "alumnado", "estudiante", "estudiantes", "familia", "familias",
    "docente", "profesor", "profesora", "maestro", "maestra", "claustro", "equipo docente",
    "centro", "colegio", "instituto", "ies", "aula", "clase", "tutoria", "tutor", "tutora",
    "infantil", "primaria", "eso", "secundaria", "bachillerato", "fp", "formacion profesional",
    "ciclo formativo", "grado medio", "grado superior", "grado basico", "familia profesional",
    "curriculo", "curriculo", "evaluacion", "promocion", "titulacion", "calificacion", "reclamacion",
    "convivencia", "conducta", "sancion alumno", "correccion", "mediacion escolar",
    "permiso docente", "licencia docente", "vacaciones docentes", "baja docente",
]

_FUERA_DOMINIO_GRUPOS = [
    # Derecho contractual, mercantil o consumo privado.
    ["empresa privada", "contrato"],
    ["contrato privado"],
    ["incumple un contrato"],
    ["incumplimiento de contrato"],
    ["contrato mercantil"],
    ["consumidor", "garantia"],
    ["factura", "iva"],
    # Ámbitos claramente no educativos.
    ["divorcio"],
    ["custodia compartida"],
    ["alquiler", "vivienda"],
    ["comunidad de propietarios"],
    ["multa de trafico"],
    ["sancion de trafico"],
    ["delito"],
    ["denuncia penal"],
    ["herencia"],
    ["testamento"],
    ["hipoteca"],
    ["desahucio"],
    ["autonomo", "cuota"],
    # r7 mínima: fiscalidad privada/autónomos fuera de dominio educativo.
    ["irpf"],
    ["autonomo", "pagar"],
    ["autonomo", "impuesto"],
    ["autonomo", "impuestos"],
    ["autonomo", "irpf"],
    ["declaracion", "renta"],
    ["iva"],

    # v069: casos fuera de dominio detectados por auditoría sintética.
    ["aerolinea"],
    ["maleta"],
    ["impuestos"],
    ["sociedad limitada"],
    ["demanda civil"],
    ["vecino"],
    ["accidente de trafico"],
    ["indemnizacion", "trafico"],
    ["casero"],
    ["caldera"],
    ["tienda online"],
    ["devolucion", "tienda"],
    ["marca comercial"],
    ["registrar marca"],
    ["registro marca"],
    ["conducir sin seguro"],
    ["despedir", "trabajador"],
    ["despido", "trabajador"],
    ["licencia de obras"],
    ["ayuntamiento", "obras"],
    ["subvencion agricola"],
    ["subvencion", "agricola"],
]


def _normalizar_dominio(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _contiene_frase_dominio(p: str, frase: str) -> bool:
    frase_n = _normalizar_dominio(frase)
    if not frase_n:
        return False
    if len(frase_n.split()) == 1:
        return frase_n in set(p.split())
    return frase_n in p


def _pregunta_tiene_indicios_educativos(pregunta: str, bloque_elegido: str = "") -> bool:
    p = _normalizar_dominio(pregunta)
    if any(_contiene_frase_dominio(p, t) for t in _DOMINIO_EDUCATIVO_TERMS):
        return True
    # Si el usuario está en FP, protegemos términos propios de FP que pueden
    # confundirse con empresa/contrato en lenguaje general.
    if bloque_elegido == "fp" and any(_contiene_frase_dominio(p, t) for t in [
        "empresa", "formacion", "practicas", "dual", "centro", "tutor", "ciclo", "grado"
    ]):
        return True
    return False


def _pregunta_fuera_de_dominio(pregunta: str, bloque_elegido: str = "") -> bool:
    """Filtro conservador: solo bloquea preguntas claramente fuera de dominio.

    No pretende resolver todos los casos. Su función es cortar antes de RAG
    cuando la pregunta apunta a derecho privado/mercantil/general y no contiene
    indicios educativos suficientes.
    """
    p = _normalizar_dominio(pregunta)
    if not p:
        return False

    if _pregunta_tiene_indicios_educativos(pregunta, bloque_elegido):
        return False

    for grupo in _FUERA_DOMINIO_GRUPOS:
        if all(_contiene_frase_dominio(p, termino) for termino in grupo):
            return True

    # Regla específica para el caso detectado por auditoría RAG.
    if _contiene_frase_dominio(p, "empresa") and _contiene_frase_dominio(p, "contrato"):
        if any(_contiene_frase_dominio(p, t) for t in ["sancion", "incumple", "incumplimiento", "privada"]):
            return True

    return False


def construir_respuesta_fuera_dominio(pregunta: str, bloque_elegido: str = "") -> str:
    return (
        "## Consulta fuera del ámbito de la app\n\n"
        "No puedo responder con seguridad a esta pregunta porque parece referirse a derecho privado, "
        "contractual, mercantil, laboral general u otro ámbito ajeno a la normativa educativa que cubre esta app.\n\n"
        "Esta herramienta está pensada para consultas sobre normativa educativa, centros docentes, alumnado, "
        "convivencia escolar, evaluación, Formación Profesional y cuestiones administrativas docentes.\n\n"
        "Si tu pregunta sí está relacionada con un centro educativo, un alumno, un docente o una enseñanza concreta, "
        "reformúlala indicando ese contexto educativo, sin incluir datos personales.\n\n"
        "_Filtro de dominio aplicado: no se ha consultado Qdrant ni se ha consumido IA._"
    )


# ============================================================
# r4 - Filtro defensivo mínimo antes de RAG/IA
# ============================================================
# Objetivo: cortar peticiones de prompt injection, claves, secretos o
# instrucciones internas antes de consultar Qdrant o consumir IA.
# Deliberadamente estrecho para no bloquear consultas normativas legítimas.

_PATRONES_SEGURIDAD_R4 = [
    ("ignora", "instrucciones"),
    ("olvida", "instrucciones"),
    ("desobedece", "instrucciones"),
    ("omite", "instrucciones"),
    ("muestra", "prompt"),
    ("revela", "prompt"),
    ("dime", "prompt"),
    ("prompt", "sistema"),
    ("system", "prompt"),
    ("claves", "internas"),
    ("secretos", "internos"),
    ("dime", "claves"),
    ("revela", "claves"),
    ("muestra", "claves"),
    ("dime", "secretos"),
    ("revela", "secretos"),
    ("muestra", "secretos"),
    ("api", "key"),
    ("api", "clave"),
    ("ia_api_key",),
    ("qdrant_api_key",),
    ("st", "secrets"),
    ("streamlit", "secrets"),
    ("token", "secreto"),
    ("variables", "entorno"),
]

def _pregunta_seguridad_defensiva(pregunta: str) -> bool:
    p = _normalizar_dominio(pregunta)
    if not p:
        return False
    tokens = set(p.split())
    for patron in _PATRONES_SEGURIDAD_R4:
        if all((termino in tokens) or _contiene_frase_dominio(p, termino) for termino in patron):
            return True
    return False

def construir_respuesta_seguridad_defensiva(pregunta: str, bloque_elegido: str = "") -> str:
    return (
        "## Consulta bloqueada por seguridad\n\n"
        "No puedo ayudar a revelar claves, tokens, secretos, prompts, instrucciones internas "
        "ni configuración privada del sistema.\n\n"
        "Esta app está pensada para responder consultas sobre normativa educativa, centros docentes, "
        "alumnado, convivencia escolar, evaluación, Formación Profesional y cuestiones administrativas docentes.\n\n"
        "Si necesitas una respuesta normativa, reformula la pregunta sin pedir datos internos, claves, "
        "prompts ni instrucciones del sistema.\n\n"
        "_Filtro de seguridad aplicado: no se ha consultado Qdrant ni se ha consumido IA._"
    )


# ============================================================
# v073 - Precisión Qdrant sin reindexar
# ============================================================
# No modifica la colección, ni embeddings, ni Qdrant.
# Mejora:
# - expansión local de consulta para intenciones detectadas;
# - bonus/penalización local de candidatos antes de cortar a Top8.

def _normalizar_ranking_v073(texto):
    import re
    import unicodedata
    texto = unicodedata.normalize("NFKD", str(texto or ""))
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9ñ]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _ranking_contiene_v073(texto_n, opciones):
    return any(_normalizar_ranking_v073(op) in texto_n for op in opciones)


def _intencion_ranking_v073(pregunta):
    p = _normalizar_ranking_v073(pregunta)

    if _ranking_contiene_v073(p, [
        "evaluacion objetiva",
        "prueba objetiva",
        "tipo test",
        "reclamar una calificacion",
        "reclamacion calificacion",
        "calificacion en secundaria",
    ]):
        return "evaluacion_objetiva"

    if _ranking_contiene_v073(p, [
        "procedimiento corrector",
        "garantias",
        "conducta que perturba la convivencia",
        "conducta perturbadora",
        "convivencia",
    ]) and _ranking_contiene_v073(p, [
        "procedimiento",
        "corrector",
        "garantias",
        "audiencia",
        "alegaciones",
        "medidas",
        "conducta",
    ]):
        return "convivencia_procedimiento"

    return ""


def _reformulaciones_precision_v073(pregunta):
    """Reformulaciones locales, sin IA, para mejorar recuperación vectorial.

    La app seguirá usando como máximo dos reformulaciones. Ponemos primero la
    reformulación especializada para que entre en el promedio de embeddings.
    """
    p = _normalizar_ranking_v073(pregunta)
    intent = _intencion_ranking_v073(pregunta)
    refs = []

    if intent == "evaluacion_objetiva":
        refs.append(
            "derecho del alumnado a una evaluación objetiva reclamación de calificaciones "
            "criterios de evaluación secundaria bachillerato"
        )

    if intent == "convivencia_procedimiento":
        refs.append(
            "procedimiento corrector convivencia escolar audiencia alegaciones garantías "
            "decreto 51 2007 derechos deberes alumnado"
        )

    base = re.sub(r"[¿?¡!]", "", pregunta or "").strip()
    if base and base not in refs:
        refs.append(base)

    return refs


def expandir_y_corregir(pregunta):
    """Sin LLM para ahorrar tokens de IA.
    La búsqueda semántica + reordenación local mantiene el coste en cero.
    """
    corregida = pregunta.strip()
    return corregida, _reformulaciones_precision_v073(corregida)


# Stopwords españolas para extracción de términos clave
_STOPWORDS = {
    "qué","cuál","cuáles","cómo","cuándo","cuánto","cuántos","cuántas",
    "dónde","quién","quiénes","por","para","con","sin","sobre","entre",
    "desde","hasta","hacia","ante","bajo","según","durante","mediante",
    "un","una","unos","unas","el","la","los","las","del","al",
    "es","son","está","están","ser","tener","tiene","tienen","haber",
    "hay","puede","pueden","debe","deben","se","me","te","le","nos",
    "de","en","a","y","o","e","u","que","si","no","más","pero",
    "yo","tú","él","ella","usted","nosotros","ellos","su","sus",
    "mi","mis","tu","tus","un","una","lo","le","les",
    # No incluimos términos jurídicos/educativos como "docente", "alumno"
    # o "derechos": pueden ser esenciales para recuperar normativa correcta.
    "tiene","tendrá","podrá","podrán",
    "favor","hacer","realizar","solicitar","pedir",
}

def extraer_terminos_clave(pregunta: str) -> list[str]:
    """Extrae 4-6 términos clave de la pregunta eliminando stopwords.
    Devuelve también bigramas de términos legales relevantes.
    """
    import re
    # Normalizar
    texto = pregunta.lower().strip("¿?.,;:")
    texto = re.sub(r"[¿?.,;:()\.\[\]{}!]", " ", texto)
    palabras = [p for p in texto.split() if len(p) > 2 and p not in _STOPWORDS]

    terminos = []
    # Añadir palabras individuales relevantes
    for p in palabras:
        if p not in terminos:
            terminos.append(p)

    # Añadir bigramas de palabras consecutivas
    for i in range(len(palabras) - 1):
        bigrama = f"{palabras[i]} {palabras[i+1]}"
        if bigrama not in terminos:
            terminos.append(bigrama)

    return terminos[:8]  # máx 8 términos/bigramas

def _qdrant_search_rest(embedding, bloque=None, threshold=None, limit=MATCH_COUNT_RETRIEVAL):
    """Búsqueda semántica via REST API directa.

    En modo coste cero y con los bloques actuales pendientes de depuración,
    NO usamos filtro duro por nivel educativo. Recuperamos candidatos de toda
    la colección y después los reordenamos localmente, dando bonus al bloque
    elegido cuando coincide.
    """
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    embedding = [float(x) for x in embedding]
    payload = {
        "vector": embedding,
        "limit": limit,
        "with_payload": True,
    }
    # Importante: no aplicar filtro por bloque aquí. El bloque actual se usa
    # como preferencia en el reranking local, no como exclusión.
    if threshold is not None:
        payload["score_threshold"] = threshold
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Qdrant respondió HTTP {r.status_code}")
        return r.json().get("result", [])
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("No se pudo conectar con Qdrant. Revisa QDRANT_URL, QDRANT_API_KEY o la red.") from exc

def _qdrant_text_search_rest(pregunta_texto, bloque=None, terminos=None):
    """Búsqueda textual desactivada como consulta remota.

    La colección actual no tiene indexado el campo `contenido` para búsqueda
    textual. Para mantener coste cero y evitar consultas lentas o frágiles,
    hacemos la parte léxica en Python sobre los candidatos vectoriales.
    """
    return []

def _normalizar_score(score):
    try:
        return float(score or 0.0)
    except Exception:
        return 0.0


def _score_lexico(contenido: str, terminos: list[str]) -> float:
    """Puntuación léxica local reforzada en v073.

    Sigue siendo coste cero y acotada, pero normaliza acentos y da más peso a
    frases exactas. Esto ayuda a que términos jurídicos fuertes compitan mejor
    frente a similitud vectorial genérica.
    """
    if not contenido or not terminos:
        return 0.0

    texto = _normalizar_ranking_v073(contenido)
    score = 0.0

    for termino in terminos:
        t = _normalizar_ranking_v073(termino)
        if not t:
            continue
        if " " in t and t in texto:
            score += 0.10
        elif t in texto:
            score += 0.04

    return min(score, 0.35)


def _bonus_bloque(payload: dict, bloque: str) -> float:
    """Usa el nivel educativo como preferencia, no como filtro excluyente."""
    if not bloque or bloque == "general":
        return 0.0
    return 0.06 if payload.get("bloque") == bloque else 0.0


def _ajuste_precision_qdrant_v073(pregunta: str, payload: dict, contenido: str) -> tuple[float, list[str]]:
    """Ajustes locales de precisión antes del corte Top8.

    No filtra de forma dura: solo suma o resta puntuación acotada.
    """
    p = _normalizar_ranking_v073(pregunta)
    texto = _normalizar_ranking_v073(
        f"{payload.get('nombre_archivo', '')} {payload.get('bloque', '')} {contenido or ''}"
    )
    doc = _normalizar_ranking_v073(payload.get("nombre_archivo", ""))
    intent = _intencion_ranking_v073(pregunta)

    score = 0.0
    motivos = []

    if intent == "evaluacion_objetiva":
        if _ranking_contiene_v073(texto, [
            "evaluacion objetiva",
            "objetividad",
            "derecho a una evaluacion objetiva",
            "reclamacion de calificaciones",
            "reclamacion contra la calificacion",
            "calificacion final",
            "criterios de calificacion",
            "instrumentos de evaluacion",
        ]):
            score += 0.30
            motivos.append("bonus_evaluacion_objetiva")

        # Evita que fragmentos de títulos FP con "criterios de evaluación"
        # dominen consultas jurídicas sobre evaluación objetiva.
        ruido_titulo_fp = _ranking_contiene_v073(texto, [
            "modulo profesional",
            "unidades de competencia",
            "catalogo nacional de cualificaciones",
            "titulo profesional basico",
            "establece titulo",
        ])
        solo_criterios_genericos = (
            "criterios de evaluacion" in texto
            and not _ranking_contiene_v073(texto, [
                "evaluacion objetiva",
                "reclamacion",
                "calificacion",
                "objetividad",
            ])
        )
        if ruido_titulo_fp or solo_criterios_genericos:
            score -= 0.22
            motivos.append("penaliza_evaluacion_generica_titulo_fp")

    if intent == "convivencia_procedimiento":
        if _ranking_contiene_v073(doc, [
            "decreto 51 2007",
            "decreto_51_2007",
            "decreto 23 2014",
            "decreto_23_2014",
        ]):
            score += 0.35
            motivos.append("bonus_decreto_convivencia")

        if _ranking_contiene_v073(texto, [
            "procedimiento corrector",
            "medidas correctoras",
            "conductas contrarias a las normas de convivencia",
            "conductas gravemente perjudiciales",
            "audiencia",
            "alegaciones",
            "instructor",
            "expediente",
            "resolucion",
            "acuerdo reeducativo",
            "convivencia escolar",
        ]):
            score += 0.25
            motivos.append("bonus_vocabulario_procedimiento_convivencia")

        ruido_fp = _ranking_contiene_v073(texto, [
            "unidades de competencia",
            "catalogo nacional de cualificaciones",
            "modulo profesional",
            "establece titulo",
            "realiza actividades de evaluacion",
            "mediadores naturales",
            "unidades de convivencia",
        ]) and not _ranking_contiene_v073(doc, [
            "decreto 51 2007",
            "decreto_51_2007",
            "decreto 23 2014",
            "decreto_23_2014",
        ])
        if ruido_fp:
            score -= 0.30
            motivos.append("penaliza_ruido_fp_o_social_no_convivencia_escolar")

    # Penalización suave y general: si la consulta no es FP, los documentos
    # de títulos/cualificaciones FP no deben dominar salvo que contengan
    # términos fuertes de la pregunta.
    pregunta_fp = _ranking_contiene_v073(p, ["fp", "formacion profesional", "ciclo formativo", "fct"])
    ruido_titulo = _ranking_contiene_v073(texto, [
        "modulo profesional",
        "unidades de competencia",
        "catalogo nacional de cualificaciones",
        "establece titulo",
    ])
    if not pregunta_fp and ruido_titulo and intent in ["evaluacion_objetiva", "convivencia_procedimiento"]:
        score -= 0.08
        motivos.append("penaliza_titulo_fp_fuera_fp")

    score = max(-0.45, min(0.55, score))
    return score, motivos


def buscar_normativa_hibrida(embedding, pregunta_texto, bloque):
    """Búsqueda semántica + reordenación local de coste cero.

    Cambio clave: el nivel educativo ya no excluye documentos. Primero se
    recuperan candidatos de toda la colección y después se da prioridad a los
    fragmentos cuyo `bloque` coincide. Esto evita perder normativa relevante
    mientras se corrigen los metadatos de Qdrant.
    """
    terminos_clave = extraer_terminos_clave(pregunta_texto)

    resultados_v = []
    for threshold in [MATCH_THRESHOLD_ALTO, MATCH_THRESHOLD_BAJO, None]:
        hits = _qdrant_search_rest(
            embedding,
            bloque=None,
            threshold=threshold,
            limit=MATCH_COUNT_RETRIEVAL,
        )
        resultados_v = hits
        if len(resultados_v) >= MATCH_COUNT:
            break

    vistos_contenido = set()
    ids_vistos = set()
    combinados = []

    for hit in resultados_v:
        rid = str(hit.get("id", ""))
        payload = hit.get("payload", {}) or {}
        contenido = payload.get("contenido", "")
        clave = contenido[:160].strip()
        if rid in ids_vistos or (clave and clave in vistos_contenido):
            continue
        ids_vistos.add(rid)
        if clave:
            vistos_contenido.add(clave)

        base = _normalizar_score(hit.get("score", 0.0))
        lexical = _score_lexico(contenido, terminos_clave)
        bloque_bonus = _bonus_bloque(payload, bloque)
        precision_bonus, precision_motivos = _ajuste_precision_qdrant_v073(
            pregunta_texto, payload, contenido
        )
        score_final = base + lexical + bloque_bonus + precision_bonus

        combinados.append({
            "id": rid,
            "contenido": contenido,
            "nombre_archivo": payload.get("nombre_archivo", ""),
            "pagina_num": payload.get("pagina_num", 0),
            "bloque": payload.get("bloque", ""),
            "similarity": score_final,
            "score_vectorial": base,
            "score_lexico": lexical,
            "bonus_bloque": bloque_bonus,
            "bonus_precision_v073": precision_bonus,
            "motivos_precision_v073": "; ".join(precision_motivos),
        })

    combinados = sorted(combinados, key=lambda x: x.get("similarity", 0), reverse=True)
    return combinados[:MATCH_COUNT]


def reranquear(pregunta, fragmentos):
    """Reordenación final local. No usa APIs externas ni genera coste."""
    return sorted(fragmentos, key=lambda x: x.get("similarity", 0), reverse=True)


def recortar_contexto_xml(contexto_xml: str, max_chars: int = MAX_CHARS_CONTEXTO) -> str:
    """Límite duro de contexto para proteger el free tier de IA."""
    if len(contexto_xml) <= max_chars:
        return contexto_xml
    return contexto_xml[:max_chars] + "\n\n[CONTEXTO RECORTADO POR LÍMITE GRATUITO DE LA APP]"

def construir_contexto_xml(fragmentos, enlaces_dict):
    """Construye contexto con identificadores [F1], [F2]...

    Esos identificadores son la única forma válida de cita que puede usar el LLM.
    Así podemos validar después que no cite fragmentos inexistentes.
    """
    contexto_xml = ""
    links_screen = []
    fuentes_pdf  = []
    for i, res in enumerate(fragmentos, 1):
        fid      = f"F{i}"
        nombre   = res.get("nombre_archivo", "")
        pagina   = res.get("pagina_num", "")
        score    = res.get("similarity", "")
        bloque   = res.get("bloque", "")
        nombre_l = nombre.replace(".pdf", "").replace("_", " ")
        score_s  = f"{score:.2f}" if isinstance(score, float) else ""
        contexto_xml += (
            f'<fragmento id="{fid}" cita_obligatoria="[{fid}]" documento="{nombre_l}" '
            f'pagina="{pagina}" bloque="{bloque}" relevancia="{score_s}">\n'
            f'{res.get("contenido", "")}\n</fragmento>\n\n'
        )
        url = enlaces_dict.get(nombre)
        if url:
            # #page=N abre el PDF directamente en la página indicada en la mayoría de navegadores
            link = f"{url}#page={pagina}"
            links_screen.append(f"[{fid}] [{nombre_l} — pág. {pagina}]({link})")
            fuentes_pdf.append(f"[{fid}] {nombre_l} (Pág. {pagina}) — {url}")
        else:
            links_screen.append(f"[{fid}] **{nombre_l}** — pág. {pagina} *(enlace no disponible)*")
            fuentes_pdf.append(f"[{fid}] {nombre_l} (Pág. {pagina})")
    return contexto_xml, links_screen, fuentes_pdf


def ids_fragmentos_validos_contexto(contexto_xml: str):
    """Extrae los identificadores F# realmente presentes en el contexto enviado al LLM.

    Se usa después de recortar el contexto para que la validación de citas no
    acepte una cita [F#] solo porque existía en la lista original de resultados,
    sino porque el identificador seguía presente en el contexto final.
    """
    import re
    ids = {int(x) for x in re.findall(r'<fragmento\s+id="F(\d+)"', contexto_xml or "")}
    return sorted(ids)


def validar_citas_fragmentos(respuesta: str, num_fragmentos: int = None, ids_validos=None):
    """Valida que las citas [F1], [F2]... existan en el contexto enviado.

    Devuelve: (es_valida, citas_detectadas, citas_invalidas).
    No comprueba si la afirmación está bien sustentada; solo evita citas inexistentes.

    `ids_validos` permite validar contra los fragmentos realmente presentes tras
    recortar el contexto. Si no se facilita, se mantiene compatibilidad con la
    validación clásica por número de fragmentos.
    """
    import re
    citas = [int(x) for x in re.findall(r"\[F(\d+)\]", respuesta or "")]

    if ids_validos is not None:
        permitidas = {int(x) for x in ids_validos}
    else:
        n = int(num_fragmentos or 0)
        permitidas = set(range(1, n + 1))

    invalidas = sorted({c for c in citas if c not in permitidas})
    return len(invalidas) == 0, sorted(set(citas)), invalidas


def respuesta_segura_por_citas_invalidas(citas_invalidas):
    inv = ", ".join(f"[F{x}]" for x in citas_invalidas)
    return (
        "## Respuesta\n"
        "No puedo mostrar una respuesta suficientemente fiable porque la respuesta generada "
        f"citó fragmentos que no existen en el contexto recuperado: {inv}.\n\n"
        "## Qué puedes hacer\n"
        "- Reformula la pregunta con más detalle.\n"
        "- Selecciona el nivel educativo más adecuado.\n"
        "- Consulta las fuentes oficiales mostradas o vuelve a intentarlo.\n\n"
        "Esta protección evita mostrar una respuesta jurídica con citas no verificables."
    )

# ============================================================
# v070 - Respuestas prudentes contextuales
# ============================================================
# Postprocesado determinista y conservador:
# - No cambia Qdrant.
# - No cambia FAQ.
# - No cambia prompt.
# - No inventa contenido normativo.
# - Solo añade orientación práctica cuando la propia respuesta ya reconoce
#   que no hay información suficiente y la pregunta depende de datos actuales,
#   locales, de centro o individualizados.

def _normalizar_prudencia_contextual(texto):
    texto = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _pregunta_contextual_actual_local(pregunta):
    p = _normalizar_prudencia_contextual(pregunta)

    patrones = [
        # Tiempo / actualidad
        "este curso", "curso actual", "ahora", "actualmente", "este ano", "este año",
        "2024 2025", "2025 2026", "plazo actual", "convocatoria actual",

        # Territorio / oferta concreta
        "valladolid", "burgos", "leon", "palencia", "salamanca", "segovia", "soria",
        "zamora", "avila", "avila", "mi provincia", "provincia", "localidad",
        "mi zona", "mi municipio",

        # Centro / individualización
        "mi centro", "mi colegio", "mi instituto", "mi ciclo", "su colegio",
        "su instituto", "en su centro", "en mi centro", "en mi instituto",
        "secretaria", "jefatura de estudios",

        # Datos cambiantes o de gestión
        "plazas libres", "plaza libre", "horario exacto", "horario", "libros concretos",
        "lista de libros", "optativas oferta", "oferta mi instituto", "que optativas",
        "profesor imparte", "que profesor", "reuniones de evaluacion", "comedor escolar",
        "transporte escolar", "empresas concretas", "convenio este curso",
        "catalogo", "oferta formativa", "ciclos hay", "que ciclos", "ofertan", "que centros", "asir", "dam", "daw", "smr",
    ]

    return any(pat in p for pat in patrones)


def _respuesta_insuficiente_sin_orientacion(respuesta):
    r = _normalizar_prudencia_contextual(respuesta)

    insuficiencia = [
        "con los fragmentos recuperados no hay informacion suficiente",
        "no hay informacion suficiente para responder con seguridad",
        "no puedo responder con seguridad",
        "no puedo determinar con seguridad",
        "no puedo confirmar",
        "no consta en los fragmentos",
        "los fragmentos no contienen",
        "no se dispone de informacion",
    ]

    orientacion = [
        "orientacion practica",
        "consulta",
        "consultar",
        "fuente oficial",
        "consejeria",
        "junta de castilla y leon",
        "catalogo",
        "oferta formativa",
        "centro educativo",
        "secretaria",
        "direccion provincial",
        "organo competente",
        "portal de educacion",
        "programacion didactica",
    ]

    tiene_insuficiencia = any(x in r for x in insuficiencia)
    tiene_orientacion = any(x in r for x in orientacion)

    return tiene_insuficiencia and not tiene_orientacion


def _orientacion_practica_contextual(pregunta, bloque_elegido=None):
    p = _normalizar_prudencia_contextual(pregunta)

    base = (
        "## Orientación práctica\n"
        "Esta consulta depende de información actual, local, del centro o de una situación individual, "
        "por lo que puede cambiar según el curso académico, la provincia, el centro educativo o la convocatoria vigente.\n\n"
    )

    # v070b: reglas específicas antes que reglas generales.
    if any(x in p for x in ["transporte escolar", "transporte", "ruta escolar"]):
        return base + (
            "Para confirmarlo, consulta la **convocatoria anual o información vigente de transporte escolar**, "
            "la **secretaría del centro**, la **Dirección Provincial de Educación** correspondiente "
            "o el **portal oficial de Educación de Castilla y León**."
        )

    if any(x in p for x in ["comedor escolar", "comedor", "plazas comedor"]):
        return base + (
            "Para confirmarlo, consulta la **información anual del servicio de comedor escolar**, "
            "la **secretaría del centro**, la **Dirección Provincial de Educación** correspondiente "
            "o el **portal oficial de Educación de Castilla y León**."
        )

    if any(x in p for x in ["horario exacto", "horario", "reuniones de evaluacion", "reunion de evaluacion", "calendario del centro"]):
        return base + (
            "Para confirmarlo, consulta el **horario oficial del centro para el curso actual**, "
            "la **jefatura de estudios**, la **secretaría del centro educativo** o la documentación organizativa interna del centro."
        )

    if any(x in p for x in ["optativas", "optativa", "oferta mi instituto", "que optativas", "materias optativas"]):
        return base + (
            "Para confirmarlo, consulta la **programación u organización del centro**, "
            "la **oferta de materias publicada para el curso actual**, la **jefatura de estudios** "
            "o la **secretaría del instituto**."
        )

    if any(x in p for x in ["profesor imparte", "que profesor", "profesorado", "asignacion docente", "imparte el modulo"]):
        return base + (
            "Para confirmarlo, consulta la **organización docente del centro**, la **jefatura de estudios**, "
            "la **secretaría del centro educativo** o el equipo directivo."
        )

    if any(x in p for x in ["libros", "lista de libros", "releo"]):
        return base + (
            "Para saber los libros concretos, consulta la **lista de libros publicada por el centro**, "
            "la **secretaría del colegio** o la información del programa **RELEO PLUS**, si el centro participa en él."
        )

    if any(x in p for x in ["plazas libres", "plaza libre", "admision", "vacantes"]):
        return base + (
            "Para comprobar plazas o vacantes, consulta el **proceso de admisión vigente**, "
            "el **portal oficial de Educación de Castilla y León**, la **Dirección Provincial de Educación** "
            "o la **secretaría del centro**."
        )

    if any(x in p for x in ["empresas concretas", "convenio este curso", "convenios", "practicas de fp", "formacion en empresa"]):
        return base + (
            "Para confirmarlo, consulta la **secretaría del centro**, la persona responsable de la **tutoría de FP dual o formación en empresa**, "
            "la **Dirección Provincial de Educación** o la documentación actualizada de convenios del centro."
        )

    if any(x in p for x in [
        "centros que ofertan", "que centros", "ofertan", "oferta fp", "oferta formativa fp",
        "asir", "dam", "daw", "smr", "ciclo", "ciclos", "fp", "formacion profesional",
        "oferta formativa", "desarrollo de aplicaciones", "catalogo"
    ]):
        return base + (
            "Para comprobarlo con seguridad, consulta el **portal oficial de Formación Profesional de la Junta de Castilla y León**, "
            "el **catálogo anual de oferta formativa**, el **buscador oficial de centros y enseñanzas**, "
            "la **Dirección Provincial de Educación** correspondiente o la **secretaría del centro**."
        )

    return base + (
        "Para confirmarlo, consulta la **fuente oficial actualizada**, la **secretaría del centro educativo**, "
        "la **Dirección Provincial de Educación** o el órgano competente que gestione esa información."
    )


def reforzar_respuesta_prudente_contextual(respuesta, pregunta, bloque_elegido=None):
    """Añade orientación práctica si la respuesta es insuficiente y la pregunta es contextual.

    No modifica respuestas que ya contienen orientación, respuestas FAQ, respuestas fuera de dominio
    ni respuestas con contenido suficiente.
    """
    if not respuesta:
        return respuesta

    if not _pregunta_contextual_actual_local(pregunta):
        return respuesta

    if not _respuesta_insuficiente_sin_orientacion(respuesta):
        return respuesta

    return respuesta.rstrip() + "\n\n" + _orientacion_practica_contextual(pregunta, bloque_elegido)


# ============================================================
# v071 - Trazabilidad compacta del historial
# ============================================================
# Objetivo:
# - Mejorar la auditabilidad del historial/exportación.
# - No cambia la lógica de respuesta.
# - No toca FAQ, Qdrant, IA, filtro ni postprocesado prudente contextual.

def _tipo_orientacion_contextual(pregunta):
    p = _normalizar_prudencia_contextual(pregunta)

    if any(x in p for x in ["transporte escolar", "transporte", "ruta escolar"]):
        return "transporte_escolar"
    if any(x in p for x in ["comedor escolar", "comedor", "plazas comedor"]):
        return "comedor_escolar"
    if any(x in p for x in ["horario exacto", "horario", "reuniones de evaluacion", "reunion de evaluacion", "calendario del centro"]):
        return "horario_centro"
    if any(x in p for x in ["optativas", "optativa", "oferta mi instituto", "que optativas", "materias optativas"]):
        return "optativas_centro"
    if any(x in p for x in ["profesor imparte", "que profesor", "profesorado", "asignacion docente", "imparte el modulo"]):
        return "profesorado"
    if any(x in p for x in ["libros", "lista de libros", "releo"]):
        return "libros_centro"
    if any(x in p for x in ["plazas libres", "plaza libre", "admision", "vacantes"]):
        return "plazas_vacantes"
    if any(x in p for x in ["empresas concretas", "convenio este curso", "convenios", "practicas de fp", "formacion en empresa"]):
        return "convenios_practicas"

    # v071c: afinar preguntas de oferta FP por familia/provincia/curso.
    # Ejemplo detectado en auditoría sintética:
    # "informática y comunicaciones este curso en mi provincia"
    if any(x in p for x in [
        "ciclo", "ciclos", "fp", "formacion profesional", "oferta formativa",
        "desarrollo de aplicaciones", "catalogo", "familia profesional",
        "familia de informatica", "informatica y comunicaciones",
        "este curso en mi provincia", "ciclos en mi provincia",
        "ciclos de informatica", "grado medio en mi provincia",
        "grado superior en mi provincia", "ofertan", "que centros",
        "centros que ofertan", "asir", "dam", "daw", "smr"
    ]):
        return "oferta_fp"

    return "generica"


def respuesta_contiene_orientacion_contextual(respuesta, pregunta=None):
    """Detecta si la respuesta final ya contiene orientación práctica contextual.

    v071b: puede haber orientación aunque no la haya añadido el postprocesado v070b,
    porque la propia IA puede generar un bloque de orientación práctica.
    """
    r = _normalizar_prudencia_contextual(respuesta or "")
    p = _normalizar_prudencia_contextual(pregunta or "")

    if not r:
        return False

    marcas_orientacion = [
        "orientacion practica",
        "para conocer",
        "para saber",
        "para confirmarlo",
        "debes consultar",
        "debe consultar",
        "consulta directamente",
        "consulta la convocatoria",
        "ponte en contacto",
    ]

    fuentes_contextuales = [
        "centro educativo",
        "secretaria del centro",
        "secretaria del colegio",
        "jefatura de estudios",
        "jefe de estudios",
        "direccion provincial",
        "consejeria de educacion",
        "junta de castilla y leon",
        "portal oficial de educacion",
        "catalogo anual",
        "oferta formativa",
        "convocatoria vigente",
        "horario oficial",
        "programacion didactica",
        "tutor del ciclo",
        "tutoria de fp",
        "transporte escolar",
        "comedor escolar",
        "lista de libros",
        "releo plus",
    ]

    tiene_marca = any(x in r for x in marcas_orientacion)
    tiene_fuente = any(x in r for x in fuentes_contextuales)

    # Evita marcar orientaciones genéricas de filtros fuera de dominio si la pregunta no es contextual.
    pregunta_contextual = bool(p) and _pregunta_contextual_actual_local(pregunta or "")

    return tiene_marca and tiene_fuente and pregunta_contextual


def construir_trazabilidad_historial(
    ruta,
    apartado,
    faq_id=None,
    faq_score=None,
    filtro_dominio=False,
    qdrant=False,
    ia=False,
    orientacion_prudente=False,
    tipo_orientacion="",
    num_fragmentos=None,
    ia_reintentos=None,
):
    """Construye una traza compacta para historial/exportación.

    No contiene pregunta, respuesta, claves ni contenido de fragmentos.
    """
    return {
        "version_app": "v073b_postvalidacion_r7_filtros_faq_prudencia",
        "ruta": ruta,
        "apartado": apartado,
        "faq_id": faq_id or "",
        "faq_score": "" if faq_score is None else faq_score,
        "filtro_dominio_aplicado": bool(filtro_dominio),
        "qdrant_consultado": bool(qdrant),
        "ia_consumida": bool(ia),
        "orientacion_prudente_contextual": bool(orientacion_prudente),
        "tipo_orientacion": tipo_orientacion or "",
        "num_fragmentos": "" if num_fragmentos is None else num_fragmentos,
        "ia_reintentos": "" if ia_reintentos is None else ia_reintentos,
    }


def _si_no(valor):
    return "sí" if bool(valor) else "no"


def formatear_trazabilidad_compacta(traza):
    if not traza:
        return ""
    partes = [
        f"Ruta: {traza.get('ruta', '-')}",
        f"Qdrant: {_si_no(traza.get('qdrant_consultado'))}",
        f"IA: {_si_no(traza.get('ia_consumida'))}",
    ]
    if traza.get("faq_id"):
        partes.append(f"FAQ: {traza.get('faq_id')}")
    if traza.get("orientacion_prudente_contextual"):
        partes.append(f"Orientación: {traza.get('tipo_orientacion') or 'sí'}")
    return "Trazabilidad: " + " · ".join(partes)


def formatear_trazabilidad_bloque(traza):
    if not traza:
        return ""
    lineas = [
        "TRAZABILIDAD",
        f"- Versión: {traza.get('version_app', '-')}",
        f"- Ruta: {traza.get('ruta', '-')}",
        f"- Apartado: {traza.get('apartado', '-')}",
        f"- FAQ activada: {traza.get('faq_id') or 'no'}",
        f"- Coincidencia FAQ: {traza.get('faq_score') if traza.get('faq_score') != '' else '-'}",
        f"- Filtro de dominio aplicado: {_si_no(traza.get('filtro_dominio_aplicado'))}",
        f"- Qdrant consultado: {_si_no(traza.get('qdrant_consultado'))}",
        f"- IA consumida: {_si_no(traza.get('ia_consumida'))}",
        f"- Fragmentos recuperados/enviados: {traza.get('num_fragmentos') if traza.get('num_fragmentos') != '' else '-'}",
        f"- Reintentos IA: {traza.get('ia_reintentos') if traza.get('ia_reintentos') != '' else '-'}",
        f"- Orientación prudente contextual: {_si_no(traza.get('orientacion_prudente_contextual'))}",
        f"- Tipo de orientación: {traza.get('tipo_orientacion') or '-'}",
    ]
    return "\n".join(lineas)


def construir_mensajes(pregunta, contexto_xml):
    PROMPT_SISTEMA = """
Eres NormaEdu 2, un asistente de consulta normativa educativa española.
Tu función es ayudar a localizar y explicar normativa estatal y de Castilla y León a partir de fragmentos oficiales recuperados por la aplicación.

REGLA PRINCIPAL E INNEGOCIABLE:
Responde SOLO con la información contenida en los <fragmento> proporcionados. No uses conocimiento jurídico propio, memoria del modelo, internet ni inferencias no apoyadas por los fragmentos.

REGLAS DE FIABILIDAD JURÍDICA:
- Cada afirmación normativa concreta debe llevar una cita de fragmento con formato [F1], [F2], etc.
- Solo puedes citar identificadores que aparezcan en el contexto: [F1], [F2], [F3]...
- No cites artículos, disposiciones, leyes, decretos, órdenes, plazos, porcentajes, requisitos ni efectos jurídicos si no aparecen literalmente o de forma inequívoca en los fragmentos.
- Si los fragmentos no contienen la respuesta exacta, di: "Con los fragmentos recuperados no hay información suficiente para responder con seguridad".
- Si la pregunta pide un número exacto, una lista cerrada, un plazo o un artículo exacto, responde solo si ese dato aparece en los fragmentos.
- No completes con "información general" ni con conocimiento externo.
- No mezcles normas: distingue con cuidado Ley Orgánica, Real Decreto, Decreto autonómico y Orden autonómica.
- No afirmes que una norma está vigente, derogada o consolidada salvo que los fragmentos lo indiquen.
- No des asesoramiento jurídico individualizado ni tomes decisiones sobre alumnado, familias, docentes o centros.

FORMATO OBLIGATORIO:

## Respuesta
Respuesta directa y prudente. Incluye citas [F#] en las frases normativas.

## Base normativa encontrada
Tabla Markdown con columnas: Fragmento | Documento | Página | Qué acredita.
Usa solo fragmentos realmente utilizados en la respuesta.

## Límites de la respuesta
Indica qué no puede afirmarse con seguridad si los fragmentos son incompletos.

## Orientación práctica
Solo incluye pasos prácticos si se desprenden directamente de los fragmentos. Si no, indica que debe consultarse la fuente oficial o al órgano competente.

ESTILO:
- Español claro.
- No inventes.
- Mejor una respuesta incompleta pero fiable que una respuesta completa sin base documental.
"""

    # Seguridad jurídica: no arrastramos respuestas previas al LLM. Cada consulta
    # debe fundamentarse solo en la pregunta actual y en los fragmentos recuperados.
    mensajes = [{"role": "system", "content": PROMPT_SISTEMA}]
    mensajes.append({
        "role": "user",
        "content": f"CONTEXTO NORMATIVO:\n{contexto_xml}\n\nPREGUNTA: {pregunta}",
    })
    return mensajes

def guardar_log(bloque, preg_orig, preg_corr, num_res, tiempo_ms, tiene_resp):
    """Logging desactivado para mantener coste 0 y minimizar riesgos RGPD.

    No se guardan preguntas, respuestas ni metadatos en bases de datos externas.
    """
    return None


def guardar_feedback(pregunta, respuesta, util):
    """Feedback no persistente: no se envía a bases de datos externas ni a terceros."""
    return None


# =============================================================================
# MODO DIAGNÓSTICO
# =============================================================================
def _diagnostico_fragmentos(fragmentos):
    """Resumen seguro de fragmentos recuperados para depuración.

    No incluye el texto completo de los fragmentos ni la pregunta del usuario.
    Sirve para ver si Qdrant está recuperando documentos coherentes sin añadir coste.
    """
    resumen = []
    for i, res in enumerate(fragmentos or [], 1):
        nombre = (res.get("nombre_archivo", "") or "").replace(".pdf", "").replace("_", " ")
        resumen.append({
            "fragmento": f"F{i}",
            "documento": nombre,
            "pagina": res.get("pagina_num", ""),
            "bloque_payload": res.get("bloque", ""),
            "score_final": round(float(res.get("similarity", 0.0) or 0.0), 4),
            "score_vectorial": round(float(res.get("score_vectorial", 0.0) or 0.0), 4),
            "score_lexico": round(float(res.get("score_lexico", 0.0) or 0.0), 4),
            "bonus_bloque": round(float(res.get("bonus_bloque", 0.0) or 0.0), 4),
        })
    return resumen


def mostrar_diagnostico(diagnostico: dict):
    """Muestra diagnóstico solo cuando el modo desarrollo está activado."""
    if not diagnostico:
        return
    with st.expander("🔎 Diagnóstico técnico de la respuesta", expanded=False):
        st.caption(
            "Este panel es solo para desarrollo. No añade coste, no llama a APIs y no guarda datos personales."
        )
        st.json(diagnostico, expanded=False)


def _resumen_error_proveedor_ia(resp) -> str:
    """Devuelve un resumen corto y seguro del error de IA, sin exponer claves."""
    try:
        data = resp.json()
        raw = json.dumps(data, ensure_ascii=False)
    except Exception:
        raw = getattr(resp, "text", "") or ""
    raw = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer [oculto]", raw)
    raw = raw.replace(IA_API_KEY, "[clave_oculta]") if IA_API_KEY else raw
    return raw[:600]


def _clasificar_error_ia(resp):
    """Clasifica errores del proveedor de IA sin asumir que un 429 sea diario."""
    status = getattr(resp, "status_code", None)
    resumen = _resumen_error_proveedor_ia(resp)
    resumen_l = resumen.lower()

    if status == 429 or any(x in resumen_l for x in ["rate", "quota", "limit", "tokens", "too many"]):
        return "limite_temporal", resumen
    if status in (401, 403) or any(x in resumen_l for x in ["unauthorized", "forbidden", "api key", "api_key", "invalid key"]):
        return "configuracion", resumen
    if status and status >= 500:
        return "servicio_temporal", resumen
    return "error_api", resumen


def _respuesta_ia_temporal_no_disponible(tipo: str = "limite_temporal"):
    """Respuesta segura cuando la IA no acepta la petición tras los reintentos.

    r5 mínima: evita que un 429/servicio temporal aparezca como un fallo opaco.
    No inventa contenido normativo y conserva la recomendación de usar FAQ o reintentar.
    """
    if tipo == "limite_temporal":
        causa = "el proveedor de IA ha devuelto un límite temporal de uso o cola saturada"
    elif tipo == "servicio_temporal":
        causa = "el proveedor de IA no está disponible temporalmente"
    else:
        causa = "no se ha podido obtener respuesta de la IA"
    return (
        "## Respuesta no generada por límite temporal de IA\n\n"
        f"No puedo responder con seguridad ahora mismo porque {causa}. "
        "No se debe interpretar este mensaje como una respuesta normativa.\n\n"
        "## Qué puedes hacer\n"
        "- Reintentar la misma pregunta en unos minutos.\n"
        "- Reformularla de forma más concreta.\n"
        "- Consultar mientras tanto las FAQ verificadas, que no consumen IA.\n\n"
        "## Límites de la respuesta\n"
        "No se ha generado una respuesta sustantiva porque el proveedor de IA no aceptó la petición tras los reintentos configurados."
    )


def _mostrar_error_ia(resp, diagnostico_base=None, modo_diagnostico=False):
    """Muestra un error de IA genérico y preciso para el usuario final."""
    tipo, resumen = _clasificar_error_ia(resp)
    status = getattr(resp, "status_code", None)

    diagnostico = dict(diagnostico_base or {})
    diagnostico.update({
        "capa_usada": "RAG_IA",
        "consume_qdrant": True,
        "consume_ia": False,
        "ia_peticion_aceptada": False,
        "ia_error_tipo": tipo,
        "ia_http_status": status,
        "ia_error_resumen": resumen,
    })
    st.session_state.ultimo_diagnostico = diagnostico

    if tipo == "limite_temporal":
        st.warning(
            "⏳ La IA ha devuelto un límite temporal de uso del plan gratuito. "
            "Puedes reintentarlo en unos minutos. Las respuestas FAQ siguen funcionando sin consumir tokens de IA."
        )
    elif tipo == "configuracion":
        st.error(
            "❌ Error de configuración de la IA. Revisa las claves y parámetros de IA en los Secrets de Streamlit."
        )
    elif tipo == "servicio_temporal":
        st.warning(
            "⏳ La IA no está disponible temporalmente. Puedes reintentarlo en unos minutos. "
            "Las respuestas FAQ siguen funcionando sin consumir tokens de IA."
        )
    else:
        st.error(
            "❌ No se ha podido obtener respuesta de la IA. Puedes reintentarlo más tarde o formular una pregunta frecuente verificada."
        )

    if modo_diagnostico:
        mostrar_diagnostico(diagnostico)


def _post_ia_con_reintento(mensajes, modo_diagnostico=False):
    """Llama a la IA y reintenta de forma prudente ante límites temporales.

    Reintenta solo errores clasificados como límite temporal o servicio temporal.
    No cambia el proveedor, no usa servicios adicionales y mantiene coste 0.
    Devuelve (response, intentos), donde intentos no contiene claves ni contenido de la pregunta.
    """
    payload = {
        "model": IA_MODEL,
        "messages": mensajes,
        "temperature": 0.1,
        "max_tokens": MAX_TOKENS_RESPUESTA,
    }
    headers = {
        "Authorization": f"Bearer {IA_API_KEY}",
        "Content-Type": "application/json",
    }

    intentos = []
    ultimo_resp = None

    for intento in range(IA_REINTENTOS_TEMPORALES + 1):
        t_intento = time.time()
        try:
            resp = _requests.post(
                IA_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
        except _requests.exceptions.RequestException as e:
            class _RespuestaIAConfigError:
                status_code = 0
                text = "Error de configuración o conexión con el proveedor de IA."

                def json(self):
                    return {
                        "error": {
                            "message": "No se pudo llamar al proveedor de IA. Revisa IA_API_URL, IA_API_KEY e IA_MODEL."
                        }
                    }

            resp = _RespuestaIAConfigError()
            intentos.append({
                "intento": intento + 1,
                "http_status": 0,
                "tipo": "configuracion",
                "duracion_ms": round((time.time() - t_intento) * 1000, 2),
            })
            return resp, intentos

        ultimo_resp = resp

        if resp.status_code == 200:
            tipo = "ok"
            resumen = ""
        else:
            tipo, resumen = _clasificar_error_ia(resp)

        registro_intento = {
            "intento": intento + 1,
            "http_status": resp.status_code,
            "tipo": tipo,
            "duracion_ms": round((time.time() - t_intento) * 1000, 2),
        }
        intentos.append(registro_intento)

        if resp.status_code == 200:
            return resp, intentos

        puede_reintentar = tipo in ("limite_temporal", "servicio_temporal")
        quedan_reintentos = intento < IA_REINTENTOS_TEMPORALES
        if puede_reintentar and quedan_reintentos:
            espera = IA_REINTENTO_SEGUNDOS * (IA_REINTENTO_BACKOFF ** intento)
            registro_intento["reintento_espera_segundos"] = espera
            st.info(
                f"⏳ La IA ha devuelto un límite temporal. Reintentando en {espera} segundos..."
            )
            time.sleep(espera)
            continue

        return resp, intentos

    return ultimo_resp, intentos

# =============================================================================
# INTERFAZ — BARRA LATERAL
# =============================================================================

def _admin_solicitado_por_url() -> bool:
    """Activa el acceso admin solo si la URL contiene ?admin.

    Usamos una sola página de Streamlit para evitar que aparezcan opciones
    multipágina tipo "app/admin" en el menú lateral.
    """
    try:
        return "admin" in st.query_params
    except Exception:
        return False


with st.sidebar:
    st.markdown("### 📊 Sesión actual")
    st.caption(f"Consultas usadas: {st.session_state.consultas_sesion}/{MAX_PREGUNTAS_SESION}")
    if st.session_state.historial_completo:
        st.caption(f"Consultas en historial local: {len(st.session_state.historial_completo)}")
    st.info("Modo coste 0: no se guardan preguntas ni respuestas en bases de datos externas.")

    modo_diagnostico = False
    _admin_url = _admin_solicitado_por_url()

    if _admin_url and not st.session_state.get("admin_diagnostico_ok", False):
        st.markdown("### 🔐 Administración")
        if not ADMIN_DIAGNOSTIC_KEY:
            st.warning("El acceso de administración no está configurado.")
        else:
            _clave_admin = st.text_input("Clave de administrador", type="password")
            if _clave_admin:
                if hmac.compare_digest(str(_clave_admin), str(ADMIN_DIAGNOSTIC_KEY)):
                    st.session_state.admin_diagnostico_ok = True
                    st.session_state.modo_diagnostico = True
                    st.success("Acceso diagnóstico activado para esta sesión.")
                    st.rerun()
                else:
                    st.session_state.admin_diagnostico_ok = False
                    st.session_state.modo_diagnostico = False
                    st.error("Clave de administrador incorrecta.")

    if st.session_state.get("admin_diagnostico_ok", False):
        st.markdown("### 🔐 Administración")
        modo_diagnostico = st.checkbox(
            "🔎 Modo diagnóstico",
            value=st.session_state.get("modo_diagnostico", False),
            help="Muestra información técnica de depuración. Solo visible tras activar el acceso admin."
        )
        st.session_state.modo_diagnostico = modo_diagnostico
        if modo_diagnostico:
            st.caption("Diagnóstico activo: visible solo en esta sesión de administrador.")
        if st.button("Cerrar acceso diagnóstico"):
            st.session_state.admin_diagnostico_ok = False
            st.session_state.modo_diagnostico = False
            st.rerun()
    else:
        st.session_state.modo_diagnostico = False

st.title("📚 NormaEdu 2")

st.warning(
    "No introduzcas nombres, DNI, expedientes, datos médicos, sanciones ni "
    "información que permita identificar a alumnos, familias, docentes u otras personas. "
    "La app no guarda tus preguntas en bases de datos externas, pero la consulta se envía "
    "al servicio gratuito de IA para generar la respuesta."
)

bloque_elegido = st.selectbox(
    "Nivel educativo:",
    ["ninguno", "general", "infantil_primaria", "secundaria_bachillerato", "fp"],
    format_func=lambda x: {
        "ninguno":                 "— Selecciona un nivel educativo —",
        "general":                 "📋 General (permisos, bajas, vacaciones, EBEP...)",
        "infantil_primaria":       "🧒 Infantil y Primaria",
        "secundaria_bachillerato": "🎓 Secundaria y Bachillerato",
        "fp":                      "🔧 Formación Profesional",
    }[x],
)


with st.form(key="form_busqueda"):
    # No usamos value=st.session_state["pregunta_actual"] porque provocaba
    # un desfase visual: al enviar una pregunta nueva, el cuadro podía seguir
    # mostrando la pregunta anterior mientras la respuesta correspondía a la nueva.
    pregunta_input = st.text_area(
        "Haz tu pregunta sobre la normativa:",
        key="pregunta_input_widget",
        height=100, max_chars=MAX_CHARS_PREGUNTA,
        placeholder="Escribe tu consulta sobre normativa educativa...",
    )
    submit = st.form_submit_button("🔍 Buscar", use_container_width=True)

# =============================================================================
# PROCESAMIENTO
# =============================================================================
if submit and pregunta_input:

    if bloque_elegido == "ninguno":
        st.warning("⚠️ Selecciona un nivel educativo antes de buscar.")

    else:
        valido, msg_error = validar_input(pregunta_input)
        if not valido:
            st.warning(f"⚠️ {msg_error}")
        else:
            # 1) FAQ verificada local: no consume IA ni Qdrant.
            faq_match, faq_score = buscar_faq_verificada(pregunta_input, bloque_elegido)
            if faq_match:
                texto_final, fuentes_u, fuentes_up = construir_respuesta_faq(faq_match, faq_score)
                trazabilidad = construir_trazabilidad_historial(
                    ruta="FAQ",
                    apartado=bloque_elegido,
                    faq_id=faq_match.get("id", ""),
                    faq_score=round(float(faq_score), 4),
                    filtro_dominio=False,
                    qdrant=False,
                    ia=False,
                    orientacion_prudente=False,
                )

                st.write("---")
                st.markdown("### 📝 Respuesta:")
                st.success("Respuesta desde FAQ verificada local: no consume tokens de IA.")
                st.markdown(texto_final)
                st.markdown("### 📚 Fuentes consultadas:")
                for f in fuentes_u:
                    st.markdown(f"- 📄 {f}", unsafe_allow_html=False)
                st.caption(formatear_trazabilidad_compacta(trazabilidad))

                diagnostico = {
                    "version": "v073b_postvalidacion_r7_filtros_faq_prudencia",
                    "capa_usada": "FAQ",
                    "consume_ia": False,
                    "consume_qdrant": False,
                    "bloque_seleccionado": bloque_elegido,
                    "faq_id": faq_match.get("id", ""),
                    "faq_score": round(float(faq_score), 4),
                    "pregunta_canonica": faq_match.get("pregunta_canonica", ""),
                    "num_fuentes": len(fuentes_u),
                    "limite_ia_usado": f"{st.session_state.consultas_sesion}/{MAX_PREGUNTAS_SESION}",
                    "trazabilidad": trazabilidad,
                }
                st.session_state.ultimo_diagnostico = diagnostico
                if modo_diagnostico:
                    mostrar_diagnostico(diagnostico)

                # Las FAQ no consumen IA, por eso no incrementan consultas_sesion.
                st.session_state.ultima_pregunta   = pregunta_input
                st.session_state.pregunta_actual   = pregunta_input
                st.session_state.ultima_respuesta  = texto_final
                st.session_state.ultimas_fuentes   = fuentes_u
                st.session_state.ultima_trazabilidad = trazabilidad
                st.session_state.historial_completo.append({
                    "pregunta":           pregunta_input,
                    "pregunta_corregida": pregunta_input,
                    "respuesta":          texto_final,
                    "fuentes":            fuentes_up,
                    "trazabilidad":       trazabilidad,
                })
                if len(st.session_state.historial_completo) > MAX_HISTORIAL_LOCAL:
                    st.session_state.historial_completo = \
                        st.session_state.historial_completo[-MAX_HISTORIAL_LOCAL:]

                st.session_state.feedback_pendiente = True
                st.session_state.feedback_pregunta  = pregunta_input
                st.session_state.feedback_respuesta = texto_final

            elif _pregunta_seguridad_defensiva(pregunta_input):
                # r4: filtro defensivo mínimo antes de consultar Qdrant o IA.
                texto_final = construir_respuesta_seguridad_defensiva(pregunta_input, bloque_elegido)
                fuentes_u = ["Filtro de seguridad: no se consultó Qdrant ni IA."]
                fuentes_up = fuentes_u[:]
                trazabilidad = construir_trazabilidad_historial(
                    ruta="FILTRO_SEGURIDAD",
                    apartado=bloque_elegido,
                    faq_id=None,
                    faq_score=None,
                    filtro_dominio=False,
                    qdrant=False,
                    ia=False,
                    orientacion_prudente=False,
                )

                st.write("---")
                st.markdown("### 📝 Respuesta:")
                st.warning("Consulta bloqueada por seguridad: no consume Qdrant ni IA.")
                st.markdown(texto_final)
                st.markdown("### 📚 Fuentes consultadas:")
                for f in fuentes_u:
                    st.markdown(f"- 📄 {f}", unsafe_allow_html=False)
                st.caption(formatear_trazabilidad_compacta(trazabilidad))

                diagnostico = {
                    "version": "v073b_postvalidacion_r7_filtros_faq_prudencia",
                    "capa_usada": "FILTRO_SEGURIDAD",
                    "consume_ia": False,
                    "consume_qdrant": False,
                    "bloque_seleccionado": bloque_elegido,
                    "faq_id": None,
                    "motivo": "pregunta_seguridad_defensiva",
                    "limite_ia_usado": f"{st.session_state.consultas_sesion}/{MAX_PREGUNTAS_SESION}",
                    "trazabilidad": trazabilidad,
                }
                st.session_state.ultimo_diagnostico = diagnostico
                if modo_diagnostico:
                    mostrar_diagnostico(diagnostico)

                st.session_state.ultima_pregunta   = pregunta_input
                st.session_state.pregunta_actual   = pregunta_input
                st.session_state.ultima_respuesta  = texto_final
                st.session_state.ultimas_fuentes   = fuentes_u
                st.session_state.ultima_trazabilidad = trazabilidad
                st.session_state.historial_completo.append({
                    "pregunta":           pregunta_input,
                    "pregunta_corregida": pregunta_input,
                    "respuesta":          texto_final,
                    "fuentes":            fuentes_up,
                    "trazabilidad":       trazabilidad,
                })
                if len(st.session_state.historial_completo) > MAX_HISTORIAL_LOCAL:
                    st.session_state.historial_completo = \
                        st.session_state.historial_completo[-MAX_HISTORIAL_LOCAL:]

                st.session_state.feedback_pendiente = True
                st.session_state.feedback_pregunta  = pregunta_input
                st.session_state.feedback_respuesta = texto_final

            elif _pregunta_fuera_de_dominio(pregunta_input, bloque_elegido):
                # v067: filtro conservador de dominio antes de consultar Qdrant o IA.
                texto_final = construir_respuesta_fuera_dominio(pregunta_input, bloque_elegido)
                fuentes_u = ["Filtro de dominio: no se consultó Qdrant ni IA."]
                fuentes_up = fuentes_u[:]
                trazabilidad = construir_trazabilidad_historial(
                    ruta="FILTRO_DOMINIO",
                    apartado=bloque_elegido,
                    faq_id=None,
                    faq_score=None,
                    filtro_dominio=True,
                    qdrant=False,
                    ia=False,
                    orientacion_prudente=False,
                )

                st.write("---")
                st.markdown("### 📝 Respuesta:")
                st.warning("Consulta fuera del ámbito educativo/docente de la app: no consume Qdrant ni IA.")
                st.markdown(texto_final)
                st.markdown("### 📚 Fuentes consultadas:")
                for f in fuentes_u:
                    st.markdown(f"- 📄 {f}", unsafe_allow_html=False)
                st.caption(formatear_trazabilidad_compacta(trazabilidad))

                diagnostico = {
                    "version": "v073b_postvalidacion_r7_filtros_faq_prudencia",
                    "capa_usada": "FILTRO_DOMINIO",
                    "consume_ia": False,
                    "consume_qdrant": False,
                    "bloque_seleccionado": bloque_elegido,
                    "faq_id": None,
                    "motivo": "pregunta_fuera_de_dominio",
                    "limite_ia_usado": f"{st.session_state.consultas_sesion}/{MAX_PREGUNTAS_SESION}",
                    "trazabilidad": trazabilidad,
                }
                st.session_state.ultimo_diagnostico = diagnostico
                if modo_diagnostico:
                    mostrar_diagnostico(diagnostico)

                st.session_state.ultima_pregunta   = pregunta_input
                st.session_state.pregunta_actual   = pregunta_input
                st.session_state.ultima_respuesta  = texto_final
                st.session_state.ultimas_fuentes   = fuentes_u
                st.session_state.ultima_trazabilidad = trazabilidad
                st.session_state.historial_completo.append({
                    "pregunta":           pregunta_input,
                    "pregunta_corregida": pregunta_input,
                    "respuesta":          texto_final,
                    "fuentes":            fuentes_up,
                    "trazabilidad":       trazabilidad,
                })
                if len(st.session_state.historial_completo) > MAX_HISTORIAL_LOCAL:
                    st.session_state.historial_completo =                         st.session_state.historial_completo[-MAX_HISTORIAL_LOCAL:]

                st.session_state.feedback_pendiente = True
                st.session_state.feedback_pregunta  = pregunta_input
                st.session_state.feedback_respuesta = texto_final

            elif st.session_state.consultas_sesion >= MAX_PREGUNTAS_SESION:
                st.error(
                    "Se ha alcanzado el límite gratuito de consultas de esta sesión. "
                    "Vuelve más tarde para seguir usando la app sin coste."
                )

            else:
                try:
                    t0 = time.time()

                    with st.spinner("✏️ Analizando la consulta..."):
                        pregunta_corregida, reformulaciones = expandir_y_corregir(pregunta_input)

                    if pregunta_corregida.strip().lower() != pregunta_input.strip().lower():
                        st.info(f"✏️ He corregido tu consulta a: **{pregunta_corregida}**")

                    with st.spinner("🔎 Buscando en la normativa..."):
                        # e5 requiere prefijo "query: " en las consultas
                        todas = [pregunta_corregida] + reformulaciones[:2]
                        embedding_avg = np.mean(
                            [model.encode('query: ' + q, normalize_embeddings=True)
                             for q in todas], axis=0
                        ).tolist()
                        resultados = buscar_normativa_hibrida(
                            embedding_avg, pregunta_corregida, bloque_elegido
                        )

                    if not resultados:
                        st.warning("No encontré normativa relacionada. Prueba a reformular la pregunta.")
                        diagnostico = {
                            "version": "v073b_postvalidacion_r7_filtros_faq_prudencia",
                            "capa_usada": "RAG",
                            "estado": "sin_resultados",
                            "consume_qdrant": True,
                            "consume_ia": False,
                            "bloque_seleccionado": bloque_elegido,
                            "faq_id": None,
                            "resultados_recuperados": 0,
                            "tiempo_ms": round((time.time()-t0)*1000, 2),
                            "limite_ia_usado": f"{st.session_state.consultas_sesion}/{MAX_PREGUNTAS_SESION}",
                        }
                        st.session_state.ultimo_diagnostico = diagnostico
                        if modo_diagnostico:
                            mostrar_diagnostico(diagnostico)
                        guardar_log(bloque_elegido, pregunta_input, pregunta_corregida,
                                    0, (time.time()-t0)*1000, False)
                    else:
                        with st.spinner("📊 Ordenando por relevancia..."):
                            resultados = reranquear(pregunta_corregida, resultados)
                            resultados = resultados[:MATCH_COUNT]

                        contexto_xml, links_screen, fuentes_pdf = construir_contexto_xml(
                            resultados, enlaces
                        )
                        contexto_xml = recortar_contexto_xml(contexto_xml)
                        ids_contexto_validos = ids_fragmentos_validos_contexto(contexto_xml)
                        mensajes = construir_mensajes(pregunta_corregida, contexto_xml)

                        st.write("---")
                        st.markdown("### 📝 Respuesta:")

                        _resp, _intentos_ia = _post_ia_con_reintento(
                            mensajes, modo_diagnostico=modo_diagnostico
                        )
                        if _resp.status_code != 200:
                            diagnostico_base = {
                                "version": "v073b_postvalidacion_r7_filtros_faq_prudencia",
                                "bloque_seleccionado": bloque_elegido,
                                "resultados_enviados_llm": len(resultados),
                                "fragmentos": _diagnostico_fragmentos(resultados),
                                "contexto_chars": len(contexto_xml),
                                "fragmentos_validos_contexto": [f"F{x}" for x in ids_contexto_validos],
                                "contexto_limite_chars": MAX_CHARS_CONTEXTO,
                                "max_tokens_respuesta": MAX_TOKENS_RESPUESTA,
                                "limite_ia_actual": f"{st.session_state.consultas_sesion}/{MAX_PREGUNTAS_SESION}",
                                "ia_intentos": _intentos_ia,
                                "ia_reintentos_usados": max(0, len(_intentos_ia) - 1),
                                "tiempo_ms": round((time.time()-t0)*1000, 2),
                            }
                            tipo_error_ia, _resumen_error_ia = _clasificar_error_ia(_resp)
                            if tipo_error_ia in ("limite_temporal", "servicio_temporal"):
                                st.warning(
                                    "⏳ La IA no ha aceptado la petición tras los reintentos. "
                                    "Se muestra una respuesta segura sin contenido normativo inventado."
                                )
                                texto_final = _respuesta_ia_temporal_no_disponible(tipo_error_ia)
                                ruta_trazabilidad = "RAG_IA_NO_DISPONIBLE"
                                tipo_orientacion_prudente = "ia_temporal_no_disponible"
                                fuentes_u = list(dict.fromkeys(links_screen))
                                fuentes_up = list(dict.fromkeys(fuentes_pdf))
                                st.markdown(texto_final)
                                trazabilidad = construir_trazabilidad_historial(
                                    ruta=ruta_trazabilidad,
                                    apartado=bloque_elegido,
                                    faq_id=None,
                                    faq_score=None,
                                    filtro_dominio=False,
                                    qdrant=True,
                                    ia=False,
                                    orientacion_prudente=False,
                                    tipo_orientacion=tipo_orientacion_prudente,
                                    num_fragmentos=len(resultados),
                                    ia_reintentos=max(0, len(_intentos_ia) - 1),
                                )
                                st.markdown("### 📚 Fuentes consultadas:")
                                for f in fuentes_u:
                                    st.markdown(f"- 📄 {f}", unsafe_allow_html=False)
                                st.caption(formatear_trazabilidad_compacta(trazabilidad))

                                diagnostico = dict(diagnostico_base)
                                diagnostico.update({
                                    "capa_usada": ruta_trazabilidad,
                                    "consume_qdrant": True,
                                    "consume_ia": False,
                                    "ia_peticion_aceptada": False,
                                    "ia_error_tipo": tipo_error_ia,
                                    "ia_error_resumen": _resumen_error_ia,
                                    "trazabilidad": trazabilidad,
                                })
                                st.session_state.ultimo_diagnostico = diagnostico
                                if modo_diagnostico:
                                    mostrar_diagnostico(diagnostico)

                                st.session_state.ultima_pregunta = pregunta_input
                                st.session_state.pregunta_actual = pregunta_input
                                st.session_state.ultima_respuesta = texto_final
                                st.session_state.ultimas_fuentes = fuentes_u
                                st.session_state.ultima_trazabilidad = trazabilidad
                                st.session_state.historial_completo.append({
                                    "pregunta": pregunta_input,
                                    "pregunta_corregida": pregunta_corregida,
                                    "respuesta": texto_final,
                                    "fuentes": fuentes_up,
                                    "trazabilidad": trazabilidad,
                                })
                                if len(st.session_state.historial_completo) > MAX_HISTORIAL_LOCAL:
                                    st.session_state.historial_completo = st.session_state.historial_completo[-MAX_HISTORIAL_LOCAL:]
                                st.stop()
                            else:
                                _mostrar_error_ia(_resp, diagnostico_base, modo_diagnostico)
                                st.stop()

                        texto_final = _resp.json()["choices"][0]["message"]["content"]

                        citas_ok, citas_detectadas, citas_invalidas = validar_citas_fragmentos(
                            texto_final, len(resultados), ids_validos=ids_contexto_validos
                        )
                        orientacion_prudente_aplicada = False
                        tipo_orientacion_prudente = ""
                        ruta_trazabilidad = "RAG_IA"

                        if not citas_ok:
                            texto_final = respuesta_segura_por_citas_invalidas(citas_invalidas)
                            ruta_trazabilidad = "RAG_IA_CITAS_BLOQUEADAS"
                            st.warning("La respuesta generada citaba fragmentos inexistentes y ha sido bloqueada.")
                        else:
                            texto_antes_prudencia = texto_final
                            texto_final = reforzar_respuesta_prudente_contextual(
                                texto_final, pregunta_input, bloque_elegido
                            )
                            orientacion_prudente_aplicada = texto_final != texto_antes_prudencia
                            orientacion_final_detectada = respuesta_contiene_orientacion_contextual(
                                texto_final, pregunta_input
                            )
                            if orientacion_prudente_aplicada or orientacion_final_detectada:
                                ruta_trazabilidad = "RAG_IA_PRUDENTE"
                                tipo_orientacion_prudente = _tipo_orientacion_contextual(pregunta_input)
                                orientacion_prudente_aplicada = True
                            if not citas_detectadas:
                                st.warning(
                                    "La respuesta no contiene citas [F#]. Revísala con especial cautela; "
                                    "en la siguiente fase podremos hacer este control aún más estricto."
                                )

                        st.markdown(texto_final)

                        fuentes_u  = list(dict.fromkeys(links_screen))
                        fuentes_up = list(dict.fromkeys(fuentes_pdf))
                        trazabilidad = construir_trazabilidad_historial(
                            ruta=ruta_trazabilidad,
                            apartado=bloque_elegido,
                            faq_id=None,
                            faq_score=None,
                            filtro_dominio=False,
                            qdrant=True,
                            ia=True,
                            orientacion_prudente=orientacion_prudente_aplicada,
                            tipo_orientacion=tipo_orientacion_prudente,
                            num_fragmentos=len(resultados),
                            ia_reintentos=max(0, len(_intentos_ia) - 1),
                        )
                        st.markdown("### 📚 Fuentes consultadas:")
                        for f in fuentes_u:
                            st.markdown(f"- 📄 {f}", unsafe_allow_html=False)
                        st.caption(formatear_trazabilidad_compacta(trazabilidad))

                        diagnostico = {
                            "version": "v073b_postvalidacion_r7_filtros_faq_prudencia",
                            "capa_usada": ruta_trazabilidad,
                            "consume_qdrant": True,
                            "consume_ia": True,
                            "bloque_seleccionado": bloque_elegido,
                            "faq_id": None,
                            "resultados_enviados_llm": len(resultados),
                            "fragmentos": _diagnostico_fragmentos(resultados),
                            "contexto_chars": len(contexto_xml),
                            "fragmentos_validos_contexto": [f"F{x}" for x in ids_contexto_validos],
                            "contexto_limite_chars": MAX_CHARS_CONTEXTO,
                            "max_tokens_respuesta": MAX_TOKENS_RESPUESTA,
                            "ia_intentos": _intentos_ia,
                            "ia_reintentos_usados": max(0, len(_intentos_ia) - 1),
                            "ia_peticion_aceptada": True,
                            "citas_detectadas": [f"[F{x}]" for x in citas_detectadas],
                            "citas_invalidas": [f"[F{x}]" for x in citas_invalidas],
                            "citas_validas": citas_ok,
                            "orientacion_prudente_contextual": orientacion_prudente_aplicada,
                            "tipo_orientacion_prudente": tipo_orientacion_prudente,
                            "trazabilidad": trazabilidad,
                            "tiempo_ms": round((time.time()-t0)*1000, 2),
                            "limite_ia_antes_de_incrementar": f"{st.session_state.consultas_sesion}/{MAX_PREGUNTAS_SESION}",
                        }
                        st.session_state.ultimo_diagnostico = diagnostico
                        if modo_diagnostico:
                            mostrar_diagnostico(diagnostico)

                        st.session_state.consultas_sesion += 1
                        st.session_state.ultima_pregunta   = pregunta_input
                        st.session_state.pregunta_actual   = pregunta_input
                        st.session_state.ultima_respuesta  = texto_final
                        st.session_state.ultimas_fuentes   = fuentes_u
                        st.session_state.ultima_trazabilidad = trazabilidad
                        st.session_state.historial_completo.append({
                            "pregunta":           pregunta_input,
                            "pregunta_corregida": pregunta_corregida,
                            "respuesta":          texto_final,
                            "fuentes":            fuentes_up,
                            "trazabilidad":       trazabilidad,
                        })
                        if len(st.session_state.historial_completo) > MAX_HISTORIAL_LOCAL:
                            st.session_state.historial_completo = \
                                st.session_state.historial_completo[-MAX_HISTORIAL_LOCAL:]

                        st.session_state.feedback_pendiente = True
                        st.session_state.feedback_pregunta  = pregunta_input
                        st.session_state.feedback_respuesta = texto_final

                        guardar_log(bloque_elegido, pregunta_input, pregunta_corregida,
                                    len(resultados), (time.time()-t0)*1000, True)

                except Exception as e:
                    err = str(e).lower()
                    if "qdrant" in err:
                        st.error(f"❌ Error en Qdrant: {e}")
                    elif "429" in err or "quota" in err or "exhausted" in err or "rate" in err:
                        st.warning("⏳ La IA ha devuelto un límite temporal de uso del plan gratuito. Puedes reintentarlo en unos minutos. Las respuestas FAQ siguen funcionando sin consumir tokens de IA.")
                    elif "api_key" in err or "invalid" in err:
                        st.error("❌ Error de configuración de la IA. Revisa las claves y parámetros de IA en los Secrets de Streamlit.")
                    else:
                        st.error(f"Error técnico: {e}")

elif st.session_state.ultima_respuesta:
    st.write("---")
    st.markdown(st.session_state.ultima_respuesta)
    st.markdown("### 📚 Fuentes consultadas:")
    for f in st.session_state.ultimas_fuentes:
        st.markdown(f"- 📄 {f}", unsafe_allow_html=False)
    if st.session_state.get("ultima_trazabilidad"):
        st.caption(formatear_trazabilidad_compacta(st.session_state.ultima_trazabilidad))
    if modo_diagnostico:
        mostrar_diagnostico(st.session_state.get("ultimo_diagnostico"))

# =============================================================================
# FEEDBACK
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
            st.success("¡Gracias! Feedback recibido solo en esta sesión; no se guarda en base de datos.")
            st.rerun()
    with c2:
        if st.button("👎 No"):
            guardar_feedback(st.session_state.feedback_pregunta,
                             st.session_state.feedback_respuesta, False)
            st.session_state.feedback_pendiente = False
            st.info("Gracias. Feedback recibido solo en esta sesión; no se guarda en base de datos.")
            st.rerun()

# =============================================================================
# HISTORIAL EN PANTALLA
# =============================================================================
historial = st.session_state.historial_completo
if len(historial) > 1:
    st.write("---")
    with st.expander(f"📋 Historial ({len(historial)} consultas)", expanded=False):
        for item in reversed(historial[:-1]):
            st.markdown(f"**Pregunta:** {item['pregunta']}")
            prev = item["respuesta"]
            st.markdown(prev[:400] + "..." if len(prev) > 400 else prev)
            if item.get("trazabilidad"):
                st.caption(formatear_trazabilidad_compacta(item.get("trazabilidad")))
            st.divider()

# =============================================================================
# BOTONES DE ACCIÓN
# =============================================================================
if historial:
    st.write("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("📄 Descargar esta consulta",
            data=generar_pdf([historial[-1]], "Consulta de Normativa Educativa"),
            file_name="consulta_normativa.pdf", mime="application/pdf",
            use_container_width=True)
    with c2:
        st.download_button("📚 Descargar historial",
            data=generar_pdf(historial, "Historial Completo"),
            file_name="historial_normativa.pdf", mime="application/pdf",
            use_container_width=True)
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
                    consultas_usadas = st.session_state.get("consultas_sesion", 0)
                    for k, v in _DEFAULTS.items():
                        st.session_state[k] = ([] if isinstance(v, list) else v)
                    # Reiniciar el chat no reinicia el contador de uso del free tier.
                    st.session_state.consultas_sesion = consultas_usadas
                    st.rerun()
            with cb:
                if st.button("❌ Cancelar", use_container_width=True):
                    st.session_state.confirmar_borrar = False
                    st.rerun()
