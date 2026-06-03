# NormaEdu 2

App Streamlit de consulta de normativa educativa con FAQ verificadas, RAG con Qdrant y una capa de IA configurada mediante Secrets. La versión pública no muestra el proveedor de IA a los usuarios.

## Archivos principales

- `app.py`: aplicación principal.
- `requirements.txt`: dependencias.
- `faq_normativa.json`: base local de FAQ verificadas.
- `MATRIZ_VERIFICACION_FAQ.md`: trazabilidad de las FAQ.
- `enlaces.csv`: correspondencia entre documentos y enlaces oficiales.
- `.streamlit/secrets.toml.example`: plantilla segura de configuración.

## Configuración de Secrets en Streamlit Cloud

Configura estos valores en **Manage app → Settings → Secrets**:

```toml
IA_API_KEY = "pega_aqui_tu_clave_de_ia"
IA_API_URL = "https://endpoint-del-proveedor/v1/chat/completions"
IA_MODEL = "nombre-del-modelo"
QDRANT_URL = "https://tu-cluster.cloud.qdrant.io"
QDRANT_API_KEY = "pega_aqui_tu_clave_de_qdrant"

# Opcional: solo para administradores. Permite activar diagnóstico accediendo a ?admin.
ADMIN_DIAGNOSTIC_KEY = "elige_una_clave_larga_para_admin"
```

No subas nunca claves reales a GitHub. El archivo `.streamlit/secrets.toml.example` es solo una plantilla.

## Control de coste 0

- Las FAQ verificadas responden sin usar Qdrant ni IA.
- Las consultas RAG usan Qdrant y la IA solo cuando no hay FAQ aplicable.
- Existe un límite duro de consultas IA por sesión.
- Si la IA devuelve un límite temporal, la app hace un reintento automático suave y no lo presenta como límite diario definitivo.

## Control de fiabilidad jurídica

- Prompt jurídico estricto: la respuesta debe basarse en fragmentos recuperados.
- Citas obligatorias por fragmento `[F1]`, `[F2]`, etc.
- Si la IA cita fragmentos inexistentes, la respuesta se bloquea.
- Si no hay base suficiente en los fragmentos, la app debe responder de forma prudente.

## FAQ normativa verificada

La app incluye `faq_normativa.json` con 163 FAQ verificadas. Esta capa se consulta antes del RAG para ahorrar tokens y reducir errores en preguntas frecuentes.

## Modo diagnóstico protegido

El modo diagnóstico ya no es visible para usuarios normales. Para activarlo:

1. Añade `ADMIN_DIAGNOSTIC_KEY` en los Secrets de Streamlit.
2. Abre la app añadiendo `?admin` al final de la URL, por ejemplo `https://tu-app.streamlit.app/?admin`.
3. Introduce la clave de administrador en la barra lateral.
4. Activa `🔎 Modo diagnóstico`.

Permite ver:

- si la respuesta viene de FAQ o RAG/IA;
- qué FAQ se activó y con qué puntuación;
- si se ha usado Qdrant;
- si se ha usado IA;
- fragmentos recuperados y puntuaciones;
- citas detectadas e inválidas;
- errores temporales de IA y reintentos si los hay.

No guarda datos personales ni añade coste.

## v056

Esta versión mantiene los cambios de v055 y cambia el acceso de administrador: el modo diagnóstico se activa entrando en la ruta `?admin`, por ejemplo `https://tu-app.streamlit.app/?admin`, y usando la clave definida en `ADMIN_DIAGNOSTIC_KEY`. Los usuarios normales no ven el botón de diagnóstico.


## Acceso diagnóstico privado

El modo diagnóstico no se muestra a los usuarios normales.

Para activarlo:

1. Añade en Streamlit Secrets:

```toml
ADMIN_DIAGNOSTIC_KEY = "elige_una_clave_larga_y_privada"
```

2. Entra en:

```text
https://tu-app.streamlit.app/?admin
```

3. Introduce la clave en la barra lateral.

No se usa `pages/admin.py`, para evitar que Streamlit muestre opciones `app/admin` en el menú lateral.


## v060 - Ajuste de FAQ de Primaria

La FAQ `primaria_no_promocion_plan_refuerzo` se ha reforzado para cubrir formulaciones como:

```text
En Primaria, si un alumno no promociona, ¿debe tener algún plan específico?
```

No se ha duplicado la FAQ porque ya existía una respuesta verificada sobre el plan específico de refuerzo tras no promocionar en Primaria.


## v061 - Repetición en Primaria y corrección del campo de pregunta

Cambios mínimos:
- La pregunta «¿En qué cursos de Primaria puede decidirse la repetición?» activa ahora la FAQ `primaria_repetir_cuando_condiciones_cyl`.
- Se corrige un desfase visual del cuadro de pregunta: la app podía mostrar la pregunta anterior mientras respondía a la nueva.
- No se modifican Qdrant, IA, prompt, admin, privacidad ni límites.


## v062 - Primer paquete de FAQ básicas

Añade un paquete de FAQ básicas tras auditar 120 preguntas frecuentes. Incluye cobertura de:
- áreas/asignaturas de Primaria;
- Religión, alternativa y Valores Cívicos y Éticos;
- calificaciones y evaluación en Primaria;
- materias y estructura básica de ESO;
- materias, modalidades y promoción básica en Bachillerato;
- conceptos básicos de FP;
- formación en empresa, régimen general/intensivo y norma estatal de FP;
- uso de fuentes oficiales.

También refuerza variantes de FAQ ya existentes sobre convivencia, permisos, privacidad y evaluación objetiva.


## v064 - Robustez coloquial

Cambios mínimos:
- Refuerza variantes coloquiales reales detectadas en pruebas de usuario.
- Añade una FAQ defensiva: trabajar en un hospital no genera por sí solo derecho a permiso.
- Corrige activación de FAQ para ESO con suspensos, Bachillerato con suspensas, FP general/intensiva, título de Técnico, grado medio, fallecimiento de padre/madre y alumno que no deja dar clase.
- No modifica Qdrant, IA, prompt, admin oculto, privacidad ni límites.


## v064b - Corrección preventiva de despliegue

Mantiene los cambios de v064 y añade:
- versión unificada en diagnósticos;
- eliminación de una modificación potencialmente conflictiva de `pregunta_input_widget` en `st.session_state`.

No cambia FAQ, IA, Qdrant, prompt, admin, límites ni privacidad.

## v065 - Corrección quirúrgica de variantes

- Añade FAQ defensiva `fp_ciclos_por_familia_no_estable`.
- Refuerza variantes cortas/coloquiales prioritarias.
- No modifica IA, Qdrant, prompt, admin oculto, privacidad, límites ni reintentos.


## v066 - Trazabilidad jurídica de fuentes

- Refuerza fuentes de FAQ prioritarias con norma exacta, artículo/apartado y fragmento oficial.
- Mejora trazabilidad en FP, Bachillerato, permisos, convivencia, Primaria, Infantil y privacidad.
- No modifica la lógica de matching, Qdrant, IA, admin oculto, límites, reintentos ni interfaz.


## v067 - Filtro de dominio antes de RAG

Cambios:
- Añade un filtro conservador antes de Qdrant/RAG para preguntas claramente fuera del ámbito educativo/docente.
- Evita que consultas de derecho privado, mercantil o contractual general recuperen normativa irrelevante.
- No modifica FAQ, Qdrant, IA, prompt, admin oculto, privacidad, límites ni reintentos.
- Las preguntas bloqueadas no consumen Qdrant ni IA.


## v068b - Rate limit IA corregido

Corrección sobre v068:
- Se mantiene v067 estable como base.
- Se refuerza `_post_ia_con_reintento`.
- Ante límite/servicio temporal IA: hasta 3 intentos totales.
- Esperas progresivas: 20s y 40s.
- Se registra `reintento_espera_segundos` en el diagnóstico técnico.
- No cambia FAQ, Qdrant, prompt, filtro de dominio, admin, privacidad ni interfaz.


## v069 - Filtro de dominio y variantes sintéticas

Cambios:
- Amplía el filtro de dominio con supuestos fuera del ámbito educativo detectados en auditoría sintética.
- Añade variantes FAQ prioritarias para reducir derivaciones innecesarias a RAG.
- Añade reglas de intención muy acotadas para evitar falsos positivos de matching en Bachillerato y FP.
- No modifica Qdrant, IA, prompt jurídico, admin oculto, privacidad, interfaz ni número de FAQ.


## v070 - Respuestas prudentes contextuales

Cambios:
- Añade orientación práctica cuando la respuesta RAG/IA ya reconoce que no hay información suficiente.
- Se aplica solo a preguntas actuales, locales, de centro o individualizadas.
- No modifica FAQ, Qdrant, prompt, filtro de dominio, rate limit, privacidad ni interfaz general.
- Objetivo: evitar respuestas secas y orientar a fuente oficial, centro, Dirección Provincial o catálogo actualizado.


## v070b - Corrección de prioridad en respuestas prudentes contextuales

Cambios:
- Ajusta el orden de las reglas de orientación práctica.
- Horario, transporte, comedor, plazas, libros, optativas, profesorado y convenios tienen prioridad sobre reglas generales de FP/centro.
- No modifica FAQ, Qdrant, prompt, filtro de dominio, rate limit, privacidad ni interfaz general.


## v071 - Trazabilidad compacta del historial

Cambios:
- Añade trazabilidad compacta al historial en pantalla y a la exportación PDF.
- Registra ruta usada: FAQ, FILTRO_DOMINIO, RAG_IA, RAG_IA_PRUDENTE o RAG_IA_CITAS_BLOQUEADAS.
- Registra si se consultó Qdrant, si se consumió IA, FAQ activada y orientación prudente contextual.
- No modifica FAQ, Qdrant, IA, prompt, filtro de dominio, rate limit ni respuestas.


## v071b - Corrección de trazabilidad de orientación contextual

Cambios:
- Corrige la trazabilidad cuando la propia IA ya genera una sección de orientación práctica contextual.
- En esos casos, la ruta pasa a reflejar `RAG_IA_PRUDENTE`.
- No modifica la respuesta mostrada, FAQ, Qdrant, prompt, filtro de dominio, rate limit ni privacidad.


## v071c - Afinado de orientación FP en trazabilidad

Cambios:
- Ajusta el tipo de orientación contextual para preguntas de oferta FP por familia, provincia o curso.
- Ejemplo: `informática y comunicaciones este curso en mi provincia` pasa a `tipo_orientacion = oferta_fp`.
- No modifica respuestas, FAQ, Qdrant, IA, prompt, filtro de dominio, rate limit ni privacidad.


## v072 - Precisión de enrutamiento FAQ/RAG

Cambios:
- Evita que la FAQ genérica `fp_norma_estatal_rd659` capture preguntas específicas sobre FCT, requisitos o formación en centros de trabajo.
- Añade variantes y regla para que `medidas inmediatas` active `cyl_actuaciones_inmediatas_convivencia`.
- No modifica Qdrant, IA, prompt jurídico, filtro de dominio, rate limit, trazabilidad ni privacidad.


## v073 - Precisión/ranking Qdrant sin reindexar

Cambios:
- Añade reformulaciones locales sin IA para consultas de evaluación objetiva y procedimiento corrector de convivencia.
- Refuerza el score léxico normalizado.
- Añade bonus y penalizaciones acotadas antes del corte Top8.
- No modifica la colección Qdrant, embeddings, IA, prompt, FAQ, filtro de dominio, rate limit ni trazabilidad.


## v073b - Routing evaluación objetiva

Cambios:
- Mantiene la mejora de ranking Qdrant de v073 para convivencia/procedimiento corrector.
- Añade regla y variantes para que preguntas como `la prueba objetiva tipo test cuenta como evaluación objetiva` activen `alumnado_derecho_evaluacion_objetiva`.
- No modifica colección Qdrant, embeddings, IA, prompt jurídico, filtro de dominio, rate limit ni trazabilidad.


## v073b post-validación - FAQ/citas/docfix

Cambios mínimos tras la validación sintética rigurosa post-v073b:
- Corrige la documentación para reflejar las 163 FAQ verificadas reales.
- Añade una tabla explícita de prioridad para variantes exactas duplicadas entre FAQ verificadas, evitando que el resultado dependa del orden del JSON.
- Refuerza la validación de citas IA comprobando los identificadores [F#] realmente presentes en el contexto final enviado al modelo tras el recorte de seguridad.
- No modifica Qdrant, embeddings, colección, proveedor IA, límites de sesión, privacidad ni filtro de dominio.


## v073b post-validación r4 - filtro defensivo mínimo

Cambios mínimos tras la prueba manual en Streamlit:

- Añade un filtro defensivo previo a Qdrant/IA para peticiones de claves, tokens, secretos, prompt del sistema o instrucciones internas.
- Añade la ruta de trazabilidad `FILTRO_SEGURIDAD`.
- Evita consumir Qdrant e IA ante intentos claros de prompt injection.
- No modifica Qdrant, IA, prompts normativos, umbrales, FAQ ni respuestas normativas ordinarias.

## v073b post-validación r5 - robustez FAQ y 429

Cambios mínimos tras la batería de 25 preguntas reales:

- Refuerza rutas FAQ para hospitalización de padre/madre, áreas de Infantil, siglas de calificación en Primaria, modalidades de Bachillerato y promoción con dos materias.
- Mantiene respuestas prudentes ante indisponibilidad temporal del proveedor IA.
- No modifica Qdrant, embeddings, colección ni estructura general.

## v073b post-validación r6 - ajustes finos del piloto

Cambios mínimos tras el piloto interno:

- Refuerza `vacaciones_docentes_agosto`.
- Prioriza `primaria_atencion_educativa_no_religion` cuando se pregunta qué hace el alumnado que no elige Religión.
- Refuerza `fp_modalidades_presencial_semipresencial_virtual` para preguntas sobre FP virtual.
- No modifica Qdrant, embeddings, proveedor IA ni arquitectura.

## v073b post-validación r7 - filtros, FAQ y prudencia

Cambios mínimos tras el piloto exhaustivo de 202 casos:

- Refuerza el filtro de dominio para preguntas fiscales/tributarias ajenas a educación, como IRPF de autónomos.
- Añade variantes abreviadas de Bachillerato, como `modalidades bach`.
- Resuelve el solapamiento entre `bachillerato_modalidades` y `bachillerato_modalidades_basica` en variantes concretas.
- Evita que preguntas sobre audiencia/defensa en sanciones de convivencia sean capturadas por FAQ genéricas de sanciones; se derivan a RAG/prudencia.
- Ajusta la clasificación de orientación prudente para consultas de oferta FP por centros, provincia, ASIR y curso actual.
- No modifica Qdrant, embeddings, colección, proveedor IA ni prompts normativos.
