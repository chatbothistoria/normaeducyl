# Informe postvalidación r5 mínima

Versión: `v073b_postvalidacion_r5_robustez_faq_429`

## Motivo

La batería de 25 preguntas reales sobre r4 produjo 19 PASS, 5 REVISAR y 1 ERROR. El ERROR correspondía a saturación temporal del proveedor IA (`429 queue_exceeded`), y los REVISAR fueron mayoritariamente rutas FAQ mejorables o solapamientos de FAQ equivalentes.

## Cambios aplicados

1. Refuerzo FAQ para evitar RAG/IA en preguntas frecuentes claras:
   - permiso por hospitalización de padre/madre,
   - áreas de Educación Infantil,
   - siglas de calificaciones IN/SU/BI/NT/SB.
2. Priorización/alias FAQ en Bachillerato:
   - modalidades de Bachillerato,
   - promoción de 1.º a 2.º con dos materias no superadas.
3. Robustez ante error temporal de IA:
   - se conserva el reintento ante 429/servicio temporal,
   - si el proveedor sigue sin aceptar la petición, se muestra respuesta segura sin inventar contenido normativo y con ruta `RAG_IA_NO_DISPONIBLE`.

## Cambios no realizados

- No se modifica Qdrant.
- No se modifica la colección vectorial.
- No se modifican umbrales globales.
- No se cambian prompts normativos principales.
- No se abre v074.

## Validación recomendada

Repetir solo los 6 casos problemáticos de la batería r4 más dos controles de seguridad/prudencia.
