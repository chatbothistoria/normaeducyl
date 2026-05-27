# Informe postvalidación r7 — filtros, FAQ y prudencia

**Versión:** `v073b_postvalidacion_r7_filtros_faq_prudencia`

## Origen

El piloto exhaustivo r6 ejecutó 202 casos con Qdrant real e IA real. El resultado fue 197 PASS, 5 REVISAR y 0 ERROR. La r7 corrige exclusivamente esos 5 puntos, sin abrir v074.

## Cambios aplicados

1. **Filtro de dominio fiscal/autónomos**
   - Preguntas como `¿Cuánto tengo que pagar de IRPF como autónomo?` se bloquean antes de Qdrant/IA mediante `FILTRO_DOMINIO`.

2. **Variante abreviada de Bachillerato**
   - `modalidades bach` y `modalidades bachiller` activan `bachillerato_modalidades`.

3. **Solapamiento de modalidades de Bachillerato**
   - `qué modalidades de Bachillerato existen` se deja asociada a `bachillerato_modalidades_basica`.
   - `cuáles son las modalidades de Bachillerato` se mantiene asociada a `bachillerato_modalidades`.

4. **Garantías/audiencia en sanciones de convivencia**
   - Preguntas sobre sanción sin oír al alumno/familia, audiencia, defensa, alegaciones o garantías procedimentales no son capturadas por FAQ genéricas de sanciones.
   - Esas preguntas se derivan a RAG/prudencia para evitar respuestas incompletas.

5. **Orientación prudente de oferta FP**
   - Consultas tipo `qué centros de Burgos ofertan ASIR este curso` se clasifican como `oferta_fp`.

## Alcance no modificado

- No se modifica la colección Qdrant.
- No se modifican embeddings.
- No se modifica proveedor IA.
- No se modifican prompts normativos.
- No se cambia la interfaz ni la estructura general.
- Se mantienen 163 FAQ.

## Verificación local previa

Se comprobó que:

- `app.py` compila.
- Se cargan 163 FAQ.
- La versión interna es `v073b_postvalidacion_r7_filtros_faq_prudencia`.
- Los cinco casos problemáticos enrutan según lo esperado.
