# Cambios post-validación v073b

Base: `v073b_evaluacion_objetiva_routing_fix`.

Nueva etiqueta de código: `v073b_postvalidacion_faq_citas_docfix`.

## Cambios aplicados

1. **Documentación**
   - `README.md` pasa de 130 FAQ a 163 FAQ.

2. **Solapamientos FAQ**
   - Se añade una tabla de prioridad exacta para variantes duplicadas detectadas en la validación sintética:
     - `consejo orientador eso` → `eso_documentos_consejo_diagnostico`
     - `modalidades bachillerato` → `bachillerato_modalidades_basica`
     - variantes de promoción/repetición de Bachillerato → `bachillerato_repeticion_promocion_basica`
     - `seguimiento alumnado empresa fp` → `fp_tutor_empresa_seguimiento_contacto`
   - No se rebajan umbrales ni se amplía el matching difuso.

3. **Citas IA**
   - `validar_citas_fragmentos` ahora puede validar contra los identificadores `[F#]` realmente presentes en el contexto final enviado al LLM.
   - Esto evita aceptar citas a fragmentos que estaban en la lista original pero pudieron quedar fuera por recorte de contexto.

## No se ha cambiado

- Qdrant.
- Embeddings.
- Colección `normativa`.
- Prompt jurídico salvo validación posterior.
- Filtro de dominio.
- Límites de sesión.
- Privacidad.
- Estructura general de la interfaz.
