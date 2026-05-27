# Cambios postvalidación r2 v073b

## Objetivo

Resolver los 7 casos `REVISAR` restantes detectados en la validación sintética rigurosa post-v073b r1.

## Diagnóstico

Los 7 casos restantes no eran fallos de IA, Qdrant ni documentación. Eran variantes exactas duplicadas en dos FAQ verificadas distintas. Una misma pregunta exacta no puede devolver dos `faq_id` simultáneamente, por lo que la auditoría marcaba `REVISAR` en la FAQ que perdía la prioridad.

## Cambios aplicados

- Se eliminan variantes duplicadas entre FAQ distintas, conservándolas en la FAQ más específica o preferida.
- Se alinea la tabla interna `_FAQ_VARIANTES_PRIORITARIAS_POSTVALIDACION` con esas decisiones.
- No se modifican umbrales globales.
- No se cambia la lógica RAG/Qdrant.
- No se cambia la capa IA salvo el refuerzo de citas ya introducido en r1.

## Decisiones de prioridad

- `consejo orientador eso` -> `eso_consejo_orientador`
- `modalidades bachillerato` -> `bachillerato_modalidades`
- `se puede repetir 1º/2º de bachillerato` -> `bachillerato_permanencia_cuatro_anos`
- `con cuántas materias se promociona de 1º a 2º de bachillerato` -> `bachillerato_promocion_dos_materias`
- `seguimiento alumnado empresa FP` -> `fp_tutor_dual_empresa`

## Alcance

Versión derivada quirúrgica:
`v073b_postvalidacion_r2_faq_dedupe_citas_docfix`

No se abre v074 porque no hay fallo funcional de arquitectura ni regresión técnica.
