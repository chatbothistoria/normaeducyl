# Informe postvalidación r6 mínima

**Versión:** `v073b_postvalidacion_r6_piloto_faq_finos`

## Motivo

La r6 se crea como ajuste mínimo tras el piloto interno r5, que finalizó con 29 PASS, 3 REVISAR y 0 ERROR. Los tres casos REVISAR eran incidencias menores de enrutamiento FAQ o prioridad entre FAQ verificadas:

1. `¿Cuándo disfrutan las vacaciones los docentes?` debía activar `vacaciones_docentes_agosto`.
2. `¿Qué hace el alumnado que no elige Religión en Primaria?` debía activar `primaria_atencion_educativa_no_religion`.
3. `¿La FP puede impartirse en modalidad virtual?` debía activar `fp_modalidades_presencial_semipresencial_virtual`.

## Alcance de cambios

Cambios aplicados:

- Añadidas prioridades exactas para las tres formulaciones detectadas.
- Añadidas reglas de intención cerradas para los tres casos.
- Añadidas variantes en `faq_normativa.json` para las tres FAQ afectadas.
- Actualizada la trazabilidad interna a `v073b_postvalidacion_r6_piloto_faq_finos`.

No se ha tocado:

- Qdrant.
- Prompt normativo principal.
- Proveedor IA.
- Umbrales globales.
- Filtros de seguridad o dominio.
- Estructura de la interfaz.

## Verificación local

Se ha comprobado que:

- `app.py` compila.
- Se mantienen 163 FAQ.
- Las tres preguntas corregidas enrutan a la FAQ esperada.
- Controles previos como `infantil_areas` y `primaria_religion_oferta_voluntaria` siguen funcionando.

## Estado

**r6 candidata para verificación corta y, si pasa, sustitución de r5 en piloto controlado.**
