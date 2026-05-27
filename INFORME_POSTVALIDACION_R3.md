# Informe postvalidación r3 — NormaEdu 2 v073b

## Motivo

La validación r2 eliminó los casos `REVISAR`, pero aparecieron 2 `ERROR` en la capa IA real. La causa no era FAQ ni Qdrant: el campo `IA_API_URL` recibió un valor sin esquema `https://`, aparentemente una clave/token pegada en el campo URL.

## Cambios aplicados

- Se mantiene la base v073b y las 163 FAQ.
- No se cambia el comportamiento FAQ.
- No se cambia Qdrant.
- No se cambia el prompt ni los umbrales.
- Se añade validación de configuración:
  - `IA_API_URL` debe empezar por `https://` o `http://`.
  - `QDRANT_URL` debe empezar por `https://` o `http://`.
- Se evita que una URL mal formada provoque una excepción cruda tipo `MissingSchema`.
- Se ocultan detalles sensibles en errores de configuración.

## Versión interna

`v073b_postvalidacion_r3_config_guard_faq_dedupe_citas_docfix`

## Cómo probar

Usar el cuaderno de una sola celda r2 y este ZIP. Para prueba fuerte:

- Qdrant real: S
- IA real: S
- `IA_API_URL`: endpoint completo del proveedor, por ejemplo `https://.../v1/chat/completions`
- `IA_MODEL`: modelo
- `IA_API_KEY`: clave/token
