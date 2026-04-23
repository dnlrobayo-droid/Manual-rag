# Assets

Esta carpeta contiene recursos visuales del proyecto.

## Archivos

- **`architecture.svg`** — Diagrama de arquitectura del sistema RAG (formato vectorial).
- **`architecture.png`** *(opcional)* — Versión PNG del diagrama, si prefieres incrustarlo en Markdown con mejor compatibilidad en GitHub. Puedes generarla desde el SVG con:
  ```bash
  # Usando rsvg-convert (paquete librsvg)
  rsvg-convert -w 1200 assets/architecture.svg -o assets/architecture.png

  # o usando Inkscape
  inkscape assets/architecture.svg --export-type=png --export-filename=assets/architecture.png
  ```
- **`demo.gif`** *(pendiente)* — Grabación de pantalla de la app en funcionamiento.
  Herramientas recomendadas:
  - [Peek](https://github.com/phw/peek) (Linux)
  - [Kap](https://getkap.co/) (macOS)
  - [ScreenToGif](https://www.screentogif.com/) (Windows)

  Flujo sugerido:
  1. Arranca la app: `streamlit run app/main.py`
  2. Graba una consulta completa (ej. "¿qué significa E05?") → respuesta → expandir fragmentos.
  3. Exporta en ≤10 MB para que GitHub lo renderice inline en el README.
