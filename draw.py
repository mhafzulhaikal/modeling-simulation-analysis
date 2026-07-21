"""
draw.py
-------
Write standard Mermaid code here, then run:

    uv run python draw.py

Supported output formats (controlled by file extension in OUTPUT):
    .svg  – vector, lossless, infinite resolution  (best for reports / LaTeX)
    .png  – raster, high-DPI  (scale=13 -> 1248 dpi)
    .pdf  – PDF

DPI reference: effective_dpi = 96 × SCALE
    scale=13  -> 1248 dpi  ✓ (>= 1200 dpi)
    scale=10  ->  960 dpi
    scale=5   ->  480 dpi
"""

import logging

from model.mermaid_diagram import save_diagram

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# 1. DIAGRAM  – write standard Mermaid code here
# ─────────────────────────────────────────────────────────────────────────────

DIAGRAM = """
---
config:
  theme: redux
  layout: elk
  look: classic
  fontFamily: "'Source Code Pro Variable', monospace"
---
flowchart TB
    S([Mulai]) --> A[Fase A: Pemodelan dan Analisis Sistem]
    A --> V1{Model terverifikasi?}
    V1 -->|Tidak| A
    V1 -->|Ya| B[Fase B: Perancangan dan Implementasi Sistem Kendali]
    B --> C[Fase C: Simulasi dan Validasi]
    C --> V2{Respons tervalidasi?}
    V2 -->|Tidak| B
    V2 -->|Ya| D[Fase D: Pengembangan Aplikasi Web]
    D --> E([Selesai])
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. OUTPUT  – file path + extension sets the format
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT = 'outputs/diagrams/metodologi.svg'  # .svg | .png | .pdf
SCALE = 13  # PNG/PDF only: 96 × 13 = 1248 dpi

# ─────────────────────────────────────────────────────────────────────────────
# 3. RENDER OPTIONS  (optional)
# ─────────────────────────────────────────────────────────────────────────────

THEME = (
    'default'  # 'default' | 'forest' | 'dark' | 'neutral'  (redux is set in diagram frontmatter)
)
BACKGROUND = 'white'  # any CSS colour, or 'transparent'
WIDTH = 1200  # viewport width  (px)
HEIGHT = 600  # viewport height (px)

# Extra Mermaid config — see https://mermaid.js.org/config/schema-docs/config.html
# htmlLabels: false  →  uses native SVG <text> instead of <foreignObject>,
#                        which makes the SVG renderable in Microsoft Word.
MERMAID_CONFIG: dict | None = {
    'htmlLabels': False,
    # In htmlLabels: false, edge labels have opacity: 0.5 and often use
    # negative HSL values (e.g. hsl(-120, 0%, 80%)) which MS Word cannot parse,
    # causing it to render as a solid black box.
    # We override it to 1.0 opacity and a safe Hex color (#e5e5e5) to fix this.
    'themeCSS': '.edgeLabel rect { opacity: 1.0 !important; fill: #e5e5e5 !important; }',
}

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    path = save_diagram(
        DIAGRAM,
        OUTPUT,
        scale=SCALE,
        theme=THEME,
        background=BACKGROUND,
        width=WIDTH,
        height=HEIGHT,
        mermaid_config=MERMAID_CONFIG,
    )
    print(f'Saved -> {path}')
