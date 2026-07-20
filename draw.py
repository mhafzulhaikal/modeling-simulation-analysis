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
flowchart LR
    SP([Setpoint r&#40;t&#41;])  --> SUM((+/-))
    SUM --> PID[[PID Controller]]
    PID --> PLANT[Plant G&#40;s&#41;]
    PLANT --> Y([Output y&#40;t&#41;])
    Y --> SENS[Sensor H&#40;s&#41;]
    SENS -->|feedback| SUM
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. OUTPUT  – file path + extension sets the format
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT = 'outputs/diagrams/pid_loop.svg'  # .svg | .png | .pdf
SCALE = 13  # PNG/PDF only: 96 × 13 = 1248 dpi

# ─────────────────────────────────────────────────────────────────────────────
# 3. RENDER OPTIONS  (optional)
# ─────────────────────────────────────────────────────────────────────────────

THEME = 'default'  # 'default' | 'forest' | 'dark' | 'neutral'
BACKGROUND = 'white'  # any CSS colour, or 'transparent'
WIDTH = 1200  # viewport width  (px)
HEIGHT = 600  # viewport height (px)

# Extra Mermaid config — see https://mermaid.js.org/config/schema-docs/config.html
# Leave as None to use Mermaid defaults.
MERMAID_CONFIG: dict | None = None
# Example:
# MERMAID_CONFIG = {'flowchart': {'curve': 'basis'}, 'fontSize': 16}

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
