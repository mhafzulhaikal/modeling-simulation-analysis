import logging
import os

from model.mermaid_diagram import save_diagram

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL CONFIGURATION (Applied to all diagrams)
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CONFIG = """---
config:
  theme: redux
  fontFamily: '''Source Code Pro Variable'', monospace'
  themeVariables:
    fontSize: '16pt'
    primaryColor: '#ffffff'
    primaryTextColor: 'black'
    lineColor: 'black'
    textColor: 'black'
  layout: elk
  look: classic
  htmlLabels: False
  themeCSS: '.edgeLabel rect { opacity: 1.0 !important; fill: #e5e5e5 !important; }'
---
"""

# ─────────────────────────────────────────────────────────────────────────────
# 2. DIAGRAMS  – write standard Mermaid code here
# ─────────────────────────────────────────────────────────────────────────────

# Define multiple diagrams as a dictionary: { "filename": "mermaid_code" }
DIAGRAMS = {
    'gambar_3-1.png': """flowchart TB
    S([Mulai]) --> A["`**Fase A**: Pemodelan dan Analisis Sistem`"]
    A --> V1{Model terverifikasi?}
    V1 -->|Tidak| A
    V1 -->|Ya| B["`**Fase B**: Perancangan dan Implementasi Sistem Kendali`"]
    B --> C["`**Fase C**: Simulasi dan Validasi`"]
    C --> V2{Respons tervalidasi?}
    V2 -->|Tidak| B
    V2 -->|Ya| D["`**Fase D**: Pengembangan Aplikasi Web`"]
    D --> E([Selesai])
""",
    'gambar_3-2.png': """flowchart TB
    A[Penetapan volume kendali] --> B[Penyusunan neraca massa total]
    B --> C[Penyusunan neraca mol tiap komponen]
    C --> D[Penyusunan neraca energi reaktor dan pendingin]
    D --> E[Sistem ODE non-linear]
    E --> F{"Konsistensi dimensi/satuan dan derajat kebebasan terpenuhi?"}
    F -->|Tidak| E
    F -->|Ya| G[Penulisan skrip model proses Python]
""",
    'gambar_3-3.png': """flowchart TB
    A[Perumusan model] --> B[Pemeriksaan neraca massa, mol, energi pada kondisi tunak]
    B --> C{Residu di bawah toleransi?}
    C -->|Tidak| B
    C -->|Ya| D[Model terverifikasi]
""",
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. OUTPUT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = 'outputs/diagrams/'
SCALE = 13  # PNG/PDF only: 96 × 13 = 1248 dpi

THEME = 'default'
BACKGROUND = 'white'
WIDTH = 1200
HEIGHT = 600

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename, source in DIAGRAMS.items():
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Inject the global configuration at the top of each diagram!
        full_source = GLOBAL_CONFIG + source

        path = save_diagram(
            full_source,
            output_path,
            scale=SCALE,
            theme=THEME,
            background=BACKGROUND,
            width=WIDTH,
            height=HEIGHT,
        )
        print(f'Saved -> {path}')
