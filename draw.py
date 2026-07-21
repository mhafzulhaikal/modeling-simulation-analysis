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
  fontFamily: 'Helvetica, sans-serif'
  themeVariables:
    fontSize: '14pt'
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
    S([Mulai]) --> A["`**Fase A**: Pemodelan dan analisis sistem`"]
    A --> V1{Kondisi tunak konsisten?}
    V1 -->|Tidak| A
    V1 -->|Ya| B["`**Fase B**: Perancangan dan implementasi sistem kendali`"]
    B --> C["`**Fase C**: Simulasi dan validasi`"]
    C --> V2{Respons tervalidasi?}
    V2 -->|Tidak| B
    V2 -->|Ya| D["`**Fase D**: Pengembangan aplikasi web`"]
    D --> E([Selesai])
""",
    'gambar_3-2.png': """flowchart TB
    S([Mulai]) --> A[Studi literatur dan penentuan studi kasus]
    A --> B[Penurunan neraca keadaan tak tunak dan tunak]
    B --> C[Penyusunan skrip Python model proses]
    C --> D[Penyelesaian kondisi tunak untuk titik operasi]
    D --> E[Linearisasi model dan penurunan fungsi alih]
    E --> F[Verifikasi konsistensi kondisi tunak]
    F --> G{Kondisi tunak konsisten?}
    G -->|Tidak| B
    G -->|Ya| H(["`Lanjut ke **Fase B**`"])
""",
    'gambar_3-3.png': """flowchart TB
    S(["`Dari **Fase A**`"]) --> A[Pemodelan elemen penyusun loop kendali]
    A --> B[Perangkaian elemen proses secara open-loop]
    B --> C[Pengujian step test]
    C --> D[/Data respons step test/]
    D --> E[Identifikasi model FOPDT]
    E --> F{Model FOPDT layak?}
    F -->|Tidak| C
    F -->|Ya| G[Pengaturan parameter pengendali]
    G --> H(["`Lanjut ke **Fase C**`"])
""",
    'gambar_3-4.png': """flowchart TB
    S(["`Dari **Fase B**`"]) --> A[Pemodelan Aspen HYSYS Dynamic]
    A --> B[Validasi fungsi tiap elemen kendali]
    B --> C[Perangkaian sistem closed-loop]
    C --> D[Simulasi skenario servo dan regulatory]
    D --> E[Analisis kinerja sistem kendali]
    E --> F[Validasi respons Python terhadap Aspen HYSYS Dynamic]
    F --> G{Respons tervalidasi?}
    G -->|Tidak| H(["`Kembali ke **Fase B**`"])
    G -->|Ya| I(["`Lanjut ke **Fase D**`"])
""",
    'gambar_3-5.png': """flowchart TB
    S(["`Dari **Fase C**`"]) --> A[Perancangan antarmuka pengguna]
    S --> B[Pengembangan mesin simulasi]
    A --> C[Integrasi aplikasi web NiceGUI]
    B --> C
    C --> D[Uji fungsional dan validasi keluaran]
    D --> E{Aplikasi diterima?}
    E -->|Tidak| C
    E -->|Ya| F[Deployment aplikasi]
    F --> G[Publikasi repositori open source]
    G --> H([Selesai])
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
