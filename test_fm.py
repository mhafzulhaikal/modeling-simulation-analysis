from model.mermaid_diagram import save_diagram

diag = """---
config:
  theme: redux
  layout: elk
  look: classic
  fontFamily: "'Source Code Pro Variable', monospace"
  htmlLabels: false
  themeCSS: 'svg { border: 2px solid #cccccc; border-radius: 8px; } .edgeLabel rect { opacity: 1.0 !important; fill: #e5e5e5 !important; }'
---
flowchart TB
S([Mulai]) --> A[Fase A: Pemodelan dan Analisis Sistem]
"""
save_diagram(diag, 'outputs/diagrams/test_frontmatter.svg', scale=1)
