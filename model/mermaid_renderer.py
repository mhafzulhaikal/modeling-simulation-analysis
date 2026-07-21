"""
mermaid_renderer.py
-------------------
Thin wrapper around the Mermaid CLI (mmdc).

Design principles (aligned with Mermaid CLI documentation):
- mmdc infers output format from the file extension → no --outputFormat flag needed.
- --scale controls Puppeteer zoom; effective DPI = 96 × scale (PNG only).
- Config JSON is passed flat (not nested under "mermaid") per mmdc --configFile spec.
- On Windows, node_modules/.bin/mmdc.cmd must be invoked via `cmd /c`.
- SVG is always lossless/vector — scale is irrelevant for SVG output.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_DPI: int = 96
DEFAULT_SCALE: int = 13  # 96 × 13 = 1248 dpi  (≥ 1200 dpi requirement)

_GENERIC_FONTS = frozenset(
    {'serif', 'sans-serif', 'monospace', 'inherit', 'cursive', 'fantasy', 'system-ui'}
)


def _find_mmdc(project_root: Path) -> list[str]:
    """Return the subprocess command list for the Mermaid CLI.

    Priority (per Mermaid CLI docs – local install recommended over global):

    1. ``node_modules/.bin/mmdc.cmd``  (Windows batch wrapper)
    2. ``node_modules/.bin/mmdc``      (POSIX executable)
    3. ``npx --yes @mermaid-js/mermaid-cli``  (fallback, downloads on first use)

    Notes
    -----
    On Windows the extensionless ``mmdc`` file is a POSIX shell script that
    cannot be executed directly by ``subprocess``.  The ``.cmd`` variant must
    be called via ``cmd /c`` to avoid *WinError 193*.
    """
    bin_dir = project_root / 'node_modules' / '.bin'
    cmd_file = bin_dir / 'mmdc.cmd'
    posix_file = bin_dir / 'mmdc'

    if cmd_file.exists():
        return ['cmd', '/c', str(cmd_file)]
    if posix_file.exists():
        return [str(posix_file)]
    return ['npx', '--yes', '@mermaid-js/mermaid-cli']


def render(
    source: str,
    output: str | Path,
    *,
    theme: str = 'default',
    background: str = 'white',
    scale: int = DEFAULT_SCALE,
    width: int = 800,
    height: int = 600,
    mermaid_config: dict | None = None,
    project_root: str | Path | None = None,
    quiet: bool = True,
) -> Path:
    """Render Mermaid source to SVG, PNG, or PDF.

    The output format is inferred from the *output* file extension, which is
    the approach recommended by the Mermaid CLI documentation.

    Parameters
    ----------
    source:
        Raw Mermaid diagram text (not a file path).
    output:
        Destination file path.  Extension determines format:
        ``.svg`` → vector SVG (lossless, resolution-independent).
        ``.png`` → raster PNG at ``scale × 96`` dpi.
        ``.pdf`` → PDF.
    theme:
        Mermaid built-in theme: ``'default'``, ``'forest'``, ``'dark'``,
        ``'neutral'``.
    background:
        Background colour (CSS string or ``'transparent'``).
    scale:
        Puppeteer scale factor (PNG/PDF only).
        Effective DPI = 96 × scale.  Default 13 → **1248 dpi**.
        Has no visual effect on SVG output.
    width / height:
        Puppeteer viewport size in pixels.  Default: 800 × 600.
    mermaid_config:
        Mermaid configuration dict passed via ``--configFile``.
        Keys map directly to the Mermaid config schema:
        https://mermaid.js.org/config/schema-docs/config.html

        **Word/LibreOffice SVG fix**: ``{'htmlLabels': False}`` forces native
        SVG ``<text>`` elements instead of ``<foreignObject>`` HTML embeds —
        the officially documented ``htmlLabels`` option.
    project_root:
        Directory containing ``node_modules/``.  Defaults to ``Path.cwd()``.
    quiet:
        Suppress mmdc log output (``--quiet`` flag).

    Returns
    -------
    Path
        Resolved path to the saved output file.

    Raises
    ------
    RuntimeError
        If mmdc exits with a non-zero code or the output file is missing.
    FileNotFoundError
        If Node.js / mmdc cannot be located.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(project_root) if project_root else Path.cwd()

    fmt = output.suffix.lstrip('.').lower()
    effective_dpi = _BASE_DPI * scale
    if fmt == 'png':
        logger.info('Rendering PNG  scale=%d  effective-DPI=%d', scale, effective_dpi)
        if effective_dpi < 1200:
            logger.warning(
                'scale=%d yields only %d dpi (< 1200). Use scale >= 13.',
                scale,
                effective_dpi,
            )
    else:
        logger.info('Rendering %s  (vector / lossless)', fmt.upper())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        mmd_path = tmp_dir / 'diagram.mmd'
        mmd_path.write_text(source, encoding='utf-8')

        cmd = [
            *_find_mmdc(root),
            '--input',
            str(mmd_path),
            '--output',
            str(output),
            '--theme',
            theme,
            '--backgroundColor',
            background,
            '--scale',
            str(scale),
            '--width',
            str(width),
            '--height',
            str(height),
        ]

        if mermaid_config:
            cfg_path = tmp_dir / 'config.json'
            cfg_path.write_text(json.dumps(mermaid_config), encoding='utf-8')
            cmd += ['--configFile', str(cfg_path)]

        if quiet:
            cmd.append('--quiet')

        logger.debug('Command: %s', ' '.join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(root),
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                'Mermaid CLI not found.  Install it with:\n'
                '  npm install --save-dev @mermaid-js/mermaid-cli'
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('mmdc timed out after 120 s.') from exc

        if result.returncode != 0:
            msg = (result.stderr or result.stdout).strip()
            raise RuntimeError(f'mmdc failed (exit {result.returncode}):\n{msg}')

        if not output.exists():
            raise RuntimeError(f'mmdc reported success but output not found: {output}')

    size_kb = output.stat().st_size / 1024
    logger.info('Saved %s  (%.1f kB)', output, size_kb)
    return output


def _find_font_file(family: str) -> Path | None:
    """Return the first ``.ttf`` / ``.otf`` file matching *family* on disk.

    Searches Windows and common Linux/macOS font directories.
    Matching is case-insensitive with spaces, hyphens and underscores ignored.
    """

    def normalise(s: str) -> str:
        return re.sub(r'[\s\-_]', '', s).lower()

    font_dirs = [
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts',
        Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft' / 'Windows' / 'Fonts',
        Path('/usr/share/fonts'),
        Path('/usr/local/share/fonts'),
        Path.home() / '.fonts',
        Path('/System/Library/Fonts'),
    ]
    needle = normalise(family)

    return next(
        (
            f
            for d in font_dirs
            if d.exists()
            for ext in ('*.ttf', '*.otf', '*.TTF', '*.OTF')
            for f in d.rglob(ext)
            if needle in normalise(f.stem)
        ),
        None,
    )


def embed_fonts_svg(svg_path: str | Path, output: str | Path | None = None) -> Path:
    """Embed all referenced fonts into an SVG as base64 ``@font-face`` blocks.

    Makes the SVG fully self-contained for Microsoft Word / LibreOffice without
    needing fonts installed on the host system.

    .. note::
        If the SVG contains ``<foreignObject>`` elements (Mermaid's default
        HTML-in-SVG text), Word will still not render text — font embedding
        cannot fix that.  Use ``{'htmlLabels': False}`` in *mermaid_config*
        when rendering, or switch to PNG output.

    Parameters
    ----------
    svg_path:
        Path to the source ``.svg`` file produced by :func:`render`.
    output:
        Destination path.  Defaults to ``<stem>_embedded.svg``.

    Returns
    -------
    Path
        Path to the font-embedded SVG.

    Examples
    --------
    >>> svg = render(source, 'outputs/diagram.svg')
    >>> word_svg = embed_fonts_svg(svg)            # diagram_embedded.svg
    >>> word_svg = embed_fonts_svg(svg, 'out.svg') # custom destination
    """
    svg_path = Path(svg_path)
    if not svg_path.exists():
        raise FileNotFoundError(svg_path)

    output = Path(output) if output else svg_path.with_stem(svg_path.stem + '_embedded')
    output.parent.mkdir(parents=True, exist_ok=True)

    content = svg_path.read_text(encoding='utf-8')

    if '<foreignObject' in content:
        logger.warning(
            'embed_fonts_svg: SVG contains <foreignObject> — Word cannot render it. '
            "Set mermaid_config={'htmlLabels': False} when rendering instead."
        )

    families = {
        name
        for m in re.finditer(r"font-family\s*:\s*([^;\"'}]+)", content)
        for part in m.group(1).split(',')
        if (name := part.strip().strip('\'"')) and name.lower() not in _GENERIC_FONTS
    }

    if not families:
        logger.info('embed_fonts_svg: no embeddable fonts found — copying as-is.')
        shutil.copy2(svg_path, output)
        return output

    font_face_blocks, missing = [], []

    for family in sorted(families):
        font_file = _find_font_file(family)
        if font_file is None:
            missing.append(family)
            logger.warning('embed_fonts_svg: font not found on disk: %r', family)
            continue

        fmt = 'opentype' if font_file.suffix.lower() == '.otf' else 'truetype'
        b64 = base64.b64encode(font_file.read_bytes()).decode('ascii')
        font_face_blocks.append(
            f'@font-face {{\n'
            f"  font-family: '{family}';\n"
            f"  src: url('data:font/{fmt};base64,{b64}') format('{fmt}');\n"
            f'}}'
        )
        logger.info(
            'embed_fonts_svg: embedded %r  (%s, %.0f kB)',
            family,
            font_file.name,
            font_file.stat().st_size / 1024,
        )

    if missing:
        logger.warning(
            'embed_fonts_svg: %d font(s) not embedded: %s',
            len(missing),
            ', '.join(repr(m) for m in missing),
        )

    style_block = '\n'.join(font_face_blocks)
    if '<style' in content:
        content = re.sub(r'(<style[^>]*>)', r'\1\n' + style_block, content, count=1)
    else:
        content = content.replace(
            '<svg',
            f'<svg><defs><style>{style_block}</style></defs\n<svg',
            1,
        )

    output.write_text(content, encoding='utf-8')
    logger.info('embed_fonts_svg: saved %s  (%.1f kB)', output, output.stat().st_size / 1024)
    return output


def add_svg_border(
    svg_path: str | Path,
    color: str = '#e5e5e5',
    width: float = 2.0,
    radius: float = 8.0,
) -> Path:
    """Draw a bounding box (outline) around the entire SVG diagram.

    This injects a `<rect>` element tightly fitted to the SVG's `viewBox`
    so the border is never cut off, even when scaled.

    Parameters
    ----------
    svg_path:
        Path to the source ``.svg`` file.
    color:
        Border stroke color (CSS string).
    width:
        Border line thickness.
    radius:
        Border corner roundness (rx/ry).

    Returns
    -------
    Path
        The path to the modified SVG file.
    """
    path = Path(svg_path)
    content = path.read_text(encoding='utf-8')

    # Find the SVG viewBox to fit the border perfectly
    m = re.search(r'viewBox="([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)"', content)
    if not m:
        logger.warning('add_svg_border: no viewBox found in %s', path.name)
        return path

    x, y, w, h = map(float, m.groups())

    # Inset by half stroke-width so the border doesn't clip outside viewBox
    rx = x + width / 2
    ry = y + width / 2
    rw = w - width
    rh = h - width

    rect = (
        f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" '
        f'fill="none" stroke="{color}" stroke-width="{width}" rx="{radius}"/>'
    )

    # Inject the rect right after the opening <svg> tag
    content = re.sub(r'(<svg[^>]*>)', r'\1\n' + rect, content, count=1)
    path.write_text(content, encoding='utf-8')

    logger.info('add_svg_border: added %s border to %s', color, path.name)
    return path
