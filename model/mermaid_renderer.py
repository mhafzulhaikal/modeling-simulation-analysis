"""
mermaid_renderer.py
-------------------
Thin, optimal wrapper around the Mermaid CLI (mmdc).

Design principles (aligned with Mermaid CLI documentation):
- mmdc infers output format from the file extension → no --outputFormat flag needed.
- --scale controls Puppeteer zoom; effective DPI = 96 × scale (PNG only).
- Config JSON is passed flat (not nested under "mermaid") per mmdc --configFile spec.
- On Windows, node_modules/.bin/mmdc.cmd must be invoked via `cmd /c`.
- SVG is always lossless/vector — scale is irrelevant for SVG output.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DPI reference
# ---------------------------------------------------------------------------
# Chromium/Puppeteer base screen density assumed by mmdc.
_BASE_DPI: int = 96

# Default scale → 96 × 13 = 1248 dpi  (satisfies ≥ 1200 dpi requirement)
DEFAULT_SCALE: int = 13


# ---------------------------------------------------------------------------
# Locate mmdc binary
# ---------------------------------------------------------------------------


def _find_mmdc(project_root: Path) -> list[str]:
    """Return the subprocess command list for the Mermaid CLI.

    Priority (per Mermaid CLI docs – local install recommended over global):
    1. node_modules/.bin/mmdc.cmd  (Windows batch wrapper)
    2. node_modules/.bin/mmdc      (POSIX executable)
    3. npx --yes @mermaid-js/mermaid-cli  (fallback, downloads on first use)

    Notes
    -----
    On Windows the extensionless ``mmdc`` file is a POSIX shell script and
    cannot be executed directly by ``subprocess``.  The ``.cmd`` variant must
    be called via ``cmd /c`` to avoid *WinError 193*.
    """
    bin_dir = project_root / 'node_modules' / '.bin'
    cmd_file = bin_dir / 'mmdc.cmd'  # Windows
    posix_file = bin_dir / 'mmdc'  # Linux / macOS

    if cmd_file.exists():
        return ['cmd', '/c', str(cmd_file)]
    if posix_file.exists():
        return [str(posix_file)]
    return ['npx', '--yes', '@mermaid-js/mermaid-cli']


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------


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
    """Render Mermaid source to SVG or PNG.

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
        Applied to PNG and SVG; ignored for PDF.
    scale:
        Puppeteer scale factor (PNG/PDF only).
        Effective DPI = 96 × scale.  Default 13 → **1248 dpi**.
        Has no visual effect on SVG output.
    width / height:
        Puppeteer viewport size in pixels.  Larger values produce wider
        diagrams before auto-fitting.  Default: 800 × 600.
    mermaid_config:
        Additional Mermaid configuration options passed via ``--configFile``.
        Keys map directly to the Mermaid config schema
        (https://mermaid.js.org/config/schema-docs/config.html).
        Example: ``{'fontSize': 16, 'flowchart': {'curve': 'basis'}}``.
    project_root:
        Directory that contains ``node_modules/``.
        Defaults to the current working directory.
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
                'scale=%d yields only %d dpi (< 1200). Use scale >= 13 for high-quality output.',
                scale,
                effective_dpi,
            )
    else:
        logger.info('Rendering %s  (vector / lossless)', fmt.upper())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Write diagram source to temp .mmd file
        mmd_path = tmp_dir / 'diagram.mmd'
        mmd_path.write_text(source, encoding='utf-8')

        # Build mmdc command
        # mmdc infers format from extension → no --outputFormat needed
        cmd: list[str] = [
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

        # Write flat Mermaid config JSON (as documented by mmdc --configFile)
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
            raise RuntimeError(f'mmdc reported success but output file not found: {output}')

    size_kb = output.stat().st_size / 1024
    logger.info('Saved %s  (%.1f kB)', output, size_kb)
    return output
