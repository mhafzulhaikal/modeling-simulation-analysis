"""
mermaid_diagram.py
------------------
Lightweight Mermaid diagram builders.

Each class assembles valid Mermaid source through a fluent API and delegates
rendering to :func:`model.mermaid_renderer.render`.

Supported diagram types
-----------------------
* :class:`Flowchart`  – flowchart / graph
* :class:`Sequence`   – sequenceDiagram
* :class:`Class`      – classDiagram
* :class:`State`      – stateDiagram-v2
* :class:`Gantt`      – gantt
"""

from __future__ import annotations

import textwrap
from abc import ABC, abstractmethod
from pathlib import Path

from model.mermaid_renderer import DEFAULT_SCALE, render

__all__ = ['Flowchart', 'Sequence', 'Class', 'State', 'Gantt', 'save_diagram']


class _Diagram(ABC):
    """Shared rendering interface for all diagram types."""

    @abstractmethod
    def source(self) -> str:
        """Return the complete Mermaid source string."""

    def save(
        self,
        output: str | Path,
        *,
        scale: int = DEFAULT_SCALE,
        theme: str = 'default',
        background: str = 'white',
        width: int = 800,
        height: int = 600,
        mermaid_config: dict | None = None,
    ) -> Path:
        """Render and save to *output*.

        The output format (SVG / PNG / PDF) is inferred from the file
        extension — the recommended approach per the Mermaid CLI docs.

        Parameters
        ----------
        output:
            Destination path, e.g. ``'outputs/diagrams/my_diagram.svg'``.
        scale:
            PNG/PDF Puppeteer scale factor.  Effective DPI = 96 × scale.
            Default 13 → 1248 dpi.  Ignored for SVG.
        theme:
            ``'default'``, ``'forest'``, ``'dark'``, or ``'neutral'``.
        background:
            CSS colour string, e.g. ``'white'``, ``'transparent'``.
        width / height:
            Puppeteer viewport dimensions (pixels).
        mermaid_config:
            Extra Mermaid config dict, e.g. ``{'htmlLabels': False}``.

        Returns
        -------
        Path
            Path to the saved file.
        """
        return render(
            self.source(),
            output,
            scale=scale,
            theme=theme,
            background=background,
            width=width,
            height=height,
            mermaid_config=mermaid_config,
        )


class Flowchart(_Diagram):
    """Mermaid flowchart builder.

    Parameters
    ----------
    direction:
        ``'LR'`` (left→right), ``'TD'`` / ``'TB'`` (top→bottom),
        ``'RL'``, ``'BT'``.
    title:
        Optional diagram title (rendered as YAML frontmatter).

    Examples
    --------
    >>> d = Flowchart('LR', title='PID Loop')
    >>> d.node('SP', 'Setpoint').node('PID', 'Controller', shape='subroutine')
    >>> d.edge('SP', 'PID', 'r(t)').save('outputs/diagrams/pid.svg')
    """

    _SHAPES: dict[str, tuple[str, str]] = {
        'rect': ('[', ']'),
        'round': ('(', ')'),
        'stadium': ('([', '])'),
        'subroutine': ('[[', ']]'),
        'cylinder': ('[(', ')]'),
        'circle': ('((', '))'),
        'diamond': ('{', '}'),
        'hexagon': ('{{', '}}'),
        'parallelogram': ('[/', '/]'),
    }

    def __init__(self, direction: str = 'LR', *, title: str = '') -> None:
        self.direction = direction
        self.title = title
        self._lines: list[str] = []

    def node(
        self, id: str, label: str | None = None, shape: str = 'rect', cls: str | None = None
    ) -> Flowchart:
        """Add a node."""
        lbl = label if label is not None else id
        o, c = self._SHAPES.get(shape, ('[', ']'))
        line = f'    {id}{o}"{lbl}"{c}'
        if cls:
            line += f':::{cls}'
        self._lines.append(line)
        return self

    def edge(self, src: str, dst: str, label: str = '', arrow: str = '-->') -> Flowchart:
        """Add a directed edge."""
        mid = f'|"{label}"| ' if label else ' '
        self._lines.append(f'    {src} {arrow}{mid}{dst}')
        return self

    def style(self, id: str, css: str) -> Flowchart:
        """Apply inline CSS to a node."""
        self._lines.append(f'    style {id} {css}')
        return self

    def classdef(self, name: str, css: str) -> Flowchart:
        """Define a reusable CSS class."""
        self._lines.append(f'    classDef {name} {css}')
        return self

    def subgraph(self, id: str, label: str, node_ids: list[str]) -> Flowchart:
        """Group nodes in a labelled subgraph box."""
        self._lines += [
            f'    subgraph {id}["{label}"]',
            *(f'        {n}' for n in node_ids),
            '    end',
        ]
        return self

    def raw(self, line: str) -> Flowchart:
        """Append a raw Mermaid line (escape hatch)."""
        self._lines.append(f'    {line}')
        return self

    def source(self) -> str:
        header = ['---', f'title: {self.title}', '---'] if self.title else []
        return '\n'.join([*header, f'flowchart {self.direction}', *self._lines])


class Sequence(_Diagram):
    """Mermaid sequenceDiagram builder.

    Examples
    --------
    >>> d = Sequence(title='Control Handshake', autonumber=True)
    >>> d.participant('Sensor', 'S').participant('Controller', 'C')
    >>> d.message('S', 'C', 'y(t)').save('outputs/diagrams/handshake.png')
    """

    def __init__(self, *, title: str = '', autonumber: bool = False) -> None:
        self.title = title
        self.autonumber = autonumber
        self._lines: list[str] = []

    def participant(self, name: str, alias: str | None = None, *, actor: bool = False) -> Sequence:
        """Declare a participant or actor."""
        kind = 'actor' if actor else 'participant'
        alias_part = f' as {alias}' if alias and alias != name else ''
        self._lines.append(f'    {kind} {name}{alias_part}')
        return self

    def message(self, src: str, dst: str, text: str, arrow: str = '->>') -> Sequence:
        """Add a message arrow."""
        self._lines.append(f'    {src}{arrow}{dst}: {text}')
        return self

    def note(self, pos: str, participant: str, text: str) -> Sequence:
        """Add a note (pos: ``'left of'``, ``'right of'``, ``'over'``)."""
        self._lines.append(f'    Note {pos} {participant}: {text}')
        return self

    def raw(self, line: str) -> Sequence:
        """Append a raw line."""
        self._lines.append(f'    {line}')
        return self

    def source(self) -> str:
        header = ['---', f'title: {self.title}', '---'] if self.title else []
        autonumber = ['    autonumber'] if self.autonumber else []
        return '\n'.join([*header, 'sequenceDiagram', *autonumber, *self._lines])


class Class(_Diagram):
    """Mermaid classDiagram builder.

    Examples
    --------
    >>> d = Class(title='Control Classes')
    >>> d.cls('PID', attrs=['Kp: float'], methods=['compute(e)'])
    >>> d.relation('PID', 'Plant', '-->', 'controls').save('outputs/diagrams/classes.svg')
    """

    def __init__(self, *, title: str = '') -> None:
        self.title = title
        self._lines: list[str] = []

    def cls(
        self, name: str, attrs: list[str] | None = None, methods: list[str] | None = None
    ) -> Class:
        """Add a class definition."""
        self._lines += [
            f'    class {name} {{',
            *(f'        +{a}' for a in (attrs or [])),
            *(f'        +{m}' for m in (methods or [])),
            '    }',
        ]
        return self

    def relation(self, src: str, dst: str, arrow: str = '-->', label: str = '') -> Class:
        """Add a relationship line."""
        lbl = f' : {label}' if label else ''
        self._lines.append(f'    {src} {arrow} {dst}{lbl}')
        return self

    def raw(self, line: str) -> Class:
        self._lines.append(f'    {line}')
        return self

    def source(self) -> str:
        header = ['---', f'title: {self.title}', '---'] if self.title else []
        return '\n'.join([*header, 'classDiagram', *self._lines])


class State(_Diagram):
    """Mermaid stateDiagram-v2 builder.

    Examples
    --------
    >>> d = State(title='PID States')
    >>> d.state('Idle').state('Running')
    >>> d.transition('[*]', 'Idle').transition('Idle', 'Running', 'SP changed')
    >>> d.save('outputs/diagrams/states.svg')
    """

    def __init__(self, *, title: str = '') -> None:
        self.title = title
        self._lines: list[str] = []

    def state(self, id: str, label: str = '') -> State:
        """Declare a state."""
        line = f'    state "{label}" as {id}' if label else f'    {id}'
        self._lines.append(line)
        return self

    def transition(self, src: str, dst: str, label: str = '') -> State:
        """Add a transition."""
        lbl = f' : {label}' if label else ''
        self._lines.append(f'    {src} --> {dst}{lbl}')
        return self

    def raw(self, line: str) -> State:
        self._lines.append(f'    {line}')
        return self

    def source(self) -> str:
        header = ['---', f'title: {self.title}', '---'] if self.title else []
        return '\n'.join([*header, 'stateDiagram-v2', *self._lines])


class Gantt(_Diagram):
    """Mermaid gantt builder.

    Examples
    --------
    >>> d = Gantt('Project Plan', date_format='YYYY-MM-DD')
    >>> d.section('Phase 1')
    >>> d.task('Model FOPDT', '2025-01-01', '7d')
    >>> d.task('Fit params',  '2025-01-08', '5d', status='done')
    >>> d.save('outputs/diagrams/plan.svg')
    """

    def __init__(
        self, title: str = '', *, date_format: str = 'YYYY-MM-DD', axis_format: str = '%b %d'
    ) -> None:
        self.title = title
        self.date_format = date_format
        self.axis_format = axis_format
        self._lines: list[str] = []

    def section(self, name: str) -> Gantt:
        """Start a new section."""
        self._lines.append(f'    section {name}')
        return self

    def task(
        self, name: str, start: str, duration: str, *, id: str = '', status: str = ''
    ) -> Gantt:
        """Add a task to the current section."""
        meta = ', '.join(filter(None, [status, id]))
        meta_str = f'    {meta},' if meta else '   '
        self._lines.append(f'    {name}    :{meta_str} {start}, {duration}')
        return self

    def raw(self, line: str) -> Gantt:
        self._lines.append(f'    {line}')
        return self

    def source(self) -> str:
        return '\n'.join(
            [
                'gantt',
                *([f'    title {self.title}'] if self.title else []),
                f'    dateFormat  {self.date_format}',
                f'    axisFormat  {self.axis_format}',
                *self._lines,
            ]
        )


def save_diagram(
    source: str,
    output: str | Path,
    *,
    scale: int = DEFAULT_SCALE,
    theme: str = 'default',
    background: str = 'white',
    width: int = 800,
    height: int = 600,
    mermaid_config: dict | None = None,
) -> Path:
    """Render raw Mermaid *source* text directly to *output*.

    A thin wrapper around :func:`~model.mermaid_renderer.render` for when you
    already have the Mermaid source string and don't need a builder class.

    Parameters
    ----------
    source:
        Complete Mermaid diagram text (leading indentation is stripped).
    output:
        Destination path — extension sets format: ``.svg``, ``.png``, ``.pdf``.
    scale:
        PNG/PDF scale factor.  Default 13 → 1248 dpi.
    theme / background / width / height / mermaid_config:
        Forwarded to :func:`~model.mermaid_renderer.render`.

    Returns
    -------
    Path
        Path to the saved file.

    Examples
    --------
    >>> save_diagram(
    ...     '''
    ...     pie title Energy losses
    ...         "Copper" : 48
    ...         "Iron"   : 32
    ...         "Other"  : 20
    ...     ''',
    ...     'outputs/diagrams/losses.svg',
    ... )
    """
    return render(
        textwrap.dedent(source).strip(),
        output,
        scale=scale,
        theme=theme,
        background=background,
        width=width,
        height=height,
        mermaid_config=mermaid_config,
    )
