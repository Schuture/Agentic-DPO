"""Online PPA renderer registry.

Each domain ships a small `build_*_multi_schema` module exposing:

  - ``VARIANTS``: tuple of variant suffix strings (e.g. ``("_base", "_json", ...)``)
  - ``build_variant(pair, variant) -> dict``: deep-copy + schema-translate the pair

The trainer's ``OnlinePPAStepDataset`` calls ``get_renderer(domain).build_variant``
on every gradient step to render a freshly sampled variant view of a canonical
step pair on the fly (see paper §3.3, Policy-Preserving Augmentation).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PPARenderer:
    name: str
    variants: tuple[str, ...]
    build_variant: Callable[[dict, str], dict]


_REGISTRY: dict[str, PPARenderer] = {}


def register_renderer(name: str, build_module: str) -> PPARenderer:
    """Lazy-import ``build_module`` and register it under ``name``."""
    module = importlib.import_module(build_module)
    renderer = PPARenderer(
        name=name,
        variants=tuple(module.VARIANTS),
        build_variant=module.build_variant,
    )
    _REGISTRY[name] = renderer
    return renderer


def get_renderer(name: str) -> PPARenderer:
    """Return the renderer for ``name``, lazy-registering known domains."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name == "stb":
        return register_renderer("stb", "scripts.build_stb_multi_schema")
    raise KeyError(
        f"No PPA renderer registered for {name!r}. Known domain: stb."
    )


def render_pair(pair: dict, *, domain: str, variant: str) -> dict:
    """Render ``pair`` under ``variant`` for the given ``domain``."""
    return get_renderer(domain).build_variant(pair, variant)
