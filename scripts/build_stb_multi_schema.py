"""STB multi-schema renderer (online PPA).

Exposes the public renderer interface consumed by `src.data.ppa_render`:

    VARIANTS: tuple[str, ...]
    build_variant(pair: dict, variant: str) -> dict

For a canonical (un-rendered) StableToolBench step pair `pair`, `build_variant`
returns a deep-copied pair rewritten under one of four schema variants:

    _base       canonical ReAct format, original tool names                (identity)
    _json       JSON-bracket format, original tool names                   (format switch)
    _rename     ReAct format, deterministically renamed tool names         (rename)
    _combined   JSON-bracket format with renamed tool names                (rename + format switch)

The renderer rewrites:
  * the system message's tool list,
  * every prior assistant turn in `state_messages`,
  * `expert_step_text`, `expert_action_text`, `expert_thought_text`,
  * `expert_tool_name` (under `_rename` / `_combined`),
  * `pair_id` (suffixed with the variant tag),

and clears any stale negative-side fields so the downstream neggen pass must
regenerate them against the rewritten state.

The rename map is deterministic per pair (RNG seeded from `pair_id`), so the
same canonical pair always produces the same rewrite — `OnlinePPAStepDataset`
relies on this to look up the corresponding negative in `negatives_lookup.json`
keyed by `f"{pair_id}__{variant}"`.

`Finish` is preserved across `_rename` / `_combined` so the StableToolBench
evaluator's terminal-action match still fires at eval time.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re


VARIANTS = ("_base", "_json", "_rename", "_combined")


# ── Deterministic per-pair RNG ──────────────────────────────────────────────

def _pair_rng(pair_id: str) -> random.Random:
    """Seed a `random.Random` from a stable hash of `pair_id`.

    Using `hashlib.blake2b` instead of Python's `hash()` because the latter is
    salted per process, which would break reproducibility across runs.
    """
    digest = hashlib.blake2b(pair_id.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "big", signed=False)
    return random.Random(seed)


# ── Tool-name synonym generator ─────────────────────────────────────────────
#
# Mirrors the deterministic synonym strategy used in our offline SIA expansion
# (scripts/h3_prepare_augmented_data.generate_tool_synonym), reproduced here so
# that the release bundle is self-contained.

_FILLER_WORDS = {"for", "the", "a", "an", "of", "in", "on", "by", "with", "to", "from", "api"}

_ACTION_SYNONYMS = {
    "get":    ["fetch", "retrieve", "obtain", "query"],
    "search": ["find", "lookup", "query", "discover"],
    "list":   ["enumerate", "show", "display", "get_all"],
    "create": ["make", "generate", "build", "new"],
    "delete": ["remove", "drop", "erase", "destroy"],
    "update": ["modify", "change", "edit", "alter"],
    "check":  ["verify", "validate", "inspect", "test"],
    "user":   ["account", "profile", "member"],
    "info":   ["details", "data", "metadata", "summary"],
}


def _synonym_for(tool_name: str, rng: random.Random) -> str:
    """Generate a synonym for a single tool name using `rng`."""
    parts = tool_name.lower().replace("-", "_").split("_")
    meaningful = [p for p in parts if p not in _FILLER_WORDS and len(p) > 1]
    if not meaningful:
        meaningful = parts[:3]

    new_parts: list[str] = []
    synonymed = False
    for p in meaningful:
        if not synonymed and p in _ACTION_SYNONYMS:
            new_parts.append(rng.choice(_ACTION_SYNONYMS[p]))
            synonymed = True
        else:
            new_parts.append(p)

    if not synonymed:
        rng.shuffle(new_parts)

    style = rng.choice(("snake", "camel", "dot"))
    if style == "snake":
        return "_".join(new_parts)
    if style == "camel":
        return new_parts[0] + "".join(p.capitalize() for p in new_parts[1:])
    return ".".join(new_parts)


# ── System-prompt parsing / rewrite ─────────────────────────────────────────

_TOOL_LIST_RE = re.compile(r"\[[\s\S]*?\]")

REACT_SYSTEM_TEMPLATE = """You are a helpful assistant with access to the following tools:
{tools}

For each step, provide your Thought, then call a tool using:
Action: tool_name
Action Input: {{"arg": "value"}}

When you have the final answer, respond with:
Final Answer: your answer"""

JSON_SYSTEM_TEMPLATE = """You are a helpful assistant with access to the following tools:
{tools}

When you need to call a tool, respond with a JSON array:
[{{"function": "tool_name", "parameters": {{"arg": "value"}}}}]

When you have the final answer, respond with:
Final Answer: your answer"""


def _extract_tool_list(system_content: str):
    """Extract the JSON tool list from the system message.

    Returns (tools, match) on success, (None, None) otherwise.
    """
    m = _TOOL_LIST_RE.search(system_content)
    if not m:
        return None, None
    try:
        return json.loads(m.group()), m
    except json.JSONDecodeError:
        return None, None


def _build_name_map(tools: list[dict], rng: random.Random) -> dict[str, str]:
    """Return {original_name: synonym}, preserving `Finish` (terminal action)."""
    name_map: dict[str, str] = {}
    for t in tools:
        old = t["name"]
        if old.lower() == "finish":
            name_map[old] = old
        else:
            name_map[old] = _synonym_for(old, rng)
    return name_map


# ── Action-text rewriters ───────────────────────────────────────────────────

def _rename_action_text(text: str, name_map: dict[str, str]) -> str:
    """Replace tool-name occurrences in a ReAct action block."""
    if not text:
        return text
    for old, new in name_map.items():
        if old == new:
            continue
        text = text.replace(f"Action: {old}", f"Action: {new}")
        text = text.replace(f"'{old}'", f"'{new}'")
        text = text.replace(f'"{old}"', f'"{new}"')
    return text


def _react_to_json_action(text: str) -> str:
    """Convert a ReAct assistant turn to the JSON-bracket variant.

    `Final Answer: ...` and free-form text are passed through untouched.
    """
    if not text or text.startswith("Final Answer"):
        return text

    action_match = re.search(r"Action:\s*(\S+)", text)
    if not action_match:
        return text
    tool_name = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(\{.*\})", text, re.S)
    try:
        params = json.loads(input_match.group(1)) if input_match else {}
    except json.JSONDecodeError:
        params = {}

    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", text, re.S)
    thought = thought_match.group(1).strip() if thought_match else ""

    json_call = json.dumps([{"function": tool_name, "parameters": params}])
    return f"{thought}\n{json_call}" if thought else json_call


# ── Per-variant transforms ──────────────────────────────────────────────────

def _apply_rename(p: dict, rng: random.Random) -> dict | None:
    """Rewrite system + assistant + expert with synonym tool names."""
    state = p["state_messages"]
    tools, tools_match = _extract_tool_list(state[0]["content"])
    if tools is None:
        return None

    name_map = _build_name_map(tools, rng)

    new_tools = [{"name": name_map.get(t["name"], t["name"])} for t in tools]
    sys_content = state[0]["content"]
    state[0]["content"] = (
        sys_content[:tools_match.start()]
        + json.dumps(new_tools, indent=2)
        + sys_content[tools_match.end():]
    )

    for m in state:
        if m["role"] == "assistant":
            m["content"] = _rename_action_text(m["content"], name_map)

    for key in ("expert_step_text", "expert_action_text"):
        if p.get(key):
            p[key] = _rename_action_text(p[key], name_map)

    if p.get("expert_tool_name") in name_map:
        p["expert_tool_name"] = name_map[p["expert_tool_name"]]

    return p


def _apply_format_switch(p: dict) -> dict | None:
    """Rewrite system + assistant + expert from ReAct to JSON-bracket format."""
    state = p["state_messages"]
    tools, _ = _extract_tool_list(state[0]["content"])
    if tools is None:
        return None

    state[0]["content"] = JSON_SYSTEM_TEMPLATE.format(tools=json.dumps(tools, indent=2))

    for m in state:
        if m["role"] == "assistant":
            m["content"] = _react_to_json_action(m["content"])

    for key in ("expert_step_text", "expert_action_text"):
        if p.get(key):
            p[key] = _react_to_json_action(p[key])

    return p


# ── Public entry point ──────────────────────────────────────────────────────

def build_variant(pair: dict, variant: str) -> dict:
    """Render `pair` under `variant`. Returns a deep-copied, schema-translated pair.

    Required input fields:
      pair_id, state_messages (system [+ history]), expert_step_text,
      expert_action_text, expert_tool_name.

    Output fields rewritten consistently:
      state_messages, expert_step_text, expert_action_text, expert_thought_text,
      expert_tool_name, pair_id (suffixed), variant (no leading underscore).
      negative_step_text, negative_type, ref_logprob_* are cleared so the
      downstream neggen pass must repopulate them.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown STB variant {variant!r}; expected one of {VARIANTS}")

    p = copy.deepcopy(pair)
    rng = _pair_rng(p["pair_id"])

    if variant in ("_rename", "_combined"):
        renamed = _apply_rename(p, rng)
        if renamed is None:
            # Tool list could not be parsed — fall through to a base rendering
            # so the trainer still has a well-formed sample (DDP-safety).
            renamed = p
        p = renamed

    if variant in ("_json", "_combined"):
        switched = _apply_format_switch(p)
        if switched is None:
            switched = p
        p = switched

    p["expert_thought_text"] = ""

    p["pair_id"] = f"{p['pair_id']}{variant}"
    p["variant"] = variant.lstrip("_")

    p["negative_step_text"] = ""
    p["negative_type"] = ""
    p["ref_logprob_expert"] = 0.0
    p["ref_logprob_negative"] = 0.0

    return p


__all__ = ["VARIANTS", "build_variant"]
