"""Walk the KVBoost source tree and emit a JSON manifest the static site reads.

For each Python file under ``src/kvboost`` we record:

- the dotted module name
- the relative path (for source links)
- the file's first-line docstring summary
- the full module docstring (rendered as a Markdown-ish blob)
- top-level classes and functions, with their one-line summaries
- line-of-code count
- size in bytes

Output is written to ``docs/site/manifest.json``. The site at
``docs/site/index.html`` loads it client-side. Re-run this script any time
the source tree changes, or wire it into the GitHub Actions workflow.

The script also copies the benchmark figures from ``docs/figures/`` into
``docs/site/figures/`` so the site is fully self-contained — works whether
you serve it via ``python -m http.server docs/site`` or upload it as a
flat directory to GitHub Pages. (The previous ``../figures/`` paths broke
both, because static servers refuse to follow ``..`` outside the doc root.)
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "kvboost"
SITE_DIR = Path(__file__).resolve().parent

# Static metadata describing the high-level "feature areas" of the project,
# used by the site to colour-code modules and group them in the explorer.
FEATURE_GROUPS: list[dict] = [
    {
        "id": "core",
        "label": "Core engine",
        "color": "#4f46e5",
        "emoji": "🧠",
        "matches": ["engine.py", "models.py", "compat.py"],
        "tagline": "Inference engine, model loading, HF compatibility.",
    },
    {
        "id": "cache",
        "label": "Chunk reuse",
        "color": "#0ea5e9",
        "emoji": "🗂️",
        "matches": [
            "chunk_registry.py", "cache_manager.py", "prompt_assembler.py",
            "selective_recompute.py", "cacheblend.py", "kv_quantize.py",
            "disk_tier.py", "batch.py",
        ],
        "tagline": "Content-hashed KV chunks, prefix assembly, selective recompute.",
    },
    {
        "id": "flash",
        "label": "Flash attention",
        "color": "#f97316",
        "emoji": "⚡",
        "matches": ["flash_attn_ext.py", "csrc"],
        "tagline": "Custom FlashAttention-2 CUDA kernel + Python wrapper.",
    },
    {
        "id": "streaming",
        "label": "AWQ Layer Streaming",
        "color": "#a855f7",
        "emoji": "🌊",
        "matches": ["streaming"],
        "tagline": "Stream AWQ layer weights from host RAM to run models bigger than VRAM.",
    },
    {
        "id": "paged",
        "label": "CPU paged decode",
        "color": "#10b981",
        "emoji": "📑",
        "matches": ["cpu_paged"],
        "tagline": "Block-paged KV cache for CPU-bound serving.",
    },
    {
        "id": "server",
        "label": "OpenAI-compatible server",
        "color": "#ec4899",
        "emoji": "🚀",
        "matches": ["server"],
        "tagline": "FastAPI server with batching, streaming, tool calls.",
    },
]


@dataclass
class Symbol:
    name: str
    kind: str  # "class" | "function"
    lineno: int
    summary: str
    is_public: bool


@dataclass
class ModuleEntry:
    name: str
    path: str
    rel_path: str
    summary: str
    docstring: str
    loc: int
    bytes: int
    classes: list[Symbol] = field(default_factory=list)
    functions: list[Symbol] = field(default_factory=list)
    group: str = "core"
    color: str = "#4f46e5"
    emoji: str = "🧠"


def _first_sentence(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.strip().split("\n\n")[0]
    # Collapse newlines into spaces for one-line summary.
    text = re.sub(r"\s+", " ", text).strip()
    # Trim at the first sentence boundary.
    m = re.match(r"^(.{0,180}?[.!?])(?:\s|$)", text)
    if m:
        return m.group(1).strip()
    return text[:180].strip()


def _classify_group(rel_path: str) -> dict:
    for grp in FEATURE_GROUPS:
        for needle in grp["matches"]:
            if needle in rel_path:
                return grp
    return FEATURE_GROUPS[0]  # default → core


def _extract_symbols(tree: ast.AST) -> tuple[list[Symbol], list[Symbol]]:
    classes: list[Symbol] = []
    functions: list[Symbol] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.ClassDef):
            classes.append(Symbol(
                name=node.name,
                kind="class",
                lineno=node.lineno,
                summary=_first_sentence(ast.get_docstring(node)),
                is_public=not node.name.startswith("_"),
            ))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(Symbol(
                name=node.name,
                kind="function",
                lineno=node.lineno,
                summary=_first_sentence(ast.get_docstring(node)),
                is_public=not node.name.startswith("_"),
            ))
    return classes, functions


def _module_dotted_name(py_path: Path) -> str:
    rel = py_path.relative_to(SRC_ROOT.parent)  # ".../kvboost/foo/bar.py"
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _git_describe() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return ""

    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": _git("config", "--get", "remote.origin.url"),
    }


def build_manifest() -> dict:
    modules: list[ModuleEntry] = []
    total_loc = 0

    for py_path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in py_path.parts:
            continue
        rel_path = py_path.relative_to(REPO_ROOT).as_posix()
        try:
            source = py_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        try:
            tree = ast.parse(source, filename=str(py_path))
        except SyntaxError:
            continue

        classes, functions = _extract_symbols(tree)
        doc = ast.get_docstring(tree) or ""
        summary = _first_sentence(doc) or _first_sentence(source[: 400])

        loc = sum(1 for ln in source.splitlines() if ln.strip())
        total_loc += loc

        grp = _classify_group(rel_path)

        modules.append(ModuleEntry(
            name=_module_dotted_name(py_path),
            path=str(py_path),
            rel_path=rel_path,
            summary=summary,
            docstring=doc,
            loc=loc,
            bytes=py_path.stat().st_size,
            classes=classes,
            functions=functions,
            group=grp["id"],
            color=grp["color"],
            emoji=grp["emoji"],
        ))

    # Also surface C/CUDA + Rust files as "non-Python" entries so the
    # explorer shows the whole stack, not just the Python.
    extras: list[dict] = []
    for ext_root, label in (
        (SRC_ROOT, "*.cu"), (SRC_ROOT, "*.cpp"), (SRC_ROOT, "*.h"),
        (REPO_ROOT / "crates", "*.rs"),
    ):
        if not ext_root.exists():
            continue
        for f in sorted(ext_root.rglob(label)):
            if "target" in f.parts or "__pycache__" in f.parts:
                continue
            try:
                source = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            extras.append({
                "rel_path": f.relative_to(REPO_ROOT).as_posix(),
                "loc": sum(1 for ln in source.splitlines() if ln.strip()),
                "bytes": f.stat().st_size,
                "lang": f.suffix.lstrip("."),
            })

    return {
        "schema": 1,
        "git": _git_describe(),
        "feature_groups": FEATURE_GROUPS,
        "modules": [
            {
                **{k: v for k, v in asdict(m).items() if k not in {"classes", "functions"}},
                "classes": [asdict(s) for s in m.classes],
                "functions": [asdict(s) for s in m.functions],
            }
            for m in modules
        ],
        "extras": extras,
        "stats": {
            "module_count": len(modules),
            "extra_file_count": len(extras),
            "total_loc": total_loc,
            "feature_count": len(FEATURE_GROUPS),
        },
    }


def copy_figures() -> int:
    """Copy benchmark figures into ``docs/site/figures/`` so the site is
    self-contained. Returns the count copied (0 if no source dir exists).
    """
    src = REPO_ROOT / "docs" / "figures"
    if not src.exists():
        return 0
    dst = SITE_DIR / "figures"
    dst.mkdir(exist_ok=True)
    copied = 0
    for img in src.glob("*.png"):
        shutil.copy2(img, dst / img.name)
        copied += 1
    return copied


def main() -> int:
    manifest = build_manifest()

    # Write both:
    # - manifest.json: machine-readable, useful for tooling / external consumers.
    # - manifest.js:   wraps the same payload in ``window.MANIFEST = {...}``.
    #                  Loaded via a regular <script> tag, so the site works
    #                  when opened directly via file:// (where browsers
    #                  block fetch()) as well as over HTTP/HTTPS.
    json_out = SITE_DIR / "manifest.json"
    json_out.write_text(json.dumps(manifest, indent=2))

    js_out = SITE_DIR / "manifest.js"
    js_payload = json.dumps(manifest, separators=(",", ":"))
    js_out.write_text(
        "/* Auto-generated by docs/site/generate_manifest.py. Do not edit. */\n"
        f"window.MANIFEST = {js_payload};\n"
    )

    n_figures = copy_figures()

    stats = manifest["stats"]
    print(
        f"Wrote {json_out} + {js_out.name} — {stats['module_count']} modules, "
        f"{stats['total_loc']} LOC across {stats['feature_count']} feature areas, "
        f"+{stats['extra_file_count']} native files. "
        f"Staged {n_figures} benchmark figures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
