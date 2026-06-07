from __future__ import annotations

import ast
from pathlib import Path


def test_benchmark_corpus_script_defers_datasets_import() -> None:
    script = Path("scripts/build_benchmark_corpus.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))
    top_level_imports = {
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    }
    top_level_from_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "datasets" not in top_level_imports
    assert "datasets" not in top_level_from_imports
