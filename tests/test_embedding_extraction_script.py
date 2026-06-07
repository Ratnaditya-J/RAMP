from __future__ import annotations

import ast
from pathlib import Path


def test_embedding_extraction_script_has_no_import_time_torch_dependency() -> None:
    script = Path("scripts/extract_gpt_oss_embeddings.py")
    tree = ast.parse(script.read_text(encoding="utf-8"))

    top_level_imports = {
        alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names
    }
    top_level_from_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "torch" not in top_level_imports
    assert "transformers" not in top_level_imports
    assert "torch" not in top_level_from_imports
    assert "transformers" not in top_level_from_imports


def test_embedding_extraction_script_supports_input_embedding_representation() -> None:
    script_text = Path("scripts/extract_gpt_oss_embeddings.py").read_text(encoding="utf-8")

    assert "--representation" in script_text
    assert "input_embedding" in script_text
    assert "model.get_input_embeddings()" in script_text
    assert "--resume" in script_text
    assert "--progress-every" in script_text
