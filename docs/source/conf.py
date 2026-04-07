"""Sphinx configuration for KVBoost documentation."""

project = "KVBoost"
copyright = "2025, Srihari Unnikrishnan"
author = "Srihari Unnikrishnan"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_title = "KVBoost"
html_logo = "../kvboost.svg"
html_favicon = "../kvboost.svg"

html_theme_options = {
    "source_repository": "https://github.com/pythongiant/kvboost",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Suppress import warnings for torch (heavy dependency)
autodoc_mock_imports = ["torch", "transformers", "accelerate", "sentencepiece"]
