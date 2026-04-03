import sys
import os
from pathlib import Path

# Add this project's root to sys.path so utils.* and nodes.* are importable
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)

# Prevent pytest from importing the package-level __init__.py as a test
# module. The root __init__.py uses ComfyUI relative imports that are only
# valid when loaded as part of the ComfyUI custom_nodes package.
collect_ignore = ["__init__.py"]


def pytest_ignore_collect(collection_path, config):
    """Skip the root __init__.py so pytest does not treat the project root as
    a Python package requiring relative-import initialization."""
    if collection_path.name == "__init__.py" and collection_path.parent == Path(_project_root):
        return True
    return None
