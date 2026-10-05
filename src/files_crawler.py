import logging
import os

"""
Welcome to the file crawler.
This module crawls through the files searching for everything needed
to build a solid index.
"""
logger = logging.getLogger(__name__)

# File extensions
_PYTHON_EXTENSIONS: frozenset[str] = frozenset({".py", ".pyw", ".pyx", ".pxd"})
_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".rst", ".txt",
     ".cfg", ".ini", ".toml", ".yaml", ".yml"}
)
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {".c", ".h", ".cpp", ".cc", ".hpp", ".cu", ".cuh",
     ".sh", ".bash", ".cmake", ".json"}
)

_INDEXABLE_EXTENSIONS = _PYTHON_EXTENSIONS.union(_TEXT_EXTENSIONS)  # .union(
# _CODE_EXTENSIONS)


def discovery(root_dir: str) -> list[str]:
    """Discover all indexable files in the root directory recursively."""
    file_paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() \
                    in _INDEXABLE_EXTENSIONS:
                file_paths.append(os.path.join(dirpath, filename))
    logger.info("Discovered %d files in %s.", len(file_paths), root_dir)
    return file_paths


if __name__ == "__main__":
    root_directory = "data/raw"
    discovered_files = discovery(root_directory)
