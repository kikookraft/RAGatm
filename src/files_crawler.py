import os

"""
Welcome to the file crawler.
This module crawls through the files searching for everything needed
to build a solid index.
"""
# File extensions
_PYTHON_EXTENSIONS: frozenset[str] = frozenset({".py", ".pyw", ".pyx", ".pxd"})
_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".rst", ".txt",
     ".cfg", ".ini", ".toml", ".yaml", ".yml"}
)


def discovery(root_dir: str) -> list[str]:
    """Discover all files in the root directory and its subdirectories."""
    file_paths: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() \
                    in _PYTHON_EXTENSIONS.union(_TEXT_EXTENSIONS):
                file_paths.append(os.path.join(dirpath, filename))
    print(f"Discovered {len(file_paths)} files in {root_dir}.")
    return file_paths


if __name__ == "__main__":
    root_directory = "data/raw"
    discovered_files = discovery(root_directory)
