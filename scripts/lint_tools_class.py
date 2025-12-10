#!/usr/bin/env python3
"""
Lint script to detect helper methods inside Tools class.

AI can invoke ALL methods inside the Tools class, regardless of naming.
The underscore prefix (_method) does NOT hide methods from AI.
This script prevents accidental exposure of helper methods.
"""
import ast
import sys
from pathlib import Path


def check_tools_class(filepath: Path) -> list[str]:
    """Detect forbidden methods inside Tools class."""
    errors = []
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        # Skip files that can't be parsed
        return [f"{filepath}: Parse error - {e}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Tools":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Allow dunder methods (__init__, __del__, etc.)
                    if item.name.startswith("_") and not item.name.startswith("__"):
                        errors.append(
                            f"{filepath}:{item.lineno}: "
                            f"Helper method '{item.name}' found in Tools class. "
                            f"Move it outside the class."
                        )
    return errors


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: lint_tools_class.py <file1.py> [file2.py ...]", file=sys.stderr)
        return 0

    errors = []
    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if path.suffix == ".py" and path.exists():
            errors.extend(check_tools_class(path))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
