#!/usr/bin/env python3
import os
import argparse

def path_to_module(file_path, root):
    rel_path = os.path.relpath(file_path, root)
    module_path = rel_path.replace(os.sep, ".")
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    return module_path

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Python file path to a module path."
    )
    parser.add_argument("file_path", help="Path to the .py file")
    parser.add_argument(
        "--root", "-r", required=True, help="Workspace root (added to sys.path)"
    )
    args = parser.parse_args()

    module_path = path_to_module(args.file_path, args.root)
    print(module_path)

if __name__ == "__main__":
    main()