#!/usr/bin/env python3
"""Fix Celery decorators by adding type ignore comments properly."""

import re
import os

def fix_decorators_in_file(file_path: str) -> None:
    """Add type ignore comments to all @app.task decorators."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Pattern to match @app.task decorators that don't already have type ignore
    # This is more specific to avoid double replacements
    pattern = r'(@app\.task(?:\([^)]*\))?)(?!\s*#\s*type:\s*ignore)'
    
    def replacement(match):
        decorator = match.group(1)
        return f"{decorator}  # type: ignore[misc]"
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Fixed decorators in {file_path}")
    else:
        print(f"No changes needed in {file_path}")

# Fix all task files
task_files = [
    "density_engine/tasks/cluster.py",
    "density_engine/tasks/jobs.py", 
    "density_engine/tasks/recovery.py"
]

for file_path in task_files:
    if os.path.exists(file_path):
        fix_decorators_in_file(file_path)
    else:
        print(f"File not found: {file_path}")

print("Done fixing decorators!")
