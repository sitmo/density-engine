#!/usr/bin/env python3
"""
Launcher script for vast.ai automation.
This script runs the vast.ai automation from the project root.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the vast.ai script
from scripts.vast.run_garch_vast import main

if __name__ == "__main__":
    main()
