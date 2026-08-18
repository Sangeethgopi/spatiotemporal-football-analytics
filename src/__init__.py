"""
Pressing Intensity & Trigger Detection Thesis Package.
"""
import sys
import os

# Ensure package root is in path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
