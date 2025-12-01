#!/usr/bin/env python
"""
Test script to verify the Python environment and imports.
"""
import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

print("\nTrying to import required packages...")

try:
    import fastapi
    print(f"✓ fastapi found: {fastapi.__version__}")
except ImportError as e:
    print(f"✗ Failed to import fastapi: {e}")

try:
    import uvicorn
    print(f"✓ uvicorn found: {uvicorn.__version__}")
except ImportError as e:
    print(f"✗ Failed to import uvicorn: {e}")

try:
    from playwright.sync_api import sync_playwright
    print(f"✓ playwright found")
except ImportError as e:
    print(f"✗ Failed to import playwright: {e}")

try:
    import requests
    print(f"✓ requests found: {requests.__version__}")
except ImportError as e:
    print(f"✗ Failed to import requests: {e}")

print("\nListing all installed packages:")
for pkg in sorted([f"{pkg.key}=={pkg.version}" for pkg in __import__("pkg_resources").working_set]):
    print(f"  {pkg}")

# Print sys.path to see where Python is looking for modules
print("\nPython module search paths:")
for path in sys.path:
    print(f"  {path}") 