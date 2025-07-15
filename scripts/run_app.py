#!/usr/bin/env python3
"""
Startup script for the Churn Prediction Streamlit App
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Run the Streamlit app
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(current_dir.parent / "app" / "main.py"),
        "--server.port", "8501",
        "--server.address", "localhost"
    ]) 