#!/usr/bin/env python3
"""
Startup script for the Churn Prediction API
"""
import uvicorn
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Add parent directory to Python path
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    
    # Run the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(parent_dir)]
    ) 