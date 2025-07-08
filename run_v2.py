#!/usr/bin/env python3
"""
Startup script for Legal Research Platform v2.0
Run this instead of main_app.py to use the new multi-page interface
"""

import subprocess
import sys
import os

def main():
    """Launch the v2.0 application"""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    try:
        # Run streamlit with the new app.py
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()