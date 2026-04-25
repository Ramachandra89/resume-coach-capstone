#!/usr/bin/env python3
"""
Converts eda_analysis.py to eda_analysis.ipynb using jupytext.
Run: python convert_to_notebook.py
"""
import subprocess
import sys

def main():
    try:
        subprocess.run(
            ["jupytext", "--to", "ipynb", "eda_analysis.py", "-o", "eda_analysis.ipynb"],
            check=True
        )
        print("✅ Notebook created: eda_analysis.ipynb")
        print("   Run: jupyter notebook eda_analysis.ipynb")
    except FileNotFoundError:
        print("jupytext not found. Install: pip install jupytext")
        print("Alternatively, open eda_analysis.py as a percent script in VS Code or JupyterLab.")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
