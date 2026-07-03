import os
import sys

# Make `import mythos` work from a plain `pytest` invocation at the repo root,
# without requiring PYTHONPATH=. or an editable install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
