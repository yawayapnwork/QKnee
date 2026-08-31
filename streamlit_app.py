"""
Root-level Streamlit entry point for Streamlit Community Cloud / Hugging
Face Spaces.

Both platforms invoke `streamlit run <app_file>` with only the script's
own directory added to `sys.path` — not the repository root. Pointing
`app_file` directly at the nested `qknee/ui/dashboard.py` would leave
every `from qknee....` import in that module (and its dependencies)
unresolvable (`ModuleNotFoundError: No module named 'qknee'`). This
wrapper stays at the repo root, adds the repo root to `sys.path` itself
before importing anything under `qknee`, then delegates straight to the
real app (`qknee.ui.dashboard.main`) — see the `app_file:` entry in this
repo's README.md frontmatter.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qknee.ui.dashboard import main

if __name__ == "__main__":
    main()
