"""
Q-Knee Diagnostic Platform — root-level Streamlit entry point for
Streamlit Community Cloud / Hugging Face Spaces.

NISQ-Ready Hybrid Quantum ML for Knee Abnormality Triage.
Author: Yashika Nayak

Both platforms invoke `streamlit run <app_file>` with only the script's
own directory added to `sys.path` — not the repository root. Pointing
`app_file` directly at the nested `qknee/ui/dashboard.py` would leave
every `from qknee....` import in that module (and its dependencies)
unresolvable (`ModuleNotFoundError: No module named 'qknee'`). This
wrapper stays at the repo root, adds the repo root to `sys.path` itself
before importing anything under `qknee`, then delegates straight to the
real app (`qknee.ui.dashboard.main`) — see the `app_file:` entry in this
repo's README.md frontmatter. `qknee.ui.dashboard.main()` renders the
PRD-aligned, ORTHOC-template public landing page
(`qknee.ui.landing_page.render_landing_page`) by default, and switches to
the Diagnostic Workstation / Benchmark console once a visitor navigates
there.

This file also injects the ORTHOC base stylesheet
(`qknee.ui.landing_page.inject_orthoc_theme`) — but *after* `main()`
returns, not before it: Streamlit requires `st.set_page_config()` (called
inside `qknee.ui.dashboard.render_header()`, deep inside `main()`) to be
the very first Streamlit command executed in a script run, so any
`st.markdown()` call placed literally above the `main()` call below would
raise `StreamlitAPIException`. A `<style>` tag's rules apply to the whole
document regardless of where in the DOM it lands, so injecting it last in
the same rerun still restyles every element `main()` rendered above it —
this achieves the same effect as a top-of-script injection would, within
that ordering constraint.

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qknee.ui.dashboard import main

try:
    from qknee.ui.landing_page import inject_orthoc_theme
except ImportError:
    def inject_orthoc_theme() -> None:
        pass  # Graceful no-op fallback if landing_page.py is reorganized.

if __name__ == "__main__":
    main()
    inject_orthoc_theme()
