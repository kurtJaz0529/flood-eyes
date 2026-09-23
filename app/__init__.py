"""慧眼识灾 · Gradio 应用包。

这里做一次 sys.path 引导，保证以下两种方式都能 import：
    python app/main.py
    python -c "from app.components import sample_choices"
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
