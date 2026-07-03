"""Put the project root on sys.path so tests can `import config`, `import tracker`, etc.

None of these tests import DeepFace/TensorFlow — they exercise pure logic with
synthetic bboxes and embeddings, so they run fast and need no camera or model.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                            # facerec package + main.py
sys.path.insert(0, os.path.join(_ROOT, "tools"))     # calibrate.py, manage_db.py, ...
