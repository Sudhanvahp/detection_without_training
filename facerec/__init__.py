"""
facerec — core library for the face recognition system.

Modules:
  config      central configuration constants
  logger      rotating file + console logging
  clihelpers  shared argparse validators
  embedding   ArcFace embedding extraction (CLAHE + DeepFace lock)
  database    SQLite layer (encrypted embeddings, detection log, meta)
  recognizer  detection + matching engine (margin check, liveness)
  tracker     persistent per-face tracking + name confirmation
  camera      webcam wrapper with auto-reconnect
  visualizer  bounding boxes, HUD, prompts, snapshots

Entry points (main.py, register_faces.py, ...) live at the project root.
"""
