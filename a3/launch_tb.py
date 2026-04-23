#!/usr/bin/env python3
"""Launch TensorBoard with a workaround for the duplicate 'projector' plugin bug."""

import sys
import tensorboard.backend.application as app

# Monkey-patch to deduplicate plugins by name
_orig_init = app.TensorBoardWSGI.__init__

def _patched_init(self, plugins, *a, **kw):
    seen = {}
    deduped = []
    for p in plugins:
        name = getattr(p, 'plugin_name', None) or type(p).__name__
        if name not in seen:
            seen[name] = True
            deduped.append(p)
    _orig_init(self, deduped, *a, **kw)

app.TensorBoardWSGI.__init__ = _patched_init

from tensorboard.main import run_main
sys.exit(run_main())
