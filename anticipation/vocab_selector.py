"""
Dynamic vocabulary selector for anticipation tokenizer.
Allows switching between full vocabulary (with control block) and vanilla (without).

Usage:
    Default (full vocab):
        from anticipation.vocab_selector import *
    
    Vanilla vocab:
        export ANTICIPATION_VANILLA=true
        python your_script.py
"""

import os

# Check environment variable
USE_VANILLA = os.getenv('ANTICIPATION_VANILLA', 'false').lower() == 'true'

# Import everything from the selected vocab
if USE_VANILLA:
    from anticipation.vocab_vanilla import *
else:
    from anticipation.vocab_ant import *
