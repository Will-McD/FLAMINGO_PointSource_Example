#!/bin/env python
"""
Telescopes and X-ray bands of the FLAMINGO lightcones.
"""

# telescopes with a response function read with read_ARF from their name alone
PRESET_TELESCOPES = ("ROSAT", "eROSITA", "none")

# X-ray bands of the SWIFT lightcones [keV], keys are lower case
FLAMINGO_BANDS = {
    "erosita-low": (0.2, 2.3),
    "erosita-high": (2.3, 8.0),
    "rosat": (0.5, 2.0),
}

