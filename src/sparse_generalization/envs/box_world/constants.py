from __future__ import annotations

import numpy as np

TILE_PIXELS = 32

# Map of color names to RGB values
import numpy as np

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "yellow": np.array([255, 255, 0]),
    "cyan": np.array([0, 255, 255]),
    "magenta": np.array([255, 0, 255]),
    "grey": np.array([100, 100, 100]),
    "brown": np.array([165, 42, 42]),
    "navy": np.array([0, 0, 128]),
    "orange": np.array([255, 165, 0]),
    "purple": np.array([128, 0, 128]),
    "teal": np.array([0, 128, 128]),
    "olive": np.array([128, 128, 0]),
    "pink": np.array([255, 192, 203]),
    "turquoise": np.array([64, 224, 208]),
    "gold": np.array([255, 215, 0]),
    "indigo": np.array([75, 0, 130]),
    "slateblue": np.array([106, 90, 205]),
    "chocolate": np.array([210, 105, 30]),
    "crimson": np.array([220, 20, 60]),
    "darkgreen": np.array([0, 100, 0]),
    "darkblue": np.array([0, 0, 139]),
    "darkred": np.array([139, 0, 0]),
    "darkorange": np.array([255, 140, 0]),
    "darkcyan": np.array([0, 139, 139]),
    "mediumvioletred": np.array([199, 21, 133]),
    "mediumslateblue": np.array([123, 104, 238]),
    "mediumturquoise": np.array([72, 209, 204]),
    "firebrick": np.array([178, 34, 34]),
    "royalblue": np.array([65, 105, 225]),
    "salmon": np.array([250, 128, 114]),
    "peru": np.array([205, 133, 63]),
    "khaki": np.array([240, 230, 140]),
    "violet": np.array([238, 130, 238]),
    "palegoldenrod": np.array([238, 232, 170]),
    "cadetblue": np.array([95, 158, 160]),
    "powderblue": np.array([176, 224, 230])
}



# Sorted color names
COLOR_NAMES = sorted(list(COLORS.keys()))

# Map color names to integers automatically
COLOR_TO_IDX = {name: idx for idx, name in enumerate(COLOR_NAMES)}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
    "tile": 11, 
    "keybox": 12, 
    "lockbox": 13, 
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]