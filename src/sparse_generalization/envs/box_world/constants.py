from __future__ import annotations

import numpy as np

TILE_PIXELS = 32

# Map of color names to RGB values
import numpy as np

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "orange": np.array([255, 165, 0]),
    "pink": np.array([255, 192, 203]),
    "brown": np.array([165, 42, 42]),
    "cyan": np.array([0, 255, 255]),
    "magenta": np.array([255, 0, 255]),
    "navy": np.array([0, 0, 128]),
    "teal": np.array([0, 128, 128]),
    "olive": np.array([128, 128, 0]),
    "maroon": np.array([128, 0, 0]),
    "silver": np.array([192, 192, 192]),
    "gold": np.array([255, 215, 0]),
    "violet": np.array([238, 130, 238]),
    "indigo": np.array([75, 0, 130]),
    "white": np.array([255, 255, 255]),
    "coral": np.array([255, 127, 80]),
    "salmon": np.array([250, 128, 114]),
    "orchid": np.array([218, 112, 214]),
    "turquoise": np.array([64, 224, 208]),
    "skyblue": np.array([135, 206, 235]),
    "chocolate": np.array([210, 105, 30]),
    "crimson": np.array([220, 20, 60]),
    "plum": np.array([221, 160, 221]),
    "peru": np.array([205, 133, 63]),
    "khaki": np.array([240, 230, 140]),
    "lavender": np.array([230, 230, 250]),
    "beige": np.array([245, 245, 220]),
    "sienna": np.array([160, 82, 45]),
    "mintcream": np.array([245, 255, 250]),
    "firebrick": np.array([178, 34, 34]),
    "tomato": np.array([255, 99, 71]),
    "deeppink": np.array([255, 20, 147]),
    "hotpink": np.array([255, 105, 180]),
    "darkorange": np.array([255, 140, 0]),
    "goldenrod": np.array([218, 165, 32]),
    "palegoldenrod": np.array([238, 232, 170]),
    "indianred": np.array([205, 92, 92]),
    "mediumvioletred": np.array([199, 21, 133]),
    "thistle": np.array([216, 191, 216]),
    "mediumslateblue": np.array([123, 104, 238]),
    "dodgerblue": np.array([30, 144, 255]),
    "royalblue": np.array([65, 105, 225]),
    "steelblue": np.array([70, 130, 180]),
    "darkslateblue": np.array([72, 61, 139]),
    "mediumturquoise": np.array([72, 209, 204]),
    "cadetblue": np.array([95, 158, 160]),
    "powderblue": np.array([176, 224, 230]),
    "tan": np.array([210, 180, 140]),
    "rosybrown": np.array([188, 143, 143]),
    "saddlebrown": np.array([139, 69, 19]),
    "darkred": np.array([139, 0, 0]),
    "mediumorchid": np.array([186, 85, 211]),
    "blueviolet": np.array([138, 43, 226]),
    "slateblue": np.array([106, 90, 205]),
    "cornflowerblue": np.array([100, 149, 237]),
    "aliceblue": np.array([240, 248, 255]),
    "ivory": np.array([255, 255, 240]),
    "lightgray": np.array([211, 211, 211]),
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
    "tile": 11
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