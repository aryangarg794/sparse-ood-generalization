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
    "powderblue": np.array([176, 224, 230]),
    
    # ood
    "aliceblue": np.array([240, 248, 255]),
    "aquamarine": np.array([127, 255, 212]),
    "azure": np.array([240, 255, 255]),
    "beige": np.array([245, 245, 220]),
    "bisque": np.array([255, 228, 196]),
    "blanchedalmond": np.array([255, 235, 205]),
    "burlywood": np.array([222, 184, 135]),
    "chartreuse": np.array([127, 255, 0]),
    "coral": np.array([255, 127, 80]),
    "cornflowerblue": np.array([100, 149, 237]),
    "cornsilk": np.array([255, 248, 220]),
    "darkkhaki": np.array([189, 183, 107]),
    "darkmagenta": np.array([139, 0, 139]),
    "darkorchid": np.array([153, 50, 204]),
    "darksalmon": np.array([233, 150, 122]),
    "darkseagreen": np.array([143, 188, 143]),
    "darkslateblue": np.array([72, 61, 139]),
    "darkslategray": np.array([47, 79, 79]),
    "deeppink": np.array([255, 20, 147]),
    "deepskyblue": np.array([0, 191, 255]),
    "dodgerblue": np.array([30, 144, 255]),
    "forestgreen": np.array([34, 139, 34]),
    "fuchsia": np.array([255, 0, 255]),
    "gainsboro": np.array([220, 220, 220]),
    "ghostwhite": np.array([248, 248, 255]),
    "honeydew": np.array([240, 255, 240]),
    "hotpink": np.array([255, 105, 180]),
    "indianred": np.array([205, 92, 92]),
    "ivory": np.array([255, 255, 240]),
    "lavender": np.array([230, 230, 250]),
    "lawngreen": np.array([124, 252, 0]),
    "lemonchiffon": np.array([255, 250, 205]),
    "lightblue": np.array([173, 216, 230]),
    "lightcoral": np.array([240, 128, 128]),
    "lime": np.array([0, 255, 0]),
    "limegreen": np.array([50, 205, 50]),
    "linen": np.array([250, 240, 230]),
    "maroon": np.array([128, 0, 0]),
    "midnightblue": np.array([25, 25, 112]),
    "moccasin": np.array([255, 228, 181]),
    "orchid": np.array([218, 112, 214]),
    "plum": np.array([221, 160, 221]),
    "rosybrown": np.array([188, 143, 143]),
    "sienna": np.array([160, 82, 45]),
    "silver": np.array([192, 192, 192]),
    "skyblue": np.array([135, 206, 235]),
    "snow": np.array([255, 250, 250]),
    "springgreen": np.array([0, 255, 127]),
    "tan": np.array([210, 180, 140]),
    "thistle": np.array([216, 191, 216]),
    "tomato": np.array([255, 99, 71]),
    "wheat": np.array([245, 222, 179]),
}

# Sorted color names
ALL_COLOR_NAMES = sorted(list(COLORS.keys()))
COLOR_NAMES = ALL_COLOR_NAMES[:37]
OOD_COLOR_NAMES = ALL_COLOR_NAMES[37:]

# Map color names to integers automatically
COLOR_TO_IDX = {name: idx for idx, name in enumerate(ALL_COLOR_NAMES)}

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