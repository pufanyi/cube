"""Rubik's Cube model: 26 cubies with position, orientation, and face colors.

Coordinate system:
  x: R(+) / L(-)
  y: U(+) / D(-)
  z: F(+) / B(-)

Each cubie is a 1x1x1 unit centered at integer coordinates in {-1, 0, 1}^3.
Face colors are tracked per-sticker via a (position, face_normal) -> color mapping.
"""

import numpy as np

# Standard Rubik's cube colors mapped to face normals
# Right=Red, Left=Orange, Up=White, Down=Yellow, Front=Green, Back=Blue
FACE_COLORS = {
    (+1, 0, 0): np.array([0.8, 0.0, 0.0]),  # R - Red
    (-1, 0, 0): np.array([1.0, 0.5, 0.0]),  # L - Orange
    (0, +1, 0): np.array([1.0, 1.0, 1.0]),  # U - White
    (0, -1, 0): np.array([1.0, 1.0, 0.0]),  # D - Yellow
    (0, 0, +1): np.array([0.0, 0.6, 0.0]),  # F - Green
    (0, 0, -1): np.array([0.0, 0.0, 0.8]),  # B - Blue
}

BLACK = np.array([0.05, 0.05, 0.05])

# Axis definitions for each move
MOVE_AXIS = {
    "R": (np.array([1.0, 0, 0]), +1),  # x-axis, layer x=+1
    "L": (np.array([1.0, 0, 0]), -1),  # x-axis, layer x=-1
    "U": (np.array([0, 1.0, 0]), +1),  # y-axis, layer y=+1
    "D": (np.array([0, 1.0, 0]), -1),  # y-axis, layer y=-1
    "F": (np.array([0, 0, 1.0]), +1),  # z-axis, layer z=+1
    "B": (np.array([0, 0, 1.0]), -1),  # z-axis, layer z=-1
    "M": (np.array([1.0, 0, 0]), 0),  # middle x
    "E": (np.array([0, 1.0, 0]), 0),  # middle y
    "S": (np.array([0, 0, 1.0]), 0),  # middle z
    "x": (np.array([1.0, 0, 0]), None),  # whole cube x
    "y": (np.array([0, 1.0, 0]), None),  # whole cube y
    "z": (np.array([0, 0, 1.0]), None),  # whole cube z
}


def rotation_matrix_90(axis: np.ndarray, clockwise: bool = True) -> np.ndarray:
    """90-degree rotation matrix around the given axis.

    Clockwise when looking from the positive end of the axis toward the origin.
    """
    ax = tuple(axis.astype(int))
    angle = -np.pi / 2 if clockwise else np.pi / 2

    c, s = round(np.cos(angle)), round(np.sin(angle))

    if ax == (1, 0, 0):
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif ax == (0, 1, 0):
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif ax == (0, 0, 1):
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError(f"Invalid axis: {ax}")


class Cubie:
    __slots__ = ("position", "stickers")

    def __init__(self, pos: tuple[int, int, int]):
        self.position = np.array(pos, dtype=float)
        # stickers: maps face_normal_tuple -> color
        self.stickers: dict[tuple[int, int, int], np.ndarray] = {}
        for normal, color in FACE_COLORS.items():
            # A cubie at pos has a sticker on face `normal` only if that face is on the outside
            n = np.array(normal)
            if np.dot(self.position, n) > 0.5:
                self.stickers[normal] = color.copy()

    def rotate(self, rot: np.ndarray):
        """Apply a 90-degree rotation matrix to this cubie."""
        self.position = np.round(rot @ self.position).astype(float)
        new_stickers = {}
        for normal, color in self.stickers.items():
            new_normal = tuple(np.round(rot @ np.array(normal)).astype(int))
            new_stickers[new_normal] = color
        self.stickers = new_stickers

    def get_face_color(self, face_normal: tuple[int, int, int]) -> np.ndarray:
        """Get the color for a face, or black if no sticker there."""
        return self.stickers.get(face_normal, BLACK)


class CubeState:
    def __init__(self):
        self.cubies: list[Cubie] = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    self.cubies.append(Cubie((x, y, z)))

    def get_affected_cubies(self, axis: np.ndarray, layer: int | None) -> list[Cubie]:
        """Get cubies in a specific layer (or all for whole-cube rotations)."""
        if layer is None:
            return list(self.cubies)
        axis_idx = int(np.argmax(np.abs(axis)))
        return [c for c in self.cubies if round(c.position[axis_idx]) == layer]

    def apply_move(self, axis: np.ndarray, layer: int | None, clockwise: bool = True):
        """Apply a 90-degree rotation to the specified layer."""
        rot = rotation_matrix_90(axis, clockwise)
        for cubie in self.get_affected_cubies(axis, layer):
            cubie.rotate(rot)


def parse_moves(notation: str) -> list[tuple[str, np.ndarray, int | None, bool, int]]:
    """Parse standard Rubik's cube notation into a list of moves.

    Returns list of (name, axis, layer, clockwise, repetitions).
    Supports: R, L, U, D, F, B, M, E, S, x, y, z
    Modifiers: ' (prime/CCW), 2 (double)
    Wide moves: Rw/r, Lw/l, Uw/u, Dw/d, Fw/f, Bw/b
    """
    moves = []
    tokens = tokenize_moves(notation)
    for token in tokens:
        name, axis, layer, clockwise, reps = interpret_token(token)
        moves.append((name, axis, layer, clockwise, reps))
    return moves


def tokenize_moves(notation: str) -> list[str]:
    """Split notation string into individual move tokens."""
    tokens = []
    i = 0
    s = notation.replace(" ", "")
    while i < len(s):
        if s[i] in "RLUDFBMESxyz":
            token = s[i]
            i += 1
            # Check for 'w' (wide move)
            if i < len(s) and s[i] == "w":
                token += "w"
                i += 1
            # Check for modifiers
            if i < len(s) and s[i] == "'":
                token += "'"
                i += 1
            elif i < len(s) and s[i] == "2":
                token += "2"
                i += 1
        elif s[i] in "rudfbl":
            # Lowercase = wide move
            token = s[i].upper() + "w"
            i += 1
            if i < len(s) and s[i] == "'":
                token += "'"
                i += 1
            elif i < len(s) and s[i] == "2":
                token += "2"
                i += 1
        else:
            i += 1
            continue
        tokens.append(token)
    return tokens


def interpret_token(token: str) -> tuple[str, np.ndarray, int | None, bool, int]:
    """Interpret a single move token.

    Returns (display_name, axis, layer, clockwise, repetitions).
    For wide moves, returns a list but we handle them as composite moves in the animator.
    """
    base = token[0]
    is_wide = "w" in token
    is_prime = "'" in token
    is_double = "2" in token

    axis_vec, layer = MOVE_AXIS[base]
    # CW direction: for positive-layer faces (R, U, F), CW is standard.
    # For negative-layer faces (L, D, B), the "clockwise" direction when
    # looking at the face is actually CCW around the positive axis.
    clockwise = True
    if layer is not None and layer < 0:
        clockwise = False
    if is_prime:
        clockwise = not clockwise

    reps = 2 if is_double else 1

    if is_wide:
        # Wide moves: we'll handle as layer=None with special treatment
        # For now, store as (axis, "wide") - the animator will handle both layers
        pass

    return (token, axis_vec, layer, clockwise, reps)


def get_wide_layers(base: str) -> list[int | None]:
    """For wide moves, return the two layers to rotate."""
    _, layer = MOVE_AXIS[base]
    if layer is None:
        return [None]
    # Wide move includes the face layer and the middle layer
    return [layer, 0]
