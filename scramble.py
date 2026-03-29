"""Random scramble generator for Rubik's Cube.

Generates WCA-style scrambles: no two consecutive moves on the same face,
no three consecutive moves on the same axis.
"""

import random

FACES = ["R", "L", "U", "D", "F", "B"]
MODIFIERS = ["", "'", "2"]

# Faces that share the same axis (and thus shouldn't appear 3 times in a row)
AXIS_GROUP = {
    "R": 0,
    "L": 0,  # x-axis
    "U": 1,
    "D": 1,  # y-axis
    "F": 2,
    "B": 2,  # z-axis
}


def generate_scramble(num_moves: int, rng: random.Random | None = None) -> str:
    """Generate a random scramble sequence.

    Follows WCA-style constraints:
    - No two consecutive moves on the same face
    - No three consecutive moves on the same axis
    """
    if rng is None:
        rng = random.Random()

    moves = []
    prev_face = None
    prev_prev_axis = None
    prev_axis = None

    for _ in range(num_moves):
        available = list(FACES)
        # Remove same face as previous
        if prev_face is not None:
            available = [f for f in available if f != prev_face]
        # Remove faces on same axis if last two were on that axis
        if prev_axis is not None and prev_prev_axis is not None and prev_axis == prev_prev_axis:
            available = [f for f in available if AXIS_GROUP[f] != prev_axis]

        face = rng.choice(available)
        modifier = rng.choice(MODIFIERS)
        moves.append(face + modifier)

        prev_prev_axis = prev_axis
        prev_axis = AXIS_GROUP[face]
        prev_face = face

    return " ".join(moves)
