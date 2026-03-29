"""ModernGL offscreen renderer for Rubik's Cube.

Renders 26 cubies with colored stickers, black body, and Phong lighting.
Hidden faces (back, left, bottom) are shown as flat "mirror" stickers
positioned behind the cube in 3D space, similar to alg.cubing.net.
"""

import moderngl
import numpy as np
from pyrr import Matrix44, Vector3

from cube_model import BLACK, CubeState


def mat44_from_axis_angle(axis: np.ndarray, angle: float) -> Matrix44:
    """Create a 4x4 rotation matrix from axis-angle (Rodrigues)."""
    ax = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    x, y, z = ax

    m = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    )
    # Transpose: pyrr sends row-major bytes, but GLSL reads column-major,
    # so the GPU sees M^T. Transposing here cancels that out.
    return Matrix44(m.T)


VERTEX_SHADER = """
#version 330

uniform mat4 mvp;
uniform mat4 model;
uniform mat3 normal_matrix;

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = normalize(normal_matrix * in_normal);
    v_position = world_pos.xyz;
    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330

uniform vec3 light_dir;
uniform vec3 camera_pos;

in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(light_dir);

    float ambient = 0.3;
    float diff = max(dot(normal, light), 0.0);

    vec3 view_dir = normalize(camera_pos - v_position);
    vec3 reflect_dir = reflect(-light, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.3;

    vec3 result = v_color * (ambient + 0.7 * diff) + vec3(1.0) * spec;
    frag_color = vec4(result, 1.0);
}
"""

# Face definitions for a unit cube: (normal, 4 corner vertices)
CUBE_FACES = [
    # Right (+x)
    ((1, 0, 0), [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)]),
    # Left (-x)
    ((-1, 0, 0), [(-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5)]),
    # Up (+y)
    ((0, 1, 0), [(-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)]),
    # Down (-y)
    ((0, -1, 0), [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)]),
    # Front (+z)
    ((0, 0, 1), [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]),
    # Back (-z)
    ((0, 0, -1), [(0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)]),
]

# Quad templates for mirror stickers. Each maps a face normal direction
# to 4 corner offsets (CCW winding facing the camera at +x,+y,+z).
# s = half-size of sticker
_MIRROR_QUAD_TEMPLATES = {
    # Back face mirror: quads in XY plane, facing +z
    (0, 0, -1): lambda s: [(-s, -s, 0), (+s, -s, 0), (+s, +s, 0), (-s, +s, 0)],
    # Left face mirror: quads in YZ plane, facing +x
    (-1, 0, 0): lambda s: [(0, -s, +s), (0, -s, -s), (0, +s, -s), (0, +s, +s)],
    # Bottom face mirror: quads in XZ plane, facing +y
    (0, -1, 0): lambda s: [(-s, 0, +s), (+s, 0, +s), (+s, 0, -s), (-s, 0, -s)],
}


def build_cubie_vertices(
    cubie_pos: np.ndarray, sticker_colors: dict, cubie_scale: float = 0.88
) -> tuple[np.ndarray, np.ndarray]:
    """Build vertex data for a single cubie."""
    vertices = []
    indices = []
    idx_offset = 0

    body_scale = cubie_scale
    sticker_inset = 0.08
    sticker_raise = 0.003

    for normal_tuple, face_verts in CUBE_FACES:
        normal = np.array(normal_tuple, dtype=float)
        abs_normal = np.abs(normal)

        for v in face_verts:
            pos = np.array(v) * body_scale + cubie_pos
            vertices.append((*pos, *normal, *BLACK))
        indices.extend(
            [idx_offset, idx_offset + 1, idx_offset + 2, idx_offset, idx_offset + 2, idx_offset + 3]
        )
        idx_offset += 4

        color = sticker_colors.get(normal_tuple)
        if color is not None:
            for v in face_verts:
                vv = np.array(v)
                tangential_scale = body_scale - sticker_inset
                pos = (
                    vv * (abs_normal * body_scale + (1 - abs_normal) * tangential_scale)
                    + cubie_pos
                    + normal * sticker_raise
                )
                vertices.append((*pos, *normal, *color))
            indices.extend(
                [
                    idx_offset,
                    idx_offset + 1,
                    idx_offset + 2,
                    idx_offset,
                    idx_offset + 2,
                    idx_offset + 3,
                ]
            )
            idx_offset += 4

    return np.array(vertices, dtype="f4"), np.array(indices, dtype="i4")


def build_mirror_stickers(
    cube: CubeState, mirror_gap: float = 4.5, sticker_half: float = 0.38, dim: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat colored quads behind the cube showing the 3 hidden faces.

    Positions mirror stickers in 3D space offset from each hidden face,
    like reflections on walls/floor behind the cube (alg.cubing.net style).
    """
    vertices = []
    indices = []
    idx = 0

    # Each entry: (face_normal, axis_index, mirror_coord)
    # face_normal: which cube face to mirror
    # axis_index: which axis the face is on (0=x, 1=y, 2=z)
    # mirror_coord: the fixed coordinate for the mirror plane
    hidden_faces = [
        ((0, 0, -1), 2, -(1.5 + mirror_gap)),  # Back face → z = -3.3
        ((-1, 0, 0), 0, -(1.5 + mirror_gap)),  # Left face → x = -3.3
        ((0, -1, 0), 1, -(1.5 + mirror_gap)),  # Bottom face → y = -3.3
    ]

    for face_normal, axis_idx, mirror_coord in hidden_faces:
        quad_offsets = _MIRROR_QUAD_TEMPLATES[face_normal](sticker_half)
        # The outward normal for the mirror (faces toward camera)
        mirror_normal = [0.0, 0.0, 0.0]
        mirror_normal[axis_idx] = 1.0 if mirror_coord < 0 else -1.0

        for cubie in cube.cubies:
            color = cubie.stickers.get(face_normal)
            if color is None:
                continue

            # Sticker center = cubie position but with the face-axis replaced by mirror_coord
            center = cubie.position.copy()
            center[axis_idx] = mirror_coord

            dimmed = color * dim
            for dx, dy, dz in quad_offsets:
                pos = center + np.array([dx, dy, dz])
                vertices.append((*pos, *mirror_normal, *dimmed))
            indices.extend([idx, idx + 1, idx + 2, idx, idx + 2, idx + 3])
            idx += 4

    if not vertices:
        return np.zeros((0, 9), dtype="f4"), np.zeros(0, dtype="i4")
    return np.array(vertices, dtype="f4"), np.array(indices, dtype="i4")


class CubeRenderer:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((width, height)),
        )

        # Camera: pulled back to show cube + all mirror stickers without occlusion
        self.camera_pos = Vector3([8.5, 6.5, 8.5])
        self.proj = Matrix44.perspective_projection(
            fovy=45.0,
            aspect=width / height,
            near=0.1,
            far=100.0,
        )
        self.view = Matrix44.look_at(
            eye=self.camera_pos,
            target=Vector3([-1.2, -1.5, -1.2]),
            up=Vector3([0.0, 1.0, 0.0]),
        )
        self.vp = self.proj * self.view

        light_dir = np.array([1.0, 1.5, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        self.prog["light_dir"].value = tuple(light_dir)
        self.prog["camera_pos"].value = tuple(self.camera_pos)

    def render_frame(
        self,
        cube: CubeState,
        rotating_cubies: list | None = None,
        rotation_axis: np.ndarray | None = None,
        rotation_angle: float = 0.0,
        mirrors: bool = True,
    ) -> bytes:
        """Render one frame of the cube state with optional mirror stickers."""
        self.fbo.use()
        self.ctx.clear(0.12, 0.12, 0.14, 1.0)

        rotating_set = set(rotating_cubies) if rotating_cubies else set()
        static_verts, static_idxs = [], []
        anim_verts, anim_idxs = [], []
        s_off, a_off = 0, 0

        for i, cubie in enumerate(cube.cubies):
            sticker_colors = {n: c for n, c in cubie.stickers.items()}
            verts, idxs = build_cubie_vertices(cubie.position, sticker_colors)
            if i in rotating_set and rotation_axis is not None:
                anim_verts.append(verts)
                anim_idxs.append(idxs + a_off)
                a_off += len(verts)
            else:
                static_verts.append(verts)
                static_idxs.append(idxs + s_off)
                s_off += len(verts)

        # Add mirror stickers to static geometry
        if mirrors:
            mv, mi = build_mirror_stickers(cube)
            if len(mv) > 0:
                static_verts.append(mv)
                static_idxs.append(mi + s_off)
                s_off += len(mv)

        # Render static
        if static_verts:
            self._render_batch(
                np.concatenate(static_verts),
                np.concatenate(static_idxs),
                Matrix44.identity(),
            )

        # Render animated
        if anim_verts and rotation_axis is not None:
            rot_mat = mat44_from_axis_angle(rotation_axis, rotation_angle)
            self._render_batch(
                np.concatenate(anim_verts),
                np.concatenate(anim_idxs),
                rot_mat,
            )

        # Read pixels
        raw = self.fbo.color_attachments[0].read()
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        frame = frame[::-1, :, :3].copy()
        return frame.tobytes()

    def _render_batch(self, vertices: np.ndarray, indices: np.ndarray, model: Matrix44):
        """Render a batch of geometry with a given model matrix."""
        mvp = self.vp * model
        self.prog["mvp"].write(mvp.astype("f4").tobytes())
        self.prog["model"].write(model.astype("f4").tobytes())

        m3 = np.array(model, dtype="f4")[:3, :3]
        normal_mat = np.linalg.inv(m3).T
        self.prog["normal_matrix"].write(normal_mat.astype("f4").tobytes())

        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        vao = self.ctx.vertex_array(
            self.prog,
            [(vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
            index_buffer=ibo,
        )
        vao.render(moderngl.TRIANGLES)
        vao.release()
        vbo.release()
        ibo.release()

    def release(self):
        self.fbo.release()
        self.ctx.release()
