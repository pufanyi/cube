"""ModernGL offscreen renderer for Rubik's Cube.

Renders 26 cubies with colored stickers, black body, and Phong lighting.
Each cubie is a slightly smaller cube (gap between cubies) with colored face quads.
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
    return Matrix44(m)


# Vertex shader: transforms vertices and passes normals/colors to fragment shader
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

# Fragment shader: Phong lighting
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

    // Ambient
    float ambient = 0.3;

    // Diffuse
    float diff = max(dot(normal, light), 0.0);

    // Specular
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


def build_cubie_vertices(
    cubie_pos: np.ndarray, sticker_colors: dict, cubie_scale: float = 0.88
) -> tuple[np.ndarray, np.ndarray]:
    """Build vertex data for a single cubie.

    Returns (vertices, indices) where each vertex has (pos, normal, color).
    The cubie body is dark, with colored sticker quads on outer faces.

    Stickers are inset tangentially (smaller) and raised slightly along the normal
    so they sit ON TOP of the body face, not behind it.
    """
    vertices = []
    indices = []
    idx_offset = 0

    body_scale = cubie_scale
    sticker_inset = 0.08  # How much smaller the sticker is (tangential)
    sticker_raise = 0.003  # How far the sticker pokes out above the body

    for normal_tuple, face_verts in CUBE_FACES:
        normal = np.array(normal_tuple, dtype=float)
        abs_normal = np.abs(normal)

        # Body face (dark)
        for v in face_verts:
            pos = np.array(v) * body_scale + cubie_pos
            vertices.append((*pos, *normal, *BLACK))
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

        # Sticker face (colored, raised above body)
        color = sticker_colors.get(normal_tuple)
        if color is not None:
            for v in face_verts:
                vv = np.array(v)
                # Scale tangential components (perpendicular to normal) inward
                tangential_scale = body_scale - sticker_inset
                # Normal component stays at body_scale + raise
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

        # Camera setup - looking at origin from a nice angle
        self.camera_pos = Vector3([4.5, 3.5, 4.5])
        self.view = Matrix44.look_at(
            eye=self.camera_pos,
            target=Vector3([0.0, 0.0, 0.0]),
            up=Vector3([0.0, 1.0, 0.0]),
        )
        self.proj = Matrix44.perspective_projection(
            fovy=40.0,
            aspect=width / height,
            near=0.1,
            far=100.0,
        )
        self.vp = self.proj * self.view

        # Light direction (from upper-right-front)
        light_dir = np.array([1.0, 1.5, 1.0])
        light_dir = light_dir / np.linalg.norm(light_dir)
        self.prog["light_dir"].value = tuple(light_dir)
        self.prog["camera_pos"].value = tuple(self.camera_pos)

        # Preallocate read buffer
        self._buf = bytearray(width * height * 3)

    def render_frame(
        self,
        cube: CubeState,
        rotating_cubies: list | None = None,
        rotation_axis: np.ndarray | None = None,
        rotation_angle: float = 0.0,
    ) -> bytes:
        """Render one frame of the cube state.

        Args:
            cube: The current cube state.
            rotating_cubies: List of cubie indices currently being animated.
            rotation_axis: Axis of rotation for the animated cubies.
            rotation_angle: Current angle of rotation in radians.

        Returns:
            Raw RGB bytes (width * height * 3).
        """
        self.fbo.use()
        self.ctx.clear(0.12, 0.12, 0.14, 1.0)

        rotating_set = set(rotating_cubies) if rotating_cubies else set()

        all_vertices = []
        all_indices = []
        static_offset = 0
        animated_vertices = []
        animated_indices = []
        animated_offset = 0

        for i, cubie in enumerate(cube.cubies):
            sticker_colors = {}
            for normal, color in cubie.stickers.items():
                sticker_colors[normal] = color

            verts, idxs = build_cubie_vertices(cubie.position, sticker_colors)

            if i in rotating_set and rotation_axis is not None:
                animated_vertices.append(verts)
                animated_indices.append(idxs + animated_offset)
                animated_offset += len(verts)
            else:
                all_vertices.append(verts)
                all_indices.append(idxs + static_offset)
                static_offset += len(verts)

        # Render static cubies (identity model matrix)
        if all_vertices:
            self._render_batch(
                np.concatenate(all_vertices), np.concatenate(all_indices), Matrix44.identity()
            )

        # Render animated cubies with rotation
        if animated_vertices:
            rot_mat = mat44_from_axis_angle(rotation_axis, rotation_angle)
            self._render_batch(
                np.concatenate(animated_vertices),
                np.concatenate(animated_indices),
                rot_mat,
            )

        # Read pixels
        raw = self.fbo.color_attachments[0].read()
        # Flip vertically (OpenGL origin is bottom-left)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        frame = frame[::-1, :, :3].copy()
        return frame.tobytes()

    def _render_batch(self, vertices: np.ndarray, indices: np.ndarray, model: Matrix44):
        """Render a batch of geometry with a given model matrix."""
        mvp = self.vp * model
        self.prog["mvp"].write(mvp.astype("f4").tobytes())
        self.prog["model"].write(model.astype("f4").tobytes())

        # Normal matrix = transpose(inverse(upper-left 3x3 of model))
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
