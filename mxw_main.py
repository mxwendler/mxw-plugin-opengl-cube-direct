"""
MXWendler media plugin: render a spinning OpenGL cube DIRECTLY into the media
texture - no readback, no texture streaming.

Address a clip's media as:  generative://cube_spin_opengl_direct

Unlike plugin_opengl_cube (render into an own fbo -> read pixels back to numpy
-> host uploads them via pbo), this plugin implements the direct hook:

    onRenderFrameGL(frame, texture, width, height) -> bool

The host (mxw_cachedmedia_plugin) passes the raw GL handle of the media's
stream texture (mxw_streamtexture::getStreamTexture()). We wrap it with
moderngl's external_texture(), attach it to our own framebuffer (plus our own
depth renderbuffer) and render the cube straight into it. Returning True tells
the host the frame is already in the texture: the pixel upload path is skipped
entirely (onRenderFrame is never called - this plugin has none).

Notes:
- ModernGL attaches to MXWendler's own GL context (moderngl.create_context()
  lazily on the first frame, never a standalone context) - same rule as
  plugin_opengl_cube, see there for the why.
- The texture handle can change (the host recreates the stream texture on a
  render size change), so the fbo is rebuilt whenever (texture, width, height)
  differs from the previous frame.
- Never release() the external texture wrapper: the texture belongs to
  MXWendler. Only our own fbo / depth renderbuffer are released.
- MXW media textures are top-down (row 0 = top of the image), a GL render is
  bottom-up, so the projection flips Y to compensate (the cpu plugin does the
  same with numpy flipud after the readback).

Install once with:
    pip install moderngl numpy
"""

import time
import math

import numpy as np
import moderngl

import mxw_imgui  # host UI: draw controls in the clip panel (onRenderPanel)


# ----------------------------------------------------------------------------------
# math helpers (row-major, textbook; uploaded transposed for column-major GLSL)
def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def translate(x, y, z):
    m = np.identity(4, dtype=np.float32)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def scale(s):
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = m[1, 1] = m[2, 2] = s
    return m


def rotate(angle_deg, x, y, z):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = c + x * x * (1 - c)
    m[0, 1] = x * y * (1 - c) - z * s
    m[0, 2] = x * z * (1 - c) + y * s
    m[1, 0] = y * x * (1 - c) + z * s
    m[1, 1] = c + y * y * (1 - c)
    m[1, 2] = y * z * (1 - c) - x * s
    m[2, 0] = z * x * (1 - c) - y * s
    m[2, 1] = z * y * (1 - c) + x * s
    m[2, 2] = c + z * z * (1 - c)
    return m


VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_col;
out vec3 v_col;
void main() {
    v_col = in_col;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_col;
out vec4 f_col;
void main() {
    f_col = vec4(v_col, 1.0);
}
"""


def _cube_geometry():
    # 8 corners, each given a colour so faces read as gradients
    v = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float32)
    c = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    idx = np.array([
        0, 1, 2, 2, 3, 0,   4, 5, 6, 6, 7, 4,
        0, 4, 7, 7, 3, 0,   1, 5, 6, 6, 2, 1,
        3, 2, 6, 6, 7, 3,   0, 1, 5, 5, 4, 0,
    ], dtype=np.int32)
    interleaved = np.hstack([v, c]).astype("f4")
    return interleaved.tobytes(), idx.tobytes()


# ----------------------------------------------------------------------------------
class cube_instance:
    def __init__(self):
        self.width = 1024
        self.height = 1024
        self.fps = 60.0
        # rotation is integrated over time so the clip playback speed can scale it
        # and reverse it: angle += dt * media_speed each frame. media_speed is set
        # per instance by the host via onSetSpeed().
        self.angle = 0.0
        self.last_time = time.monotonic()
        self.media_speed = 1.0
        # cube edge size, controlled by the onRenderPanel() slider (per instance)
        self.scale = 1.0
        self.ctx = None
        self.prog = None
        self.vao = None
        self.fbo = None
        self.depth = None
        self.fbo_key = None   # (texture, w, h) the current fbo was built for

    def ensure_gl(self, texture, width, height):
        # build GL objects lazily, on the render thread, with MXWendler's
        # context current -> create_context() attaches to *that* context.
        if self.ctx is None:
            self.ctx = moderngl.create_context()

            self.prog = self.ctx.program(
                vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

            vbo_data, ibo_data = _cube_geometry()
            vbo = self.ctx.buffer(vbo_data)
            ibo = self.ctx.buffer(ibo_data)
            self.vao = self.ctx.vertex_array(
                self.prog, [(vbo, "3f 3f", "in_pos", "in_col")], ibo)

        # (re)wrap the host's texture whenever the handle or size changed (the
        # host recreates the stream texture on a render size change). release
        # only OUR objects; the external texture wrapper must never be released,
        # the underlying GL texture belongs to MXWendler.
        if self.fbo_key != (texture, width, height):
            if self.fbo is not None:
                self.fbo.release()
            if self.depth is not None:
                self.depth.release()
            color = self.ctx.external_texture(texture, (width, height), 4, 0, "f1")
            self.depth = self.ctx.depth_renderbuffer((width, height))
            self.fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=self.depth)
            self.fbo_key = (texture, width, height)


storage = {}


def onOpen(uri):
    # do NOT create the GL context here: defer to the first onRenderFrameGL, where
    # MXWendler guarantees its own GL context is current.
    inst = cube_instance()
    storage[media_id] = inst

    # width, height, length(frames), fps, has_alpha
    return (inst.width, inst.height, 1, inst.fps, True)


def onRenderFrameGL(frame, texture, width, height):
    # direct hook: 'texture' is the raw GL handle of the media's stream texture.
    # render into it via our own fbo and return True -> the host skips the pixel
    # upload path (no readback, no pbo streaming).
    inst = storage.get(media_id)
    if inst is None:
        return False

    inst.width, inst.height = int(width), int(height)
    inst.ensure_gl(int(texture), inst.width, inst.height)

    # integrate rotation scaled by the clip playback speed. speed 0 freezes,
    # negative spins backwards.
    now = time.monotonic()
    dt = now - inst.last_time
    inst.last_time = now
    inst.angle += dt * inst.media_speed

    t = inst.angle
    aspect = inst.width / float(inst.height)

    model = rotate(t * 40.0, 1, 0, 0) @ rotate(t * 55.0, 0, 1, 0) @ scale(inst.scale)
    view = translate(0, 0, -4.5)
    proj = perspective(45.0, aspect, 0.1, 100.0)
    # MXW media textures are top-down, a GL render is bottom-up: flip Y in clip
    # space (the cpu sibling plugin flips with numpy flipud instead)
    proj[1, 1] = -proj[1, 1]
    mvp = proj @ view @ model

    inst.ctx.enable(moderngl.DEPTH_TEST)
    inst.fbo.use()
    inst.ctx.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    # GLSL is column-major -> upload the transpose of our row-major matrices
    inst.prog["mvp"].write(np.ascontiguousarray(mvp.T).tobytes())
    inst.vao.render()

    # the host snapshots and restores all GL state around this call, so we don't
    # need to unbind our fbo / reset enables here.
    return True


def onRenderPanel():
    # draw the plugin's controls in the clip ui, right above the Video Info panel.
    # the host sets the module global 'media_id' before the call, so we address this
    # instance. mxw is mid-frame with an active imgui context here.
    inst = storage.get(media_id)
    if inst is None:
        return
    mxw_imgui.set_next_item_width(200)
    changed, value = mxw_imgui.slider_float("Cube Size", inst.scale, 0.1, 3.0)
    if changed:
        inst.scale = value


def onSpeedRange():
    # the host forwards this to mxw_cached_media::get_speed_range(). allow the clip
    # playback speed from -5..5: the cube integrates speed into its rotation, so 0
    # freezes the spin and negative values spin it backwards.
    return (-5.0, 5.0)


def onSetSpeed(speed):
    # the host changed the clip playback speed. store it per instance: scales the
    # rotation, 0 freezes the spin and negative values spin it backwards.
    inst = storage.get(media_id)
    if inst is None:
        return
    inst.media_speed = float(speed)


def onSizeChange(w, h):
    # the host changed our render size. nothing to rebuild here: it also recreates
    # the stream texture, so the next onRenderFrameGL sees a new (texture, w, h)
    # key and ensure_gl() rewraps it. just record the size.
    inst = storage.get(media_id)
    if inst is None:
        return
    inst.width = int(w)
    inst.height = int(h)


def onClose():
    inst = storage.pop(media_id, None)
    if inst is None:
        return
    # our fbo / depth renderbuffer live in MXWendler's context; releasing the
    # python wrappers is enough. never touch the external texture (host-owned)
    # or the context itself -- we did not create it.
    inst.fbo = None
    inst.depth = None
    inst.vao = None
    inst.prog = None
    inst.ctx = None
