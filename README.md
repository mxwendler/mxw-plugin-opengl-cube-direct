# mxw-plugin-opengl-cube-direct

An MXWendler StageDesigner **media plugin** that renders a spinning, colour-shaded
OpenGL cube directly into the media's stream texture, using
[ModernGL](https://github.com/moderngl/moderngl).

It is a variant of the [`plugin_opengl_cube`](../plugin_opengl_cube) example: instead
of rendering into an own offscreen framebuffer and handing the host a pixel buffer
(readback -> numpy -> host re-uploads via PBO), this plugin implements the direct
GL hook and renders straight into the host's texture. No readback, no texture
streaming.

## Usage

In MXWendler, create the media with the URI:

```
generative://cube_spin_opengl_direct
```

(or pick **Spinning Cube (Direct)** from the media create dropdown).

## How it works

MXWendler's media-plugin host calls these entry points in `mxw_main.py`:

| Callback | Returns | Purpose |
|----------|---------|---------|
| `onOpen(uri)` | `(width, height, length, fps, has_alpha)` | report the surface format |
| `onRenderFrameGL(frame, texture, width, height)` | `bool` | render directly into the host's stream texture |
| `onRenderPanel()` | – | draw plugin controls (cube-size and top-scale sliders) in the clip panel |
| `onSpeedRange()` | `(min, max)` | allowed clip playback speed range |
| `onSetSpeed(speed)` | – | clip playback speed changed; store per instance |
| `onSizeChange(w, h)` | – | host changed the render size; record it |
| `onClose()` | – | release resources |

`onRenderFrameGL` is the direct hook: `texture` is the raw GL handle of the media's
stream texture (`mxw_streamtexture::getStreamTexture()`). The plugin wraps it with
ModernGL's `external_texture()`, attaches it to its own framebuffer (plus its own
depth renderbuffer), and renders the cube straight into it. Returning `True` tells
the host the frame is already in the texture, so the pixel-upload path
(`onRenderFrame`, which this plugin does not implement) is skipped entirely.

The texture handle can change — the host recreates the stream texture on a render
size change — so the fbo is rebuilt whenever `(texture, width, height)` differs
from the previous frame. The external texture wrapper itself is never released
(the underlying GL texture belongs to MXWendler); only the plugin's own fbo and
depth renderbuffer are.

Per-instance state is keyed by the integer `media_id`, which the host sets on the
module before each call. The host pushes the clip playback speed through
`onSetSpeed(speed)`, which the cube stores per instance and uses to scale its
rotation, so `0` freezes the spin and negative values spin it backwards.

### GL context

ModernGL attaches to MXWendler's own OpenGL context — `moderngl.create_context()`
is called lazily on the first `onRenderFrameGL`, where the host guarantees its
context is current, never a standalone context (see `plugin_opengl_cube`'s README
for why that would crash).

### Orientation

No y-flip: the host expects standard GL bottom-up content in plugin media
textures (row 0 = bottom of the image), which a plain GL render delivers as-is.
Earlier versions flipped Y in the projection matrix - that showed the render
upside down, unnoticeable on a symmetric cube. Drag the *Top Scale* slider to 0
(4-sided pyramid) and set the clip speed to 0 to verify: the apex must point up.

## Requirements

```
pip install moderngl numpy
```

## License

MIT
