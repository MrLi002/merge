"""Headless textured OBJ rendering in CV coordinates and linear light.

OBJ vertex coordinates are interpreted as metres without rescaling. A local
bbox-centering transform is explicit in model_info. Output camera axes are
+X right, +Y down, +Z forward; integer coordinates denote pixel centres.
Texture sRGB is decoded before filtering, exposure integration or events.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def srgb_to_linear(value):
    value = np.asarray(value, dtype=np.float32)
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4).astype(np.float32)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _index(token, length):
    index = int(token)
    index = index-1 if index > 0 else length+index if index < 0 else -1
    if not 0 <= index < length:
        raise ValueError("OBJ index outside previously declared vertices/UVs")
    return index


def _load_obj(path):
    vertices, uvs, groups, materials, libraries = [], [], {}, {}, []
    material = "__default__"
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        fields = raw.partition("#")[0].split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "vt":
            uvs.append([float(v) for v in fields[1:3]])
        elif fields[0] == "mtllib":
            libraries.extend(path.parent / name for name in fields[1:])
        elif fields[0] == "usemtl":
            material = " ".join(fields[1:])
        elif fields[0] == "f":
            face = []
            for spec in fields[1:]:
                pair = spec.split("/")
                vi = _index(pair[0], len(vertices))
                ti = _index(pair[1], len(uvs)) if len(pair) > 1 and pair[1] else None
                face.append((vi, ti))
            if len(face) < 3:
                raise ValueError("OBJ face has fewer than three vertices")
            destination = groups.setdefault(material, [])
            for triangle in _triangulate(face, vertices):
                destination.extend(triangle)
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or not groups:
        raise ValueError("OBJ requires finite 3D vertices and triangle faces")
    for library in libraries:
        current = None
        for raw in library.read_text(encoding="utf-8-sig").splitlines():
            fields = raw.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "newmtl":
                current = materials.setdefault(" ".join(fields[1:]), {})
            elif current is not None and fields[0] == "Kd":
                current["diffuse"] = [float(v) for v in fields[1:4]]
            elif current is not None and fields[0] == "map_Kd":
                if fields[1].startswith("-"):
                    raise ValueError("MTL map_Kd options are unsupported; use a plain texture filename")
                current["texture"] = library.parent / " ".join(fields[1:]).strip('"')
    center = (vertices.min(axis=0)+vertices.max(axis=0))/2
    size = np.ptp(vertices, axis=0)
    drawables = []
    for name, entries in groups.items():
        packed = np.empty((len(entries), 5), dtype=np.float32)
        for row, (vi, ti) in enumerate(entries):
            packed[row, :3] = vertices[vi]-center
            packed[row, 3:] = uvs[ti] if ti is not None else [0., 0.]
        if not np.isfinite(packed).all():
            raise ValueError("OBJ has nonfinite UV coordinates")
        drawables.append((name, packed, materials.get(name, {})))
    return center, size, drawables, libraries, len(vertices)


def _triangulate(face, vertices):
    """Ear clipping supports simple convex/concave OBJ polygons."""
    if len(face) == 3:
        return [face]
    points = np.array([vertices[v] for v, _ in face])
    normal = np.cross(points-np.roll(points, 1, axis=0), points+np.roll(points, 1, axis=0)).sum(axis=0)
    xy = np.delete(points, np.argmax(np.abs(normal)), axis=1)
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    area = sum(cross(a, b) for a, b in zip(xy, np.roll(xy, -1, axis=0)))
    if abs(area) < 1e-20:
        raise ValueError("Degenerate OBJ polygon")
    orientation = 1 if area > 0 else -1
    remaining, result = list(range(len(face))), []
    while len(remaining) > 3:
        for j, b in enumerate(remaining):
            a, c = remaining[j-1], remaining[(j+1) % len(remaining)]
            A, B, C = xy[[a, b, c]]
            if orientation*cross(B-A, C-B) <= 1e-20:
                continue
            contained = any(all(orientation*cross(v-u, xy[k]-u) >= -1e-20
                                for u, v in ((A, B), (B, C), (C, A)))
                            for k in remaining if k not in (a, b, c))
            if not contained:
                result.append([face[a], face[b], face[c]])
                remaining.pop(j)
                break
        else:
            raise ValueError("Cannot triangulate self-intersecting or degenerate OBJ polygon")
    result.append([face[k] for k in remaining])
    return result


_VERTEX = """
#version 330
in vec3 in_position;
in vec2 in_uv;
uniform mat4 model_to_camera;
uniform mat4 projection;
out vec2 uv;
out float camera_z;
void main() {
    vec4 point = model_to_camera * vec4(in_position, 1.0);
    gl_Position = projection * point;
    uv = in_uv;
    camera_z = point.z;
}
"""

_FRAGMENT = """
#version 330
uniform sampler2D color_texture;
in vec2 uv;
in float camera_z;
layout(location=0) out vec4 rgba;
layout(location=1) out float metric_z;
void main() {
    rgba = vec4(texture(color_texture, uv).rgb, 1.0);
    metric_z = camera_z;
}
"""


class MeshRenderer:
    """GPU depth-tested, perspective-correct ambient textured rendering.

    Supersampling averages linear RGB. At silhouettes the depth is the nearest
    visible object subpixel and mask means at least one object subpixel; depth
    is never averaged with the 3-metre background. A standalone context is
    owned by this instance; use close() or a with statement to release it.
    """

    def __init__(self, obj_path, K, width, height, supersample=1):
        import moderngl

        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
               for v in (width, height, supersample)):
            raise ValueError("width, height and supersample must be positive integers")
        self.width, self.height, self.supersample = int(width), int(height), int(supersample)
        self.K = np.asarray(K, dtype=float).copy()
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all() or
                self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or
                not np.allclose(self.K[2], [0, 0, 1]) or
                self.K[0, 1] != 0 or self.K[1, 0] != 0):
            raise ValueError("K must be finite zero-skew pinhole intrinsics")
        self.obj_path = Path(obj_path).resolve()
        center, size, meshes, libraries, vertex_count = _load_obj(self.obj_path)
        self.original_center, self.size = center, size
        self.T_original_center = np.eye(4)
        self.T_original_center[:3, 3] = center
        self.near, self.background_depth = .005, 3.
        self._closed, self._resources = False, []
        self.ctx = moderngl.create_standalone_context(require=330)
        self.renderer = self.ctx.info["GL_RENDERER"]
        self._drawables = []
        texture_info = []
        try:
            self._program = self._keep(self.ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT))
            for name, packed, material in meshes:
                texture_path = material.get("texture")
                if texture_path is not None:
                    with Image.open(texture_path) as img:
                        pixels = srgb_to_linear(np.asarray(img.convert("RGB"), dtype=np.float32)/255.)
                    texture_info.append({"material": name, "path": str(texture_path.resolve()),
                                         "sha256": _hash(texture_path), "size": [pixels.shape[1], pixels.shape[0]]})
                else:
                    diffuse = np.asarray(material.get("diffuse", [.7, .7, .7]), dtype=np.float32)
                    if diffuse.shape != (3,) or not np.isfinite(diffuse).all() or np.any(diffuse < 0):
                        raise ValueError("MTL diffuse color must contain three nonnegative values")
                    pixels = np.clip(diffuse, 0, 1).reshape(1, 1, 3)
                # OBJ v=0 is bottom; image row zero is top.
                texture = self._keep(self.ctx.texture((pixels.shape[1], pixels.shape[0]), 3,
                                    np.ascontiguousarray(pixels[::-1]).tobytes(), dtype="f4"))
                texture.build_mipmaps()
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                texture.repeat_x = texture.repeat_y = False
                vbo = self._keep(self.ctx.buffer(packed.tobytes()))
                vao = self._keep(self.ctx.vertex_array(self._program, [(vbo, "3f 2f", "in_position", "in_uv")]))
                self._drawables.append((vao, texture))
            render_size = (self.width*self.supersample, self.height*self.supersample)
            self._color = self._keep(self.ctx.texture(render_size, 4, dtype="f4"))
            self._z = self._keep(self.ctx.texture(render_size, 1, dtype="f4"))
            self._depth = self._keep(self.ctx.depth_renderbuffer(render_size))
            self._fbo = self._keep(self.ctx.framebuffer([self._color, self._z], self._depth))
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            n, f = self.near, self.background_depth
            projection = np.array([[2*fx/width, 0, 2*(cx+.5)/width-1, 0],
                                   [0, -2*fy/height, 1-2*(cy+.5)/height, 0],
                                   [0, 0, (f+n)/(f-n), -2*f*n/(f-n)], [0, 0, 1, 0]], dtype=np.float32)
            self._program["projection"].write(projection.T.tobytes())
            self._program["color_texture"].value = 0
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE | moderngl.BLEND)
            self.ctx.depth_func = "<"
            # Fixed sensor-coordinate linear-light background; no moving light.
            sy, sx = np.indices((render_size[1], render_size[0]), dtype=np.float32)
            sx, sy = (sx+.5)/self.supersample-.5, (sy+.5)/self.supersample-.5
            gray = .18+.018*np.sin(sx*.041)*np.cos(sy*.037)
            self._background = np.stack([gray*.94, gray, gray*.88], axis=-1).astype(np.float32)
            self.model_info = {
                "obj_path": str(self.obj_path), "obj_sha256": _hash(self.obj_path),
                "raw_model_sha256": _hash(self.obj_path), "vertex_count": vertex_count,
                "triangle_count": sum(len(mesh)//3 for _, mesh, _ in meshes),
                "original_center": center.tolist(), "size": size.tolist(), "units": "m",
                "T_original_center": self.T_original_center.tolist(),
                "centering_convention": "X_original = X_centered + original_center; no scaling or axis reflection",
                "materials": [{"path": str(p.resolve()), "sha256": _hash(p)} for p in libraries],
                "textures": texture_info, "texture_input_color_space": "sRGB",
                "output_color_space": "linear RGB", "lighting": "ambient texture only",
                "texture_filter": "linear-light trilinear mipmap minification; bilinear magnification",
                "renderer": self.renderer, "opengl_version": self.ctx.info["GL_VERSION"],
                "supersample": self.supersample, "near_clip_m": self.near,
                "background_depth_m": self.background_depth,
                "pixel_convention": "integer centres; x right, y down, z forward",
                "supersample_depth_policy": "nearest visible object subpixel; object mask is any coverage",
            }
        except Exception:
            self.close()
            raise

    def _keep(self, resource):
        self._resources.append(resource)
        return resource

    def render(self, position, q, T_event_camera=None):
        """Render a centred-object pose in event coordinates into camera axes."""
        if self._closed:
            raise RuntimeError("Renderer is closed")
        p, q = np.asarray(position, dtype=float), np.asarray(q, dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or np.linalg.norm(q) < 1e-12:
            raise ValueError("Expected finite position and nonzero xyzw quaternion")
        model = np.eye(4)
        model[:3, :3], model[:3, 3] = Rotation.from_quat(q).as_matrix(), p
        if T_event_camera is not None:
            T = np.asarray(T_event_camera, dtype=float)
            if (T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]) or
                    not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6) or
                    not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-6)):
                raise ValueError("T_event_camera must be a rigid camera-to-event transform")
            inverse = np.eye(4)
            inverse[:3, :3] = T[:3, :3].T
            inverse[:3, 3] = -T[:3, :3].T @ T[:3, 3]
            model = inverse @ model
        h, w, s = self.height, self.width, self.supersample
        with self.ctx:
            self._program["model_to_camera"].write(model.astype(np.float32).T.tobytes())
            self._fbo.use()
            self._fbo.clear(0., 0., 0., 0., depth=1.)
            for vao, texture in self._drawables:
                texture.use(location=0)
                vao.render()
            rgba = np.frombuffer(self._fbo.read(components=4, attachment=0, dtype="f4"), dtype=np.float32).reshape(h*s, w*s, 4)[::-1]
            z = np.frombuffer(self._fbo.read(components=1, attachment=1, dtype="f4"), dtype=np.float32).reshape(h*s, w*s)[::-1]
        mask = rgba[..., 3] > .5
        rgb = np.where(mask[..., None], rgba[..., :3], self._background)
        z = np.where(mask, z, self.background_depth)
        if s != 1:
            rgb = rgb.reshape(h, s, w, s, 3).mean(axis=(1, 3))
            z = z.reshape(h, s, w, s).min(axis=(1, 3))
            mask = mask.reshape(h, s, w, s).any(axis=(1, 3))
        return np.ascontiguousarray(rgb, dtype=np.float32), np.ascontiguousarray(z, dtype=np.float32), np.ascontiguousarray(mask)

    def close(self):
        if not self._closed:
            with self.ctx:
                for resource in reversed(self._resources):
                    resource.release()
            self._resources.clear()
            self.ctx.release()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
