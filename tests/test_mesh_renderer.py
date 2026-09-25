"""GPU projection/depth/color checks against analytic geometry."""
import hashlib

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("moderngl")
from ev6d.mesh_renderer import MeshRenderer, srgb_to_linear


K = np.array([[100., 0., 31.], [0., 110., 23.], [0., 0., 1.]])
IDENTITY = [0., 0., 0., 1.]


def plane(tmp_path, shifted=(0., 0., 0.), depth_slope=0., negative=False):
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    texture[:4, :4] = [255, 0, 0]
    texture[:4, 4:] = [0, 255, 0]
    texture[4:, :4] = [0, 0, 255]
    texture[4:, 4:] = [128, 128, 128]
    Image.fromarray(texture).save(tmp_path / "checker.png")
    (tmp_path / "plane.mtl").write_text("newmtl checker\nmap_Kd checker.png\n", encoding="utf-8")
    verts = np.array([[-.1, -.1, -.1*depth_slope], [.1, -.1, .1*depth_slope],
                      [.1, .1, .1*depth_slope], [-.1, .1, -.1*depth_slope]])+shifted
    lines = ["mtllib plane.mtl", "usemtl checker"]
    lines += ["v "+" ".join(map(str, v)) for v in verts]
    lines += ["vt 0 1", "vt 1 1", "vt 1 0", "vt 0 0"]
    lines.append("f -4/-4 -3/-3 -2/-2 -1/-1" if negative else "f 1/1 2/2 3/3 4/4")
    path = tmp_path / "plane.obj"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_projection_metric_scale_centering_uv_and_linear_color(tmp_path):
    path = plane(tmp_path, shifted=(10., 20., 30.), negative=True)
    with MeshRenderer(path, K, 64, 48) as renderer:
        rgb, z, mask = renderer.render([0., 0., 1.], IDENTITY)
        np.testing.assert_allclose(renderer.original_center, [10., 20., 30.])
        np.testing.assert_allclose(renderer.size, [.2, .2, 0.], atol=1e-12)
        np.testing.assert_allclose(renderer.T_original_center[:3, 3], [10., 20., 30.])
        assert renderer.model_info["raw_model_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert rgb.dtype == z.dtype == np.float32 and mask.dtype == bool
        assert mask.sum() == 20*22
        np.testing.assert_allclose(z[mask], 1., atol=1e-6)
        np.testing.assert_array_equal(z[~mask], 3.)
        np.testing.assert_allclose(rgb[18, 26], [1., 0., 0.], atol=1e-6)
        np.testing.assert_allclose(rgb[18, 36], [0., 1., 0.], atol=1e-6)
        np.testing.assert_allclose(rgb[28, 26], [0., 0., 1.], atol=1e-6)
        expected_gray = ((128/255.+.055)/1.055)**2.4
        np.testing.assert_allclose(rgb[28, 36], expected_gray, atol=1e-6)


def test_positive_cv_axes_translate_right_and_down_and_extrinsic_is_inverted(tmp_path):
    path = plane(tmp_path)
    with MeshRenderer(path, K, 64, 48) as renderer:
        _, _, mask = renderer.render([0., 0., 1.], IDENTITY)
        _, _, moved = renderer.render([.03, .02, 1.], IDENTITY)
        centroid = lambda m: np.column_stack(np.nonzero(m)).mean(axis=0)
        np.testing.assert_allclose(centroid(moved)-centroid(mask), [2., 3.], atol=.25)
        T = np.eye(4)
        T[:3, 3] = [.03, .02, 0.]
        rgb_a, z_a, mask_a = renderer.render([.03, .02, 1.], IDENTITY, T_event_camera=T)
        rgb_b, z_b, mask_b = renderer.render([0., 0., 1.], IDENTITY)
        np.testing.assert_array_equal(mask_a, mask_b)
        np.testing.assert_allclose(rgb_a, rgb_b, atol=1e-6)
        np.testing.assert_allclose(z_a, z_b, atol=1e-6)


def test_tilted_plane_z_is_perspective_correct_and_integer_pixel_centered(tmp_path):
    path = plane(tmp_path, depth_slope=.7)
    with MeshRenderer(path, K, 64, 48) as renderer:
        rgb, z, mask = renderer.render([0., 0., 1.], IDENTITY)
        y, x = np.nonzero(mask)
        # Plane camera equation Z=1+.7 X, with X=(u-cx)*Z/fx.
        expected = 1/(1-.7*(x-K[0, 2])/K[0, 0])
        # Hardware triangle setup quantizes subpixel vertex positions. Allow
        # 20 micrometres here; a half-pixel centre error would exceed 3 mm.
        np.testing.assert_allclose(z[y, x], expected, atol=2e-5)
        assert z[23, 31] == pytest.approx(1., abs=2e-5)
        # Independent ray/plane UV and bilinear reference. Affine screen-space
        # UV interpolation would warp the checker on this tilted plane.
        X, Y = (x-K[0, 2])*expected/K[0, 0], (y-K[1, 2])*expected/K[1, 1]
        tx, ty = np.clip((X+.1)/.2*8-.5, 0, 7), np.clip((Y+.1)/.2*8-.5, 0, 7)
        x0, y0 = np.floor(tx).astype(int), np.floor(ty).astype(int)
        x1, y1 = np.minimum(x0+1, 7), np.minimum(y0+1, 7)
        ax, ay = (tx-x0)[:, None], (ty-y0)[:, None]
        with Image.open(tmp_path / "checker.png") as im:
            pixels = srgb_to_linear(np.asarray(im)/255.)
        expected_rgb = ((1-ay)*((1-ax)*pixels[y0, x0]+ax*pixels[y0, x1])+
                        ay*((1-ax)*pixels[y1, x0]+ax*pixels[y1, x1]))
        # GPU bilinear weights can be quantized to 8 fractional bits.
        np.testing.assert_allclose(rgb[y, x], expected_rgb, atol=1/256)


def test_occlusion_depth_test_works_independent_of_face_order(tmp_path):
    # Two overlapping quads at local Z +/-.1. Far red is drawn after near green.
    (tmp_path / "two.mtl").write_text("newmtl green\nKd 0 1 0\nnewmtl red\nKd 1 0 0\n")
    lines = ["mtllib two.mtl"]
    for z in (-.1, .1):
        lines += [f"v {x} {y} {z}" for x, y in ((-.1, -.1), (.1, -.1), (.1, .1), (-.1, .1))]
    lines += ["usemtl green", "f 1 2 3 4", "usemtl red", "f 5 6 7 8"]
    path = tmp_path / "two.obj"
    path.write_text("\n".join(lines))
    with MeshRenderer(path, K, 64, 48) as renderer:
        rgb, z, mask = renderer.render([0., 0., 1.], IDENTITY)
        assert mask[23, 31]
        assert z[23, 31] == pytest.approx(.9, abs=1e-6)
        np.testing.assert_allclose(rgb[23, 31], [0., 1., 0.], atol=1e-6)
        _, background_z, background_mask = renderer.render([0., 0., 4.], IDENTITY)
        assert not background_mask.any()
        np.testing.assert_array_equal(background_z, 3.)


def test_rotation_and_camera_transform_preserve_handedness(tmp_path):
    path = plane(tmp_path)
    rotation = Rotation.from_rotvec([.16, -.22, .13])
    position = np.array([.04, -.02, 1.])
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = position-rotation.apply([0., 0., 1.])
    with MeshRenderer(path, K, 64, 48) as renderer:
        a = renderer.render(position, rotation.as_quat(), T_event_camera=T)
        b = renderer.render([0., 0., 1.], IDENTITY)
        np.testing.assert_array_equal(a[2], b[2])
        np.testing.assert_allclose(a[0], b[0], atol=1e-6)
        np.testing.assert_allclose(a[1], b[1], atol=1e-6)
        T[0, :3] *= -1
        with pytest.raises(ValueError, match="rigid"):
            renderer.render(position, rotation.as_quat(), T_event_camera=T)


def test_supersampling_and_multiple_contexts_and_close(tmp_path):
    path = plane(tmp_path)
    with MeshRenderer(path, K, 64, 48) as first, MeshRenderer(path, K, 64, 48, supersample=2) as second:
        for renderer in (first, second, first):
            rgb, z, mask = renderer.render([.001, .001, 1.], IDENTITY)
            assert rgb.shape == (48, 64, 3) and mask.sum() > 400
            np.testing.assert_allclose(z[mask], 1., atol=1e-6)
    first.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.render([0, 0, 1], IDENTITY)


def test_srgb_thresholds_match_standard():
    np.testing.assert_allclose(srgb_to_linear([0., .04045, .5, 1.]),
                               [0., .04045/12.92, ((.5+.055)/1.055)**2.4, 1.], atol=1e-7)


def test_concave_polygon_is_not_filled_by_naive_fan_triangulation(tmp_path):
    xy = [(-.1, -.1), (.1, -.1), (.1, .1), (.02, .1),
          (.02, 0.), (-.02, 0.), (-.02, .1), (-.1, .1)]
    path = tmp_path / "concave.obj"
    path.write_text("\n".join([f"v {x} {y} 0" for x, y in xy]+["f 1 2 3 4 5 6 7 8"]))
    with MeshRenderer(path, K, 64, 48) as renderer:
        _, _, mask = renderer.render([0., 0., 1.], IDENTITY)
        assert mask[18, 31]
        assert not mask[29, 31]
        assert mask[29, 25] and mask[29, 37]


def test_raw_obj_units_are_never_inferred_or_rescaled(tmp_path):
    path = tmp_path / "large.obj"
    path.write_text("v -100 -100 0\nv 100 -100 0\nv 100 100 0\nv -100 100 0\nf 1 2 3 4\n")
    with MeshRenderer(path, K, 64, 48) as renderer:
        np.testing.assert_array_equal(renderer.size, [200., 200., 0.])
