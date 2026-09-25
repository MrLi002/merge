"""Seeded smooth 6-DoF motion for synthetic dataset generation only.

The paper describes randomized regular/fast trajectories but does not release
their parameters. These analytic paths are reproducible engineering choices,
not the original paper's trajectories. The runtime tracker must not import
this module or use its velocities as observations.

Pose is T_event_object (metres, xyzw quaternion). Velocity is a spatial twist
in event-camera axes: Xdot = vo + omega cross X. In particular, vo differs
from the velocity of the object's origin when the object rotates.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation


class MotionTrajectory:
    """Two smooth harmonics per translation/rotation axis, sampled by seed.

    Amplitudes bound the absolute displacement from the t=0 position and the
    absolute excursion of each elementary Euler angle from t=0. They do not
    bound the SO(3) geodesic angle of the composed rotation. The initial pose
    is [0, 0, center_depth] and Exp(base_rotation). At arbitrary time:

        R(t) = Rz(angle_z) Ry(angle_y) Rx(angle_x) R_base.

    "fast" evaluates the identical sampled path with a faster clock; it does
    not draw new parameters. Angles, positions and their derivatives are
    continuous, including at t=0. Scalar negative times are supported to
    permit central-difference verification at the start of a sequence.
    """

    def __init__(self, seed=7, speed="regular", center_depth=.75,
                 translation_amplitude=(.08, .06, .04),
                 rotation_amplitude=(.3, .4, .4), speed_factor=3.,
                 base_rotation=(.1, -.2, .1)):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if speed not in ("regular", "fast"):
            raise ValueError("speed must be 'regular' or 'fast'")
        self.seed, self.speed = int(seed), speed
        self.center_depth, self.speed_factor = float(center_depth), float(speed_factor)
        if (not np.isfinite([self.center_depth, self.speed_factor]).all() or
                self.center_depth <= 0 or self.speed_factor <= 1):
            raise ValueError("center_depth must be positive and speed_factor must exceed one")
        self.translation_amplitude = self._vector(translation_amplitude, "translation_amplitude", nonnegative=True)
        self.rotation_amplitude = self._vector(rotation_amplitude, "rotation_amplitude", nonnegative=True)
        self.base_rotation = self._vector(base_rotation, "base_rotation")
        if self.center_depth <= self.translation_amplitude[2]:
            raise ValueError("center_depth must exceed the Z translation amplitude")
        self.time_scale = 1. if self.speed == "regular" else self.speed_factor
        self._base = Rotation.from_rotvec(self.base_rotation)
        rng = np.random.default_rng(self.seed)
        self._translation_frequency = rng.uniform(.30, .85, size=(3, 2))
        self._rotation_frequency = rng.uniform(.25, .70, size=(3, 2))
        self._translation_phase = rng.uniform(-np.pi, np.pi, size=(3, 2))
        self._rotation_phase = rng.uniform(-np.pi, np.pi, size=(3, 2))
        # Positive weights summing to one and a factor 1/2 give strict bounds
        # even after subtracting the initial sine value to start at zero.
        self._weights = np.array([.7, .3])
        self._translation_omega = 2*np.pi*self._translation_frequency
        self._rotation_omega = 2*np.pi*self._rotation_frequency
        self._translation_initial = np.sin(self._translation_phase)
        self._rotation_initial = np.sin(self._rotation_phase)

    @staticmethod
    def _vector(value, name, nonnegative=False):
        array = np.asarray(value, dtype=float)
        if array.shape != (3,) or not np.isfinite(array).all() or (nonnegative and np.any(array < 0)):
            raise ValueError(f"{name} must contain three finite" + (" nonnegative values" if nonnegative else " values"))
        return array.copy()

    def _time(self, t):
        value = np.asarray(t, dtype=float)
        if value.shape != () or not np.isfinite(value):
            raise ValueError("t must be one finite scalar timestamp in seconds")
        scaled = float(value)*self.time_scale
        if not np.isfinite(scaled) or abs(scaled) > np.finfo(float).max/(2*np.pi):
            raise ValueError("Scaled timestamp exceeds floating-point range")
        return scaled

    def _harmonics(self, tau, amplitude, omega, phase, initial):
        argument = omega*tau+phase
        value = .5*amplitude*((np.sin(argument)-initial) @ self._weights)
        rate = .5*amplitude*((np.cos(argument)*omega) @ self._weights)*self.time_scale
        return value, rate

    def _evaluate(self, t):
        tau = self._time(t)
        offset, linear_rate = self._harmonics(tau, self.translation_amplitude,
                                              self._translation_omega, self._translation_phase,
                                              self._translation_initial)
        angles, angle_rate = self._harmonics(tau, self.rotation_amplitude,
                                             self._rotation_omega, self._rotation_phase,
                                             self._rotation_initial)
        position = offset + [0., 0., self.center_depth]
        return position, angles, linear_rate, angle_rate

    def pose(self, t):
        """Return position (metres) and unit xyzw orientation at scalar t."""
        position, angles, _, _ = self._evaluate(t)
        # SciPy lowercase xyz means extrinsic rotations, hence Rz Ry Rx.
        orientation = Rotation.from_euler("xyz", angles)*self._base
        return position, orientation.as_quat()

    def velocity(self, t):
        """Return analytic [vo, omega] in event axes (m/s and rad/s).

        For Rz Ry Rx R_base, omega = zdot ez + ydot Rz ey +
        xdot Rz Ry ex. The constant right-multiplied base does not change
        spatial angular velocity. vo = position_dot - omega cross position.
        """
        position, angles, position_rate, angle_rate = self._evaluate(t)
        _, ay, az = angles
        dx, dy, dz = angle_rate
        sy, cy, sz, cz = np.sin(ay), np.cos(ay), np.sin(az), np.cos(az)
        omega = np.array([dx*cz*cy-dy*sz, dx*sz*cy+dy*cz, dz-dx*sy])
        vo = position_rate-np.cross(omega, position)
        return np.concatenate([vo, omega])

    @property
    def metadata(self):
        """JSON-safe parameters, including all random draws and conventions."""
        return self.to_dict()

    def to_dict(self):
        return {
            "type": "seeded_two_harmonic_6dof",
            "version": 1,
            "purpose": "Independent synthetic motion; not the unpublished paper trajectories",
            "seed": self.seed,
            "rng": "numpy.random.default_rng/PCG64",
            "speed": self.speed,
            "speed_factor": self.speed_factor,
            "time_scale": self.time_scale,
            "center_depth": self.center_depth,
            "translation_amplitude": self.translation_amplitude.tolist(),
            "rotation_amplitude": self.rotation_amplitude.tolist(),
            "base_rotation": self.base_rotation.tolist(),
            "translation_frequency_hz": self._translation_frequency.tolist(),
            "rotation_frequency_hz": self._rotation_frequency.tolist(),
            "translation_phase_rad": self._translation_phase.tolist(),
            "rotation_phase_rad": self._rotation_phase.tolist(),
            "harmonic_weights": self._weights.tolist(),
            "amplitude_factor": .5,
            "harmonic_formula": "a/2 * sum_j w_j * (sin(2*pi*f_j*time_scale*t+phase_j)-sin(phase_j))",
            "rotation_composition": "Rz(angle_z) @ Ry(angle_y) @ Rx(angle_x) @ Exp(base_rotation)",
            "pose_convention": "T_event_object; quaternion_xyzw",
            "velocity_convention": "Xdot=vo+omega_cross_X; vo=position_dot-omega_cross_position; event axes",
            "units": {"time": "s", "position": "m", "angle": "rad",
                      "linear_velocity": "m/s", "angular_velocity": "rad/s", "frequency": "Hz"},
        }

    @classmethod
    def from_dict(cls, metadata):
        """Restore from recorded draws without relying on RNG version behavior."""
        data = deepcopy(metadata)
        if data.get("type") != "seeded_two_harmonic_6dof" or data.get("version") != 1:
            raise ValueError("Unsupported trajectory metadata type/version")
        result = cls(**{key: data[key] for key in ("seed", "speed", "center_depth",
                      "translation_amplitude", "rotation_amplitude", "speed_factor", "base_rotation")})
        for name in ("translation_frequency_hz", "rotation_frequency_hz", "translation_phase_rad", "rotation_phase_rad"):
            array = np.asarray(data[name], dtype=float)
            if array.shape != (3, 2) or not np.isfinite(array).all() or ("frequency" in name and np.any(array <= 0)):
                raise ValueError(f"Invalid trajectory metadata: {name}")
            attribute = "_"+name.removesuffix("_hz").removesuffix("_rad")
            setattr(result, attribute, array.copy())
        weights = np.asarray(data["harmonic_weights"], dtype=float)
        if (weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights < 0) or
                not np.isclose(weights.sum(), 1.) or data.get("amplitude_factor") != .5 or
                data.get("time_scale") != result.time_scale):
            raise ValueError("Invalid harmonic weights, scale, or amplitude factor")
        result._weights = weights.copy()
        result._translation_omega = 2*np.pi*result._translation_frequency
        result._rotation_omega = 2*np.pi*result._rotation_frequency
        result._translation_initial = np.sin(result._translation_phase)
        result._rotation_initial = np.sin(result._rotation_phase)
        return result
