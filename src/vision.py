from dataclasses import dataclass
from typing import List, Union, Literal

import numpy as np

from .helpers import (
    get_grid,
    get_meshgrid,
    get_meshgrid_linspace,
    get_empty_grid,
)

from .plot import plot_matrix


@dataclass
class Vision:
    x: float
    y: float
    head_angle: float
    speed: float
    pitch_length: Union[float, int] = 100
    pitch_width: Union[float, int] = 64
    max_occlusion_alpha: float = 0.7
    wedge_radius: int = 150
    vision_angle: int = 120
    human_width: float = 1.8
    smoothing_multiplier: float = 1.0

    def __post_init__(self):
        self.human_width = self.human_width * self.smoothing_multiplier
        self.human_depth = self.human_width / 2.0

        grid_length = self.pitch_length * self.smoothing_multiplier
        grid_width = self.pitch_width * self.smoothing_multiplier

        self.radius = self.wedge_radius * self.smoothing_multiplier
        self.vision_radians = self.vision_angle * (np.pi / 360)

        self.grid_x = get_grid(v=self.x, dim=self.pitch_length, grid_dim=grid_length)
        self.grid_y = get_grid(v=self.y, dim=self.pitch_width, grid_dim=grid_width)

        self.y_idxs, self.x_idxs = get_meshgrid(
            grid_length=grid_length, grid_width=grid_width
        )
        self.x_idxs_ls, self.y_idxs_ls = get_meshgrid_linspace(
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
            grid_length=grid_length,
            grid_width=grid_width,
        )
        self.grid = get_empty_grid(grid_length=grid_length, grid_width=grid_width)
        self.set_grid_dxdy()
        self.set_self_dxdy()
        self.set_self_grid_dxdy()

    @staticmethod
    def __distance(dx, dy):
        return np.sqrt(dx**2 + dy**2)

    @staticmethod
    def __angle(dx, dy):
        return np.arctan2(dy, dx)

    def __theta(self, angle, player_angle):
        if isinstance(player_angle, np.ndarray):
            player_angle = player_angle[:, np.newaxis, np.newaxis]
        return np.arctan2(np.sin(angle - player_angle), np.cos(angle - player_angle))

    def __datm(self, dx, dy, player_angle):
        distance = self.__distance(dx, dy)
        angle = self.__angle(dx, dy)
        theta = self.__theta(angle, player_angle)

        mask = (distance <= self.radius) & (np.abs(theta) <= self.vision_radians)
        return distance, angle, theta, mask

    def set_grid_dxdy(self):
        self.grid_dx = self.x_idxs - self.grid_x
        self.grid_dy = self.y_idxs - self.grid_y

    def set_self_dxdy(self):
        self.self_dx = self.x_idxs_ls - self.x
        self.self_dy = self.y_idxs_ls - self.y

    def set_self_grid_dxdy(self):
        self.self_grid_dx = self.x_idxs_ls - self.grid_x
        self.self_grid_dy = self.y_idxs_ls - self.grid_y

    def _head_angle_difference(self, angle):
        return self.__datm(self.grid_dx, self.grid_dy, angle)

    def _inter_player_angle_difference(self, angle):
        _, _, theta_diff_ij, _ = self.__datm(self.self_dx, self.self_dy, angle)
        return theta_diff_ij

    @staticmethod
    def _gaussian(c, a, sigma):
        return np.exp(-c * (a / sigma) ** 2)

    @staticmethod
    def _normalize(m):
        if len(m.shape) == 3:
            return m / np.max(m, axis=(1, 2), keepdims=True)
        elif len(m.shape) == 2:
            return m / np.max(m)
        else:
            raise NotImplementedError()

    def binary_field_of_view(self):
        player_grid = np.copy(self.grid)

        self.distance, self.angle_head, self.theta_head, self.mask_head = (
            self._head_angle_difference(angle=self.head_angle)
        )
        player_grid[self.mask_head] = 1

        self.fov_b = player_grid
        return self.fov_b

    def field_of_view(self, sigma_r=75, sigma_a=0.4):
        def __radial_scalar(v1):
            return np.min([0.25 * v1 + 0.1, 2.6])

        def __angular_scalar(v1):
            return np.min([0.03 * v1 + 0.2, 0.5])

        if not hasattr(self, "fov_b"):
            self.binary_field_of_view()

        # Generate the radial Gaussian decay
        sigma_r = (
            sigma_r * self.smoothing_multiplier
        )  # Standard deviation for radial Gaussian
        cr = __radial_scalar(self.speed)
        radial_gaussian = self._gaussian(cr, self.distance, sigma_r)

        ca = __angular_scalar(self.speed)
        angular_gaussian = self._gaussian(ca, self.theta_head, sigma_a)

        # Combine both Gaussians and apply the wedge mask
        pvp = radial_gaussian * angular_gaussian * self.fov_b
        self.fov = self._normalize(pvp)
        return self.fov

    def _occlusion_vec(self, x2s, y2s, shoulder_angles, dist_ij):
        # Helper function to compute shoulder locations
        def __shoulder_location(c, func, perp_angle):
            return c + (self.human_width / (2.0 * self.smoothing_multiplier)) * func(
                perp_angle
            )

        # Compute perpendicular angles for left and right shoulders
        perp_angle_left = shoulder_angles + np.pi / 2  # Shape: (6,)
        perp_angle_right = shoulder_angles - np.pi / 2  # Shape: (6,)

        # Compute shoulder coordinates for all players
        shoulder_left_x = __shoulder_location(
            x2s, np.cos, perp_angle_left
        )  # Shape: (6,)
        shoulder_left_y = __shoulder_location(
            y2s, np.sin, perp_angle_left
        )  # Shape: (6,)
        shoulder_right_x = __shoulder_location(
            x2s, np.cos, perp_angle_right
        )  # Shape: (6,)
        shoulder_right_y = __shoulder_location(
            y2s, np.sin, perp_angle_right
        )  # Shape: (6,)

        # Observer's position
        observer_x, observer_y = self.x, self.y

        # Compute distances from observer to shoulders for all players
        dist_ij_shoulder_left = self.__distance(
            dx=shoulder_left_x - observer_x, dy=shoulder_left_y - observer_y
        )  # Shape: (6,)
        dist_ij_shoulder_right = self.__distance(
            dx=shoulder_right_x - observer_x, dy=shoulder_right_y - observer_y
        )  # Shape: (6,)

        # Minimum distance (element-wise) for all players
        min_dist_ij = np.minimum.reduce(
            [dist_ij, dist_ij_shoulder_left, dist_ij_shoulder_right]
        )  # Shape: (6,)

        distance = self.__distance(
            dx=self.x_idxs_ls - observer_x, dy=self.y_idxs_ls - observer_y
        )  # Shape: (68, 105)

        # Expand distance grid to match player axis
        distance = distance[np.newaxis, :, :]  # Shape: (1, 68, 105)

        # Expand min_dist_ij for broadcasting
        min_dist_ij = min_dist_ij[:, np.newaxis, np.newaxis]  # Shape: (6, 1, 1)

        # Compute shadow mask for all players
        shadow_mask = (distance > min_dist_ij) & (
            distance <= np.inf
        )  # Shape: (6, 68, 105)

        # Initialize player masks and apply shadow mask
        player_masks = np.zeros(
            (x2s.shape[0], *self.grid.shape)
        )  # Shape: (6, grid.shape[0], grid.shape[1])
        player_masks[shadow_mask] = self.max_occlusion_alpha

        return player_masks

    def _apparent_width_vec(self, x2s, y2s, angles):
        half_width = self.human_width / 2
        half_depth = self.human_depth / 2

        # Define local corners (4 corners of the rectangle)
        local_corners = np.array(
            [
                [half_width, half_depth],
                [-half_width, half_depth],
                [-half_width, -half_depth],
                [half_width, -half_depth],
            ]
        )  # Shape: (4, 2)

        # Create rotation matrices for all angles
        cos_angles = np.cos(angles)  # Shape: (6,)
        sin_angles = np.sin(angles)  # Shape: (6,)
        rotation_matrices = np.array(
            [[[cos, -sin], [sin, cos]] for cos, sin in zip(cos_angles, sin_angles)]
        )  # Shape: (6, 2, 2)

        # Rotate and translate corners for each (x2, y2, angle)
        # Use broadcasting to handle the rotation and translation
        global_corners = (
            np.einsum("nij,kj->nki", rotation_matrices, local_corners)
            + np.array([x2s, y2s]).T[:, np.newaxis, :]
        )
        # global_corners shape: (6, 4, 2)

        # Observer position
        observer_position = np.array([self.x, self.y])  # Shape: (2,)

        # Compute vectors from observer to global corners
        vectors = global_corners - observer_position[np.newaxis, np.newaxis, :]
        # vectors shape: (6, 4, 2)

        # Get vector pairs for angles calculation
        v1 = vectors[:, [0, 1], :]  # Shape: (6, 2, 2)
        v2 = vectors[:, [2, 3], :]  # Shape: (6, 2, 2)

        # Dot products and magnitudes for all pairs
        dot_products = np.sum(v1 * v2, axis=2)  # Shape: (6, 2)
        magnitudes = np.linalg.norm(v1, axis=2) * np.linalg.norm(
            v2, axis=2
        )  # Shape: (6, 2)

        # Compute angles and find the maximum
        angles_between = np.arccos(dot_products / (magnitudes + 1e-10))  # Shape: (6, 2)
        max_angles = np.max(angles_between, axis=1)  # Shape: (6,)

        return max_angles

    def _occlusion_profile(self, cs, alpha, sigma_a=0.4):
        if isinstance(cs, np.ndarray):
            cs = cs[:, np.newaxis, np.newaxis]
        angular_gaussian = self._gaussian(cs, alpha, sigma_a)
        return self._normalize(angular_gaussian)

    def occlusions(
        self,
        x2s: np.array,
        y2s: np.array,
        shoulder_angles: np.array,
        sigma_a: float = 0.2,
    ):
        if isinstance(x2s, list):
            x2s = np.asarray(x2s)
        if isinstance(y2s, list):
            y2s = np.asarray(y2s)
        if isinstance(shoulder_angles, list):
            shoulder_angles = np.asarray(shoulder_angles)

        theta_ij = self.__angle(dx=x2s - self.x, dy=y2s - self.y)
        dist_ij = self.__distance(dx=x2s - self.x, dy=y2s - self.y)
        theta_diff_ij = self._inter_player_angle_difference(angle=theta_ij)
        omega_alpha = self._apparent_width_vec(x2s, y2s, shoulder_angles)

        c_s = 1 / (omega_alpha / (dist_ij + 1e-10))

        occlusion_profile = self._occlusion_profile(
            cs=c_s, alpha=theta_diff_ij, sigma_a=sigma_a
        )
        player_mask = self._occlusion_vec(x2s, y2s, shoulder_angles, dist_ij)

        player_masks = 1 - (occlusion_profile * player_mask)

        self.occs = np.prod(player_masks, axis=0)
        return self.occs

    def compute(self, x2s: np.array, y2s: np.array, shoulder_angles: np.array):
        fov = self.field_of_view()
        occs = self.occlusions(x2s, y2s, shoulder_angles)
        self.vision = fov * occs
        return self.vision

    def plot(self, plot_type: Literal["fov_b", "fov", "occ", "vision"], **kwargs):
        if plot_type == "fov_b":
            matrix = self.binary_field_of_view()
            matrix_type = "fov"
        elif plot_type == "fov":
            matrix = self.field_of_view()
            matrix_type = "pvp"
        elif plot_type == "occ":
            matrix = self.occlusions(
                x2s=kwargs["x2s"], y2s=kwargs["y2s"], shoulder_angles=kwargs["a2s"]
            )
            matrix_type = "pvp"
        elif plot_type == "vision":
            matrix = self.compute(
                x2s=kwargs["x2s"], y2s=kwargs["y2s"], shoulder_angles=kwargs["a2s"]
            )
            matrix_type = "pvp"

        plot_matrix(
            grid=matrix,
            x=self.x,
            y=self.y,
            head_angle=self.head_angle,
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
            matrix_type=matrix_type,
            x2s=kwargs.get("x2s"),
            y2s=kwargs.get("y2s"),
            shoulder=kwargs.get("a2s"),
            human_width=self.human_width,
            smoothing_multiplier=self.smoothing_multiplier,
        )
