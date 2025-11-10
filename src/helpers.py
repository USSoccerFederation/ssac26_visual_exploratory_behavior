import numpy as np


def get_grid(v, dim, grid_dim):
    return int((v / dim) * grid_dim)


def get_meshgrid(grid_length, grid_width):
    y_idxs, x_idxs = np.meshgrid(
        np.arange(grid_width), np.arange(grid_length), indexing="ij"
    )
    return y_idxs, x_idxs


def get_meshgrid_linspace(pitch_length, pitch_width, grid_length, grid_width):
    x_space = np.linspace(0, pitch_length, int(grid_length))
    y_space = np.linspace(0, pitch_width, int(grid_width))
    return np.meshgrid(x_space, y_space)


def get_empty_grid(grid_length, grid_width):
    return np.zeros((int(grid_width), int(grid_length)))


def shoulder_angle_to_xy(x, y, angle, human_width, smooth_multiplier):
    perp_angle_left = angle + np.pi / 2
    perp_angle_right = angle - np.pi / 2

    shoulder_left_x = x + (human_width / (2.0 * smooth_multiplier)) * np.cos(
        perp_angle_left
    )
    shoulder_left_y = y + (human_width / (2.0 * smooth_multiplier)) * np.sin(
        perp_angle_left
    )

    shoulder_right_x = x + (human_width / (2.0 * smooth_multiplier)) * np.cos(
        perp_angle_right
    )
    shoulder_right_y = y + (human_width / (2.0 * smooth_multiplier)) * np.sin(
        perp_angle_right
    )

    return {
        "left_x": shoulder_left_x,
        "left_y": shoulder_left_y,
        "right_x": shoulder_right_x,
        "right_y": shoulder_right_y,
    }
