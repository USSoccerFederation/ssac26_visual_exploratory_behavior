import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from mplsoccer import Pitch
import numpy as np

from .helpers import shoulder_angle_to_xy


def plot_matrix(grid, x, y, head_angle, matrix_type="fov", **kwargs):
    # Initialize a pitch
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=kwargs["pitch_length"],
        pitch_width=kwargs["pitch_width"],
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    # Add the arrowhead
    arrow_length = kwargs.get("arrow_length", 3)  # Length of the arrow
    arrow_dx = arrow_length * np.cos(head_angle)
    arrow_dy = arrow_length * np.sin(head_angle)
    # Draw the arrow using annotate
    ax.annotate(
        "",  # No text
        xy=(x + arrow_dx, y + arrow_dy),  # Arrow tip
        xytext=(x, y),  # Arrow tail
        arrowprops=dict(arrowstyle="->", color="red", lw=3),
    )
    extent = (0, kwargs["pitch_length"], 0, kwargs["pitch_width"])

    if matrix_type == "fov":
        # Plot the grid
        im = ax.imshow(
            grid * 0.5,
            origin="lower",
            extent=extent,
            cmap="Greys" if kwargs.get("cmap", None) is None else kwargs.get("cmap"),
            interpolation="bilinear",
            alpha=0.5,
        )
    elif matrix_type == "pvp":
        im = ax.imshow(
            grid * 0.5,
            origin="lower",
            extent=extent,
            cmap="hot_r" if kwargs.get("cmap", None) is None else kwargs.get("cmap"),
            interpolation="bilinear",
            alpha=0.5,
        )
    else:
        raise NotImplementedError()

    ax.scatter(x, y, color="red", s=100)  # Human's location
    if "x2s" in kwargs and "y2s" in kwargs:
        ax.scatter(
            kwargs["x2s"], kwargs["y2s"], color="blue", s=100
        )  # Human's location

        if "shoulder" in kwargs and isinstance(kwargs["shoulder"], list):
            for x, y, a in zip(kwargs["x2s"], kwargs["y2s"], kwargs["shoulder"]):
                s = shoulder_angle_to_xy(
                    x=x,
                    y=y,
                    angle=a,
                    human_width=kwargs["human_width"] * 4,
                    smooth_multiplier=kwargs["smoothing_multiplier"],
                )

                ax.plot(
                    [s["left_x"], s["right_x"]],
                    [s["left_y"], s["right_y"]],
                    color="blue",
                    linewidth=4,
                )

    plt.show()
