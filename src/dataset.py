import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from typing import List, Optional, Literal, Union

from dataclasses import dataclass

from .helpers import *
from .colors import TEAM_COLORS, DEFAULT_CMAP, DEFAULT_NORM
from .vision import Vision
from .schema import schema

from mplsoccer import Pitch
import matplotlib.patheffects as path_effects


class AngleType:
    HEAD = "head_angle"
    HIP = "hips_angle"
    SHOULDER_DIRECTION = "shoulders_dir"
    SHOULDER_PLOT_ANGLE = "shoulders_plot_angle"
    RUN = "run_angle"


def rotate_angle(angle, by):
    new_angle = angle + by
    return (new_angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Dataset:
    parquet_file: str
    pitch_width: int = PITCH_WIDTH
    pitch_length: int = PITCH_LENGTH
    pitch_margin: int = 2
    fps: int = FPS

    def __post_init__(self):
        self.data = pl.read_parquet(self.parquet_file)
        schema.validate(self.data)

    @property
    def players(self):
        return (self.data
            .select(['player', 'player_id', 'team_id'])
            .unique()
            .sort(['team_id', 'player_id'])
        )

    def __plot_frame(self, d: dict):

        def __angled_bar(
            x, y, ax, angle, color, line_length=3, start_point=1, end_point=100
        ):
            x_start = x - (line_length / start_point) * np.cos(angle)
            y_start = y - (line_length / start_point) * np.sin(angle)
            x_end = x + (line_length / end_point) * np.cos(angle)
            y_end = y + (line_length / end_point) * np.sin(angle)

            ax.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=5)
            return ax

        self.ax.clear()
        self.pitch.draw(ax=self.ax)

        arrow_multiplier = 1.0
        smoothing_multiplier: int = 3
        new_shape = (
            self.pitch_width * smoothing_multiplier,
            self.pitch_length * smoothing_multiplier,
        )

        frame_id = d["frame_id"]
        game_id = d["game_id"]
        period_id = d["period_id"]

        segment = self.segment[
            (self.segment["frame_id"] == frame_id)
            & (self.segment["game_id"] == game_id)
            & (self.segment["period_id"] == period_id)
        ]
        
        cmap = DEFAULT_CMAP if self.cmap is None else self.cmap
        human_width = 1.8 if self.human_width is None else self.human_width
        is_normed = False 

        focus_player = segment[segment["player_id"] == self.focus_player_id].iloc[0]

        other_players = segment[
            (segment["player_id"] != self.focus_player_id)
            & (segment["team_id"] != "ball")
        ].dropna(subset="shoulders_angle")

        vision = Vision(
            x=focus_player["x_adj"],
            y=focus_player["y_adj"],
            head_angle=focus_player["head_angle"],
            speed=focus_player["v"],
            smoothing_multiplier=smoothing_multiplier,
            human_width=human_width,
        )
        grid = vision.compute(
            x2s=other_players["x_adj"].values,
            y2s=other_players["y_adj"].values,
            shoulder_angles=other_players["shoulders_angle"].values,
        )
    
        im = self.ax.imshow(
            grid if not is_normed else grid / np.max(grid),
            origin="lower",
            extent=(
                -self.pitch_length / 2,
                self.pitch_length / 2,
                -self.pitch_width / 2,
                self.pitch_width / 2,
            ),
            cmap=cmap,
            norm=DEFAULT_NORM,
            interpolation="bilinear",
            alpha=0.5,
            zorder=1,
        )

        for i, row in segment.iterrows():
            x, y, vx, vy = row["x"], row["y"], row["vx"], row["vy"]

            player_id, team_id, head_angle, hips_angle, shoulders_angle, run_angle = (
                row["player_id"],
                row["team_id"],
                row["head_angle"],
                row["hips_angle"],
                row["shoulders_angle"],
                row["run_angle"],
            )

            # Plot player position
            self.pitch.scatter(
                x,
                y,
                ax=self.ax,
                s=150 if team_id != "ball" else 75,
                color=TEAM_COLORS[team_id][0],
                zorder=5 if team_id != "ball" else 15,
            )

            # Plot velocity arrow (if not the ball)
            if team_id != "ball" and ~np.isnan(head_angle):
                self.pitch.arrows(
                    x,
                    y,
                    x + (vx * arrow_multiplier),
                    y + (vy * arrow_multiplier),
                    ax=self.ax,
                    width=2,
                    headwidth=5,
                    headlength=5,
                    color=TEAM_COLORS[team_id][0],
                )

                if AngleType.HEAD in self.plot_angles:
                    wedge = Wedge(
                        (x, y),  # Center of the wedge
                        r=5,  # Radius of the wedge
                        theta1=(
                            np.degrees(head_angle) - 22.5
                        ),  # Start angle in degrees
                        theta2=(np.degrees(head_angle) + 22.5),  # End angle in degrees
                        color=TEAM_COLORS[team_id][0],
                        alpha=0.5,  # Transparency for better visualization
                    )
                    self.ax.add_patch(wedge)

                if AngleType.RUN in self.plot_angles:
                    __angled_bar(x, y, ax=self.ax, angle=run_angle, color="green")

                if AngleType.HIP in self.plot_angles:
                    __angled_bar(x, y, ax=self.ax, angle=hips_angle, color="orange")

                if AngleType.SHOULDER_DIRECTION in self.plot_angles:
                    __angled_bar(x, y, ax=self.ax, angle=shoulders_angle, color="red")

                if AngleType.SHOULDER_PLOT_ANGLE in self.plot_angles:
                    __angled_bar(
                        x,
                        y,
                        ax=self.ax,
                        angle=rotate_angle(shoulders_angle, by=90),
                        color=TEAM_COLORS[team_id][0],
                        start_point=2,
                        end_point=2,
                    )
                # Text with white border
                text = self.ax.text(
                    x,
                    y,
                    int(row["jersey_number"]),
                    color=TEAM_COLORS[team_id][0],
                    fontsize=8,
                    ha="center",
                    va="center",
                    zorder=5,
                )
                text.set_path_effects(
                    [
                        path_effects.Stroke(
                            linewidth=2, foreground="white"
                        ),  # White border
                        path_effects.Normal(),  # Restore normal text appearance
                    ]
                )

    def to_movie(
        self,
        file_path_mp4: str,
        focus_player_id: int,
        game_id: Union[str, int] = None,
        period_id: int = None,
        start_frame_id: int = None,
        end_frame_id: int = None,
        angles: List[AngleType] = [],
        cmap: str = None,
        human_width: float = None 
    ):
        all_none = all(
            param is None
            for param in [game_id, period_id, start_frame_id, end_frame_id]
        )
        all_provided = all(
            param is not None
            for param in [game_id, period_id, start_frame_id, end_frame_id]
        )

        if not (all_none or all_provided):
            raise ValueError(
                f"All filter parameters must be provided together, or should all be None."
            )

        if all_provided:
            segment = self.data.filter(
                (pl.col("game_id") == game_id)
                & (pl.col("period_id") == period_id)
                & (pl.col("frame_id").is_between(start_frame_id, end_frame_id))
            )
        else:
            segment = self.data

        from matplotlib.animation import FuncAnimation

        self.focus_player_id = focus_player_id

        self.segment = segment.to_pandas()
        self.plot_angles = angles
        self.cmap = cmap
        self.human_width = human_width
        
        self.pitch = Pitch(
            pitch_type="secondspectrum",
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
        )
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 7))

        frame_info = (
            segment[["game_id", "period_id", "frame_id"]]
            .unique()
            .to_pandas()
            .to_dict(orient="records")
        )

        frame_info = sorted(frame_info, key=lambda x: x["frame_id"], reverse=False)

        ani = FuncAnimation(
            self.fig,
            self.__plot_frame,
            frames=frame_info,
            repeat=False,
        )
        ani.save(file_path_mp4, fps=25, extra_args=["-vcodec", "libx264"], dpi=300)
