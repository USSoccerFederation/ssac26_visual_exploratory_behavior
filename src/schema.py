import polars as pl
import pandera.polars as pa

schema = pa.DataFrameSchema(
    {
        "frame_id": pa.Column(pl.Int64),
        "period_id": pa.Column(pl.Int64),
        "ball_possession": pa.Column(pl.String),
        "player_id": pa.Column(pl.Int64),
        "team_id": pa.Column(pl.String),
        "x": pa.Column(pl.Float64),
        "y": pa.Column(pl.Float64),
        "vx": pa.Column(pl.Float64),
        "vy": pa.Column(pl.Float64),
        "v": pa.Column(pl.Float64),
        "head_angle": pa.Column(pl.Float64, nullable=True),
        "shoulders_angle": pa.Column(pl.Float64, nullable=True),
        "hips_angle": pa.Column(pl.Float64, nullable=True),
        "is_imputed": pa.Column(pl.Boolean),
        "game_id": pa.Column(pl.String),
        "timestamp": pa.Column(pl.Float64),
        "ball_status": pa.Column(pl.String),
        "ball_dist": pa.Column(pl.Float64),
        "run_angle": pa.Column(pl.Float64, nullable=True),
        "x_adj": pa.Column(pl.Float64),
        "y_adj": pa.Column(pl.Float64),
        "ball_x_adj": pa.Column(pl.Float64),
        "ball_y_adj": pa.Column(pl.Float64),
        "player": pa.Column(pl.String, nullable=True),
        "jersey_number": pa.Column(pl.Int64, nullable=True),
        "play_type": pa.Column(pl.String),
        "is_on_ball": pa.Column(pl.Boolean),
        "is_gk": pa.Column(pl.Boolean, nullable=True),
    }
)
