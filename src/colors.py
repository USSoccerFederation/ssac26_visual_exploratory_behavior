import matplotlib as mpl

TEAM_COLORS = {
    "costa-rica": ["#EC1D24", "#132361"],
    "bolivia": ["#C4E8AC", "#DF0000"],
    "colombia": ["#FF6600", "#141414", "#FFFF00"],
    "venezuela": ["#8B0000", "#FFFFFF"],
    "argentina": ["#6AB5FF", "#0035D5"],
    "peru": ["#FFFFFF", "#000000"],
    "panama": ["#F60309", "#FFFFFF"],
    "mexico": ["#620D05", "#BBDCD5"],
    "paraguay": ["#EE0507", "#333AB7"],
    "chile": ["#FF0000", "#FFFFFF"],
    "brazil": ["#FFDC02", "#19AE47"],
    "uruguay": ["#51AEDD", "#000000"],
    "canada": ["#C5281C", "#000000"],
    "united-states": ["#BB2533", "#1F2742"],
    "ecuador": ["#2A3676", "#FFFFFF"],
    "jamaica": ["#F1D300", "#2F7229"],
    "ball": ["#000000", "#FFFFFF"],
}

ATTACK_COLOR = "#C51635"
DEFENSE_COLOR = "#212844"

color_list = [DEFENSE_COLOR, "white", ATTACK_COLOR]
DEFAULT_CMAP = mpl.colors.LinearSegmentedColormap.from_list("", color_list)
DEFAULT_NORM = mpl.colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
