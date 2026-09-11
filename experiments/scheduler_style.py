SCHEDULER_MAP = {
    'VLLM':     'vLLM',
    'SARATHI':  'Sarathi',
    'PACE':     'PaceServe',
    'PACE_NEW': 'PaceServe (New)',
}

SCHEDULER_COLORS = {
    "PaceServe":       "#2CA02C",  # green
    "PaceServe (New)": "#98DF8A",  # light green
    "vLLM":            "#1F77B4",  # blue
    "Sarathi":         "#FF7F0E",  # orange
    "DistServe":       "#D62728",  # red
    "2H1L":            "#D62728",  # red
}

SCHEDULER_MARKERS = {
    "PaceServe":       "*",  # star
    "PaceServe (New)": "o",  # circle
    "vLLM":            "s",  # square
    "Sarathi":         "^",  # triangle up
    "DistServe":       "P",  # plus (filled)
}

SCHEDULER_LINESTYLES = {
    "PaceServe":       "-.",
    "PaceServe (New)": ":",
    "vLLM":            "-",
    "Sarathi":         "--",
    "DistServe":       (0, (3, 1, 1, 1)),
}

SCHEDULER_ORDER = ["vLLM", "Sarathi", "DistServe", "PaceServe", "PaceServe (New)"]

DATASET_MAP = {
    'sharegpt': 'ShareGPT',
    'longbench': 'LongBench',
}

FONT_LABEL  = 16  # axis labels
FONT_TICK   = 12  # tick numbers
FONT_LEGEND = 12  # legend text
FONT_TITLE  = 16  # subplot / figure titles

def apply_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size':        FONT_LABEL,
        'axes.labelsize':   FONT_LABEL,
        'xtick.labelsize':  FONT_TICK,
        'ytick.labelsize':  FONT_TICK,
        'legend.fontsize':  FONT_LEGEND,
        'axes.titlesize':   FONT_TITLE,
    })
