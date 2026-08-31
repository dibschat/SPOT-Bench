"""Per-task timing tolerances and thresholds for SPOT-Bench scoring.

A ground-truth response time `t` defines a slot window `[t - offset, t + offset]`
inside which a prediction scores a full Timeliness of 1.0. Outside the window the
score decays as a Gaussian: `sigma_early` governs predictions that arrive before
the window, `sigma_late` those that arrive after. Predictions scoring below
`timeliness_threshold` are never matched.

`sim_threshold` is the minimum LLM-judge score (0-5) required for an open-ended
answer to count as semantically correct. It is `None` for the detection tasks
(ABD, PNR), which are matched with exact keyword rules instead of a judge.
"""

#     gold interval (offset) = +/- IQR = +/- 1.349*sigma   (human consensus core)
#     sigma_late             = sigma                       (late shoulder = human dispersion)
#     sigma_early            = 2*sigma                     (2:1 anticipation preference)

TASK_TIME_CONFIG = {
    "PNR": {
        "offset": 0.5,
        "sigma_early": 0.74,
        "sigma_late": 0.37,
        "timeliness_threshold": 0.05,
        "sim_threshold": None,
    },
    "ABD": {
        "offset": 1.0,
        "sigma_early": 1.48,
        "sigma_late": 0.74,
        "timeliness_threshold": 0.05,
        "sim_threshold": None,
    },
    "SQA": {
        "offset": 1.0,
        "sigma_early": 1.48,
        "sigma_late": 0.74,
        "timeliness_threshold": 0.05,
        "sim_threshold": 3,
    },
    "SPG": {
        "offset": 1.5,
        "sigma_early": 2.22,
        "sigma_late": 1.11,
        "timeliness_threshold": 0.05,
        "sim_threshold": 3,
    },
    "SI": {
        "offset": 1.5,
        "sigma_early": 2.22,
        "sigma_late": 1.11,
        "timeliness_threshold": 0.05,
        "sim_threshold": 2,
    },
    "UI": {
        "offset": 0.5,
        "sigma_early": 0.74,
        "sigma_late": 0.37,
        "timeliness_threshold": 0.05,
        "sim_threshold": 2,
    },
}

NO_EVENT_TEXTS = {
    "no",
    "none",
    "nothing",
    "no answer",
    "no response",
    "no reply",
}

# Default per-slot occupancy budget K for Timeliness-F1@K matching.
DEFAULT_OCCUPANCY_K = 5
