"""Monte Carlo: fly the rocket many times with scattered inputs and collect the impact and apogee points."""

import numpy as np
import matplotlib.pyplot as plt

from rocket import simulate, MOCK_ROCKET

# standard deviation of every input that gets scattered
# for now all these numbers are placeholders

# errors that grow with the input, as a fraction of its value
RELATIVE_SIGMAS = {
    "thrust": 0.02,             # 2%
    "specific_impulse": 0.005,  # 0.5%
    "drag_coefficient": 0.05,   # 5%
    "liftoff_mass": 0.01,       # 1%
    "sea_level_density": 0.05,  # 5%
}

# errors that don't, in the input's own units: a fraction of zero wind would be zero
ABSOLUTE_SIGMAS = {
    "launch_elevation": 0.5,    # deg
    "launch_azimuth": 0.5,      # deg
    "wind_east": 1.0,           # m/s
    "wind_north": 1.0,          # m/s
}

NOMINAL_LAUNCH = dict(MOCK_ROCKET)
RUNS = 1000

def scatter(nominal, relative_sigmas, absolute_sigmas, rng):
    params = dict(nominal)
    for name in relative_sigmas:
        error = rng.normal(0.0, relative_sigmas[name])
        params[name] = nominal[name] * (1 + error)
    for name in absolute_sigmas:
        error = rng.normal(0.0, absolute_sigmas[name])
        params[name] = nominal[name] + error
    return params


def run_monte_carlo(nominal, relative_sigmas, absolute_sigmas, runs, rng):
    impacts = []  # one row [east, north] per flight
    apogees = []  # one row [east, north, up] per flight
    for run in range(runs):
        result = simulate(scatter(nominal, relative_sigmas, absolute_sigmas, rng))
        if result is None:  # the flight was discarded
            continue
        impacts.append(result["impact_state"][:2])
        apogees.append(result["apogee_state"][:3])
    return np.array(impacts), np.array(apogees)


def main():
    rng = np.random.default_rng(11)  # 11 lucky number

    nominal = simulate(NOMINAL_LAUNCH)
    nominal_impact = nominal["impact_state"][:2]
    nominal_apogee = nominal["apogee_state"][:3]
    impacts, apogees = run_monte_carlo(NOMINAL_LAUNCH, RELATIVE_SIGMAS, ABSOLUTE_SIGMAS, RUNS, rng)
    mean = impacts.mean(axis=0)        # axis=0: one value per column, so [east, north]
    std = impacts.std(axis=0, ddof=1)  # ddof=1: divide by runs - 1, like std in MATLAB

    print(f"flights  {len(impacts)} of {RUNS}")
    print(f"nominal  east {nominal_impact[0]:.1f} m  north {nominal_impact[1]:.1f} m")
    print(f"mean     east {mean[0]:.1f} m  north {mean[1]:.1f} m")
    print(f"std      east {std[0]:.1f} m  north {std[1]:.1f} m")

    # impact points seen from above
    plt.figure()
    plt.scatter(impacts[:, 0], impacts[:, 1], s=4)
    plt.scatter(*nominal_impact, color="black")
    plt.title("impact points")
    plt.xlabel("east [m]")
    plt.ylabel("north [m]")
    plt.axis("equal")

    # apogee points seen from the side: distance from the ramp against altitude
    plt.figure()
    plt.scatter(np.hypot(apogees[:, 0], apogees[:, 1]), apogees[:, 2], s=4)
    plt.scatter(np.hypot(nominal_apogee[0], nominal_apogee[1]), nominal_apogee[2], color="black")
    plt.title("apogee points")
    plt.xlabel("distance from the ramp [m]")
    plt.ylabel("altitude [m]")
    plt.axis("equal")

    plt.show()  # opens both figures


# true only when this file is run directly, not when another file imports it
if __name__ == "__main__":
    main()
