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

# size of the ellipse that holds 95% of a 2D gaussian cloud, in sigmas: about 2.448
# in 2D the k-sigma ellipse holds 1 - e^(-k^2 / 2), so 2 sigma is only 86.5%, not 95% like in 1D
K_95 = np.sqrt(-2.0 * np.log(1.0 - 0.95))


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


def confidence_ellipse(mean, cov_matrix, k):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # eigh for symmetric matrices
    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    circle = np.array([np.cos(angle), np.sin(angle)])
    # stretch the unit circle to k sigma along each axis, turn it onto the axes' directions, move it to the mean
    return mean[:, None] + eigenvectors @ (k * np.sqrt(eigenvalues)[:, None] * circle)


def fraction_inside(points, mean, cov_matrix, k):
    d = points - mean
    # Mahalanobis distance squared: how many sigmas away each point is, measured along the ellipse's own axes
    distance_squared = np.sum((d @ np.linalg.inv(cov_matrix)) * d, axis=1)
    return np.mean(distance_squared <= k**2)


def print_gaussian_check(points, mean, cov_matrix):
    # the cloud is gaussian if the share inside each ellipse matches what a gaussian puts there
    for k in (1.0, 2.0, 3.0, K_95):
        inside = fraction_inside(points, mean, cov_matrix, k)
        print(f"inside {k:.2f} sigma  {inside * 100:.1f} %  expected {(1 - np.exp(-k**2 / 2)) * 100:.1f} %")


def main():
    rng = np.random.default_rng(11)  # 11 lucky number

    nominal = simulate(NOMINAL_LAUNCH)
    nominal_impact = nominal["impact_state"][:2]
    nominal_apogee = nominal["apogee_state"][:3]
    impacts, apogees = run_monte_carlo(NOMINAL_LAUNCH, RELATIVE_SIGMAS, ABSOLUTE_SIGMAS, RUNS, rng)
    mean = impacts.mean(axis=0)        # axis=0: one value per column, so [east, north]
    std = impacts.std(axis=0, ddof=1)  # ddof=1: divide by runs - 1, like std in MATLAB

    cov_matrix = np.cov(impacts, rowvar=False)  # rowvar=False: one column per variable, else it is 1000 x 1000
    eigenvalues = np.linalg.eigh(cov_matrix).eigenvalues  # variances along the ellipse's axes, smallest first

    # apogees seen from the side: one row [distance from the ramp, altitude] per flight
    apogees_side = np.column_stack([np.hypot(apogees[:, 0], apogees[:, 1]), apogees[:, 2]])
    apogee_mean = apogees_side.mean(axis=0)
    apogee_cov_matrix = np.cov(apogees_side, rowvar=False)

    print(f"flights  {len(impacts)} of {RUNS}")
    print(f"nominal  east {nominal_impact[0]:.1f} m  north {nominal_impact[1]:.1f} m")
    print(f"mean     east {mean[0]:.1f} m  north {mean[1]:.1f} m")
    print(f"std      east {std[0]:.1f} m  north {std[1]:.1f} m")
    print(f"axes     {np.sqrt(eigenvalues[1]):.1f} m  {np.sqrt(eigenvalues[0]):.1f} m")

    print("impacts")
    print_gaussian_check(impacts, mean, cov_matrix)
    print("apogees")
    print_gaussian_check(apogees_side, apogee_mean, apogee_cov_matrix)

    # impact points seen from above
    plt.figure()
    plt.scatter(impacts[:, 0], impacts[:, 1], s=4)
    plt.scatter(*nominal_impact, color="black")
    plt.plot(*confidence_ellipse(mean, cov_matrix, K_95), color="black")  # 95% ellipse
    plt.title("impact points")
    plt.xlabel("east [m]")
    plt.ylabel("north [m]")
    plt.axis("equal")

    # apogee points seen from the side: distance from the ramp against altitude
    plt.figure()
    plt.scatter(apogees_side[:, 0], apogees_side[:, 1], s=4)
    plt.scatter(np.hypot(nominal_apogee[0], nominal_apogee[1]), nominal_apogee[2], color="black")
    plt.plot(*confidence_ellipse(apogee_mean, apogee_cov_matrix, K_95), color="black")  # 95% ellipse
    plt.title("apogee points")
    plt.xlabel("distance from the ramp [m]")
    plt.ylabel("altitude [m]")
    plt.axis("equal")

    plt.show()  # opens both figures


# true only when this file is run directly, not when another file imports it
if __name__ == "__main__":
    main()
