"""3DOF point-mass rocket: one flight from the ramp to the impact."""

import warnings

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# fixed number in the definition of specific impulse: thrust = mass flow * specific impulse * STANDARD_GRAVITY
# it is not the local gravity, that one is "gravity" in the dict below
STANDARD_GRAVITY = 9.80665  # m/s^2

# every number that defines one flight; simulate() takes a dict like this one
MOCK_ROCKET = {
    "gravity": 9.81,             # m/s^2
    "liftoff_mass": 1000.0,      # kg, structure + propellant
    "propellant_mass": 600.0,    # kg
    "thrust": 20000.0,           # N
    "specific_impulse": 20000.0 / (30.0 * STANDARD_GRAVITY),  # s, about 68: the value that makes this engine burn 30 kg/s
    "launch_elevation": 80.0,    # degrees above the horizon
    "launch_azimuth": 45.0,      # degrees clockwise from north
    "ramp_length": 10.0,         # m
    "drag_coefficient": 0.75,    # dimensionless
    "reference_area": 0.5,       # m^2
    "sea_level_density": 1.225,  # kg/m^3
    "scale_height": 8500.0,      # m
}

# solver settings, the same for every flight
# with some tests i found that the tolerances add about 10x the accuracy with little to no steps more
# maybe in the future they will be changeds (defaults are 1e-3 / 1e-6)
SOLVER_TOLERANCE = 1e-6
MAX_FLIGHT_TIME = 1000.0  # s, upper limit for the coast phase

# there is always a ramp: in flight the thrust follows the velocity, and the ramp gives the rocket its first one
MIN_RAMP_LENGTH = 3.0  # m


def direction_from_angles(elevation, azimuth):
    """Unit vector [east, north, up] for an elevation and an azimuth in degrees."""
    elevation_rad = np.radians(elevation)
    azimuth_rad = np.radians(azimuth)
    return np.array([np.cos(elevation_rad) * np.sin(azimuth_rad),   # east
                     np.cos(elevation_rad) * np.cos(azimuth_rad),   # north
                     np.sin(elevation_rad)])                        # up
    # in the future i'll use NED coordinates 


def air_density(altitude, params):
    return params["sea_level_density"] * np.exp(-altitude / params["scale_height"])


def derivative(t, state, thrust, mdot, on_ramp, launch_direction, params):
    """state = [x, y, z, vx, vy, vz, mass], x east, y north, z up. Returns d(state)/dt."""
    velocity = state[3:6]
    speed = np.linalg.norm(velocity)
    m = state[-1]

    if on_ramp or speed == 0.0:
        thrust_direction = launch_direction  # on the ramp, or at rest, there is no velocity to follow this should happen only on ramp
    else:
        thrust_direction = velocity / speed  # in flight the thrust follows the velocity

    drag = (-0.5 * air_density(state[2], params) * speed
            * params["drag_coefficient"] * params["reference_area"] * velocity)
    acceleration = (drag + thrust * thrust_direction) / m
    acceleration[2] -= params["gravity"]

    if on_ramp:
        # bolted to the ramp: keep only the part of the acceleration that points along it
        # max(..., 0.0): the bottom of the ramp holds the rocket, so it cannot slide backwards
        acceleration = max(np.dot(acceleration, launch_direction), 0.0) * launch_direction

    return [*velocity, *acceleration, -mdot]

# ---- define events and when they fire ----
# the solver calls them with the same arguments as derivative

def hit_ground(t, state, *args):
    return state[2]

hit_ground.terminal = True
hit_ground.direction = -1  # only when altitude is decreasing, else it fires at t = 0

def apogee(t, state, *args):
    return state[5]

apogee.direction = -1  # vertical velocity going from positive to negative

def leave_ramp(t, state, thrust, mdot, on_ramp, launch_direction, params):
    return np.dot(state[0:3], launch_direction) - params["ramp_length"]  # distance travelled along the ramp, minus its length

leave_ramp.terminal = True # if you finish the ramp you get into powered ascent
leave_ramp.direction = 1  # distance going from shorter to longer than the ramp

# ----                                        ----


def simulate(params):
    if params["ramp_length"] < MIN_RAMP_LENGTH:
        raise ValueError(f"ramp_length must be at least {MIN_RAMP_LENGTH} m")

    launch_direction = direction_from_angles(params["launch_elevation"], params["launch_azimuth"])
    thrust = params["thrust"]
    mdot = thrust / (params["specific_impulse"] * STANDARD_GRAVITY)  # the engine's mass flow follows from its thrust and specific impulse
    burn_time = params["propellant_mass"] / mdot
    initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, params["liftoff_mass"]]

    # i have to divide the flight phases because i'm modelling the rocket with
    # a 20kN trhust that istantly goes out, so it's not smooth

    # ramp phase: it ends with an event, because the time the rocket leaves the ramp is not known in advance
    ramp = solve_ivp(derivative, (0.0, burn_time), initial_state,
                     events=[leave_ramp], args=(thrust, mdot, True, launch_direction, params),
                     rtol=SOLVER_TOLERANCE, atol=SOLVER_TOLERANCE, dense_output=True)

    # no leave_ramp event: the thrust never beat the weight, so there is no flight to return
    if len(ramp.t_events[0]) == 0:
        warnings.warn("the rocket never left the ramp, flight discarded")
        return None

    ramp_exit_time = ramp.t_events[0][0]
    ramp_exit_state = ramp.y_events[0][0]

    burn = solve_ivp(derivative, (ramp_exit_time, burn_time), ramp_exit_state,
                     events=[hit_ground, apogee], args=(thrust, mdot, False, launch_direction, params),
                     rtol=SOLVER_TOLERANCE, atol=SOLVER_TOLERANCE, dense_output=True)
                     
    # status 1 means a terminal event stopped the run, here hit_ground: the rocket
    # nosed into the ground with the engine still on, so there is no burnout and no coast
    hit_ground_during_burn = burn.status == 1

    if hit_ground_during_burn:
        burnout_time = None
        burnout_state = None
        last_phase = burn
    else:
        burnout_time = burn_time
        burnout_state = burn.y[:, -1]
        coast = solve_ivp(derivative, (burn_time, MAX_FLIGHT_TIME), burnout_state,
                          events=[hit_ground, apogee], args=(0.0, 0.0, False, launch_direction, params),
                          rtol=SOLVER_TOLERANCE, atol=SOLVER_TOLERANCE, dense_output=True)
        last_phase = coast

    # t_events / y_events are indexed like the events list: [0] = hit_ground, [1] = apogee
    impact_time = last_phase.t_events[0][0]
    impact_state = last_phase.y_events[0][0]

    # the rocket can tip over and start falling with the engine still on,
    # so the apogee is looked for in the burn phase first
    if len(burn.t_events[1]) > 0:
        apogee_phase = burn
    else:
        apogee_phase = coast

    # time history of the whole flight; .sol gives the state at any time, not only at the solver's few steps
    # endpoint=False leaves out the last time of a phase, because the next phase starts with it
    t_ramp = np.linspace(0.0, ramp_exit_time, 20, endpoint=False)
    t_burn = np.linspace(ramp_exit_time, burn.t[-1], 100, endpoint=hit_ground_during_burn)  # burn.t[-1] is when the burn run stopped
    times = [t_ramp, t_burn]
    states = [ramp.sol(t_ramp), burn.sol(t_burn)]
    if not hit_ground_during_burn:
        t_coast = np.linspace(burn_time, impact_time, 400)
        times.append(t_coast)
        states.append(coast.sol(t_coast))

    return {
        "ramp_exit_time": ramp_exit_time,
        "ramp_exit_state": ramp_exit_state,
        "burnout_time": burnout_time,    # None if the rocket hit the ground before burnout
        "burnout_state": burnout_state,  # None too
        "apogee_time": apogee_phase.t_events[1][0],
        "apogee_state": apogee_phase.y_events[1][0],
        "impact_time": impact_time,
        "impact_state": impact_state,
        "time": np.concatenate(times),
        "states": np.concatenate(states, axis=1),  # shape (7, times)
    }


def print_summary(result):
    ramp_exit = result["ramp_exit_state"]
    burnout = result["burnout_state"]
    top = result["apogee_state"]
    impact = result["impact_state"]
    ramp_exit_time = result["ramp_exit_time"]
    impact_time = result["impact_time"]

    print(f"ramp     {ramp_exit_time:.2f} s  {np.linalg.norm(ramp_exit[3:6]):.2f} m/s")
    if burnout is None:
        print("burnout none, hit the ground with the engine still on")
    else:
        print(f"burnout  {np.linalg.norm(burnout[3:6]):.2f} m/s  {burnout[2]:.1f} m  {burnout[6]:.1f} kg")
    print(f"apogee   {top[2]:.1f} m")
    print(f"impact   {impact_time:.2f} s  east {impact[0]:.1f} m  north {impact[1]:.1f} m  range {np.hypot(impact[0], impact[1]):.1f} m")


def plot_trajectory(result):
    """3d plot of the flight path: ramp in black, burn in blue, coast in orange."""
    time = result["time"]
    position = result["states"][:3]  # rows 0, 1, 2 of the state are x, y, z

    burn_end_time = result["burnout_time"]
    if burn_end_time is None:  # no burnout: the burn lasted until the impact
        burn_end_time = result["impact_time"]

    # True/False for every time sample, used to pick the samples of one phase
    on_ramp = time <= result["ramp_exit_time"]
    burning = (time >= result["ramp_exit_time"]) & (time <= burn_end_time)
    coasting = time >= burn_end_time

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(*position[:, on_ramp], color="black", linewidth=3)
    ax.plot(*position[:, burning])
    ax.plot(*position[:, coasting])
    ax.set(xlabel="east [m]", ylabel="north [m]", zlabel="up [m]")
    ax.set_aspect("equal", adjustable="datalim")  # same scale on all three axes
    plt.show()


def main():
    result = simulate(MOCK_ROCKET)
    if result is None:  # the flight was discarded
        return
    print_summary(result)
    plot_trajectory(result)


# true only when this file is run directly, not when another file imports it
if __name__ == "__main__":
    main()
