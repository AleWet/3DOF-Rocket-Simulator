import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# consts dump temporary
GRAVITY = 9.81            

LIFTOFF_MASS = 1000.0     
PROPELLANT_MASS = 600.0   
MASS_FLOW_RATE = 30.0     
THRUST = 20000.0          
BURN_TIME = PROPELLANT_MASS / MASS_FLOW_RATE
LAUNCH_ELEVATION = 45
LAUNCH_AZIMUTH = 45

elevation_rad = np.radians(LAUNCH_ELEVATION)
azimuth_rad = np.radians(LAUNCH_AZIMUTH)
THRUST_DIRECTION = np.array([np.cos(elevation_rad) * np.sin(azimuth_rad),   # east
                             np.cos(elevation_rad) * np.cos(azimuth_rad),   # north
                             np.sin(elevation_rad)])                        # up

RAMP_LENGTH = 10.0 # meters 
RAMP_DIRECTION = THRUST_DIRECTION  # the ramp points the same way as the thrust

DRAG_COEFFICIENT = 0.75
REFERENCE_AREA = 0.5
SEA_LEVEL_DENSITY = 1.225
SCALE_HEIGHT = 8500.0


def air_density(altitude):
    return SEA_LEVEL_DENSITY * np.exp(-altitude / SCALE_HEIGHT)


def derivative(t, state, thrust, mdot, on_ramp):
    """state = [x, y, z, vx, vy, vz, mass], z up. Returns d(state)/dt."""
    velocity = state[3:6]
    speed = np.linalg.norm(velocity)
    m = state[-1]

    drag = -0.5 * air_density(state[2]) * speed * DRAG_COEFFICIENT * REFERENCE_AREA * velocity
    acceleration = (drag + thrust * THRUST_DIRECTION) / m
    acceleration[2] -= GRAVITY

    if on_ramp:
        # bolted to the ramp: keep only the part of the acceleration that points along it
        acceleration = np.dot(acceleration, RAMP_DIRECTION) * RAMP_DIRECTION

    return [*velocity, *acceleration, -mdot]

# ---- define events and when they fire ----

def hit_ground(t, state, *args):
    return state[2]

hit_ground.terminal = True
hit_ground.direction = -1  # only when altitude is decreasing, else it fires at t = 0

def apogee(t, state, *args):
    return state[5]

apogee.direction = -1  # vertical velocity going from positive to negative

def leave_ramp(t, state, *args):
    return np.dot(state[0:3], RAMP_DIRECTION) - RAMP_LENGTH  # distance travelled along the ramp, minus its length

leave_ramp.terminal = True # if you finish the ramp you get into powered ascent
leave_ramp.direction = 1  # distance going from shorter to longer than the ramp

# ----                                        ----

initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, LIFTOFF_MASS]


# i have to divide the flight phases because i'm modelling the rocket with 
# a 20kN trhust that istantly goes out, so it's not smooth

# with some tests i found that the tolerances add about 10x the accuracy with little to no steps more
# maybe in the future they will be changeds (defaults are 1e-3 / 1e-6)

# ramp phase: it ends with an event, because the time the rocket leaves the ramp is not known in advance
ramp = solve_ivp(derivative, (0.0, BURN_TIME), initial_state,
                 events=[leave_ramp], args = (THRUST, MASS_FLOW_RATE, True), rtol=1e-6, atol=1e-6, dense_output=True)

ramp_exit_time = ramp.t_events[0][0]
ramp_exit_state = ramp.y_events[0][0]

burn = solve_ivp(derivative, (ramp_exit_time, BURN_TIME), ramp_exit_state, 
                 events=[hit_ground, apogee], args = (THRUST, MASS_FLOW_RATE, False), rtol=1e-6, atol=1e-6, dense_output=True)

burnout_state = burn.y[:, -1]

coast = solve_ivp(derivative, (BURN_TIME, 1000.0), burnout_state, 
                 events=[hit_ground, apogee], args = (0, 0, False), rtol=1e-6, atol=1e-6, dense_output=True)

# t_events / y_events are indexed like the events list: [0] = hit_ground, [1] = apogee
flight_time = coast.t_events[0][0]
impact = coast.y_events[0][0]
top = coast.y_events[1][0]

print(f"ramp     {ramp_exit_time:.2f} s  {np.linalg.norm(ramp_exit_state[3:6]):.2f} m/s")
print(f"burnout  {np.linalg.norm(burnout_state[3:6]):.2f} m/s  {burnout_state[2]:.1f} m  {burnout_state[6]:.1f} kg")
print(f"apogee   {top[2]:.1f} m")
print(f"impact   {flight_time:.2f} s  east {impact[0]:.1f} m  north {impact[1]:.1f} m  range {np.hypot(impact[0], impact[1]):.1f} m")

# trajectory in 3d; .sol gives the state at any time, not only at the solver's few steps
t_ramp = np.linspace(0.0, ramp_exit_time, 20)
t_burn = np.linspace(ramp_exit_time, BURN_TIME, 100)
t_coast = np.linspace(BURN_TIME, flight_time, 400)
ax = plt.figure().add_subplot(projection="3d")
ax.plot(*ramp.sol(t_ramp)[:3], color="black", linewidth=3)  # rows 0, 1, 2 of the state are x, y, z
ax.plot(*burn.sol(t_burn)[:3])
ax.plot(*coast.sol(t_coast)[:3])
ax.set(xlabel="east [m]", ylabel="north [m]", zlabel="up [m]")
ax.set_aspect("equal", adjustable="datalim")  # same scale on all three axes
plt.show()
