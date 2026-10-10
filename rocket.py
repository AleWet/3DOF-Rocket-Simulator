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
INITIAL_ANGLE = 45
DRAG_COEFFICIENT = 0.75
REFERENCE_AREA = 0.5
SEA_LEVEL_DENSITY = 1.225
SCALE_HEIGHT = 8500.0


def air_density(altitude):
    return SEA_LEVEL_DENSITY * np.exp(-altitude / SCALE_HEIGHT)


def derivative(t, state, thrust, mdot):
    """state = [x, y, z, vx, vy, vz, mass], z up. Returns d(state)/dt."""
    velocity = state[3:6]
    speed = np.linalg.norm(velocity)
    m = state[-1]

    drag = -0.5 * air_density(state[2]) * speed * DRAG_COEFFICIENT * REFERENCE_AREA * velocity
    acceleration = drag / m
    acceleration[2] += thrust / m - GRAVITY
    
    return [*velocity, *acceleration, -mdot]

# ---- define states and when to fire events ----

def hit_ground(t, state, *args):
    return state[2]


hit_ground.terminal = True
hit_ground.direction = -1  # only when altitude is decreasing, else it fires at t = 0


def apogee(t, state, *args):
    return state[5]

# ----                                        ----

apogee.direction = -1  # vertical velocity going from positive to negative


initial_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, LIFTOFF_MASS]


# i have to divide the two phases because i'm modelling the rocket with 
# a 20kN trhust that istantly goes out, so it's not smooth

# with some tests i found that the tolerances add about 10x the accuracy with little to no steps more
# maybe in the future they will be changeds (defaults are 1e-3 / 1e-6)

burn = solve_ivp(derivative, (0.0, BURN_TIME), initial_state, 
                 events=[hit_ground, apogee], args = (THRUST, MASS_FLOW_RATE), rtol=1e-6, atol=1e-6, dense_output=True)

burnout_state = burn.y[:, -1]

coast = solve_ivp(derivative, (BURN_TIME, 1000.0), burnout_state, 
                 events=[hit_ground, apogee], args = (0, 0), rtol=1e-6, atol=1e-6, dense_output=True)

# t_events / y_events are indexed like the events list: [0] = hit_ground, [1] = apogee
flight_time = coast.t_events[0][0]
impact = coast.y_events[0][0]
top = coast.y_events[1][0]

print(f"burnout  {burnout_state[5]:.2f} m/s  {burnout_state[2]:.1f} m  {burnout_state[6]:.1f} kg")
print(f"apogee   {top[2]:.1f} m")
print(f"impact   {flight_time:.2f} s")

# altitude over time; .sol gives the state at any time, not only at the solver's few steps
t_burn = np.linspace(0.0, BURN_TIME, 100)
t_coast = np.linspace(BURN_TIME, flight_time, 400)
plt.plot(t_burn, burn.sol(t_burn)[2])
plt.plot(t_coast, coast.sol(t_coast)[2])
plt.show()
