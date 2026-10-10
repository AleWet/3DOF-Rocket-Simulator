import numpy as np
from scipy.integrate import solve_ivp

G = 9.81  # m/s^2


def derivative(t, y):
    """State y = [x, y, z, vx, vy, vz], z up. Returns dy/dt."""
    vx, vy, vz = y[3:]
    return [vx, vy, vz, 0.0, 0.0, -G]

# ---- define states and when to fire states ----

def hit_ground(t, y):
    return y[2]


hit_ground.terminal = True
hit_ground.direction = -1  # only when altitude is decreasing, else it fires at t = 0


def apogee(t, y):
    return y[5]

# ----                                        ----

apogee.direction = -1  # vertical velocity going from positive to negative


speed = 100.0  # m/s
elevation = np.radians(45.0)
y0 = [0.0, 0.0, 0.0, speed * np.cos(elevation), 0.0, speed * np.sin(elevation)]

sol = solve_ivp(derivative, (0.0, 1000.0), y0, events=[hit_ground, apogee])

# t_events / y_events are indexed like the events list: [0] = hit_ground, [1] = apogee
flight_time = sol.t_events[0][0]
impact = sol.y_events[0][0]
top = sol.y_events[1][0]

print(f"range       {np.hypot(impact[0], impact[1]):.2f} m")
print(f"apogee      {top[2]:.2f} m")
print(f"flight time {flight_time:.3f} s")
print(f"solver steps: {sol.t.size - 1}, highest sample: {sol.y[2].max():.1f} m")
