# python-rocket

A 3DOF (point-mass) rocket flight model in Python. One flight goes through three
phases, ramp, burn and coast, with drag in an exponential atmosphere and a
constant wind. In flight the rocket points into the airflow, so it turns into
the wind.

This project is still in development. What is missing is listed under
[Planned](#planned).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python rocket.py
```

This flies the rocket defined by `MOCK_ROCKET`, prints a summary and shows the
trajectory in 3D.

```bash
python monte_carlo.py
```

This flies 1000 rockets with scattered inputs and shows where they land.

## Files

- `rocket.py`: the flight model. `simulate(params)` flies one rocket and returns the results.
- `monte_carlo.py`: the Monte Carlo. It scatters the inputs, calls `simulate` many times and collects the impact points.
- `requirements.txt`: the libraries the project needs.

## Planned

Not implemented yet:

- Wind that changes with altitude, and wind in the Monte Carlo.
- Realistic input uncertainties. The ones in `monte_carlo.py` are placeholders.
- Impact statistics: confidence ellipses for the impact point and the probability of landing outside a safe zone.
- Convergence analysis: how the statistics change with the number of runs.