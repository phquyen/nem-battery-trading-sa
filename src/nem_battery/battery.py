"""Simplified battery model and daily dispatch optimisation (linear programming)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp
from scipy.optimize import linprog


@dataclass
class BatterySpec:
    power_mw: float = 1.0            # max charge/discharge rate
    energy_mwh: float = 2.0          # storage capacity
    efficiency_rt: float = 0.90      # round-trip efficiency
    start_soc_mwh: float = 0.0       # state of charge at start (and end) of each day
    degradation_cost: float = 10.0   # $ per MWh discharged
    dt_h: float = 0.5                # hours per interval (30 min)


def optimise_day(prices, spec: BatterySpec = BatterySpec()) -> pd.DataFrame:
    """Best charge/discharge schedule for one day, given (forecast) prices.

    Decision variables, stacked in one vector x of length 3n:
      x[0:n]   = charge (MW)      x[n:2n] = discharge (MW)      x[2n:3n] = state of charge (MWh)
    """
    p = np.asarray(prices, dtype=float)
    n = len(p)
    dt = spec.dt_h
    eta = spec.efficiency_rt ** 0.5          # losses split equally between charge and discharge
    ic, idis, isoc = 0, n, 2 * n             # where each block starts in x

    # Objective. linprog MINIMISES, so we minimise the NEGATIVE profit:
    # profit = sum(p * (discharge - charge) * dt) - degradation * discharge * dt
    cost = np.zeros(3 * n)
    cost[ic:ic + n] = p * dt
    cost[idis:idis + n] = -(p - spec.degradation_cost) * dt

    # Equality constraints (the battery physics), one row per half-hour + one for end of day
    A = np.zeros((n + 1, 3 * n))
    b = np.zeros(n + 1)
    for t in range(n):
        A[t, isoc + t] = 1                   # soc_t
        if t > 0:
            A[t, isoc + t - 1] = -1          # - soc_(t-1)
        A[t, ic + t] = -eta * dt             # - energy stored from charging
        A[t, idis + t] = dt / eta            # + energy drawn for discharging
    b[0] = spec.start_soc_mwh                # first half-hour starts from the initial charge
    A[n, isoc + n - 1] = 1                   # end of day...
    b[n] = spec.start_soc_mwh                # ...back at the starting charge

    bounds = [(0, spec.power_mw)] * (2 * n) + [(0, spec.energy_mwh)] * n

    res = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    assert res.success, res.message

    x = res.x
    return pd.DataFrame({
        "charge_mw": x[ic:ic + n],
        "discharge_mw": x[idis:idis + n],
        "soc_mwh": x[isoc:isoc + n],
    })

def realised_profit(schedule: pd.DataFrame, actual_prices, spec: BatterySpec = BatterySpec()) -> float:
    """Profit of a fixed schedule, paid at ACTUAL prices."""
    p = np.asarray(actual_prices, dtype=float)
    energy_net = (schedule["discharge_mw"] - schedule["charge_mw"]).to_numpy() * spec.dt_h
    degradation = spec.degradation_cost * schedule["discharge_mw"].sum() * spec.dt_h
    return float((p * energy_net).sum() - degradation)