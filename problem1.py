import pandas as pd
import gurobipy as gp
import math
import os

from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
from gurobipy import GRB


os.environ["GRB_LICENSE_FILE"] = (
    r"\gurobi.lic" 
)


@dataclass
class Order:
    car_level: int
    pick_time: datetime
    return_time: datetime
    pick_station: int
    return_station: int
    return_int: int = 0  # ex: 1 : first 30 min
    pick_int: int = 0  # ex: 1 : first 30 min
    hours: int = 0

def problem1(file_name):
    dir = os.getcwd()
    paras_dict = {}
    init_cars = None
    car_rate = None
    orders = []
    starting_time = datetime(2023, 1, 1, 0, 0)
    T = None

    # parsing the txt file
    with open(dir + f"/data/{file_name}", "r") as f:
        # red the first two line
        paras_symbol = f.readline().strip().split(",")
        paras_val = f.readline().strip().split(",")
        for a, b in zip(paras_symbol, paras_val):
            paras_dict[a] = int(b)
        n_L = paras_dict["n_L"]
        n_S = paras_dict["n_S"]
        T = [[0] * n_S for _ in range(n_S)]
        init_cars = [[0] * n_L for _ in range(n_S)]
        f.readline()
        n_C = paras_dict["n_C"]
        car_rate = [0] * n_L

        header = f.readline()  # read the header of car table
        for _ in range(n_C):
            car_info = f.readline().strip().split(",")
            init_cars[int(car_info[2]) - 1][int(car_info[1]) - 1] += 1

        f.readline()
        f.readline()
        for i in range(n_L):
            ex = f.readline().strip().split(",")
            car_rate[i] = int(ex[1])

        f.readline()
        f.readline()
        n_K = paras_dict["n_K"]
        for _ in range(n_K):
            order_info = f.readline().strip().split(",")
            order = Order(
                car_level=int(order_info[1]) - 1,
                pick_station=int(order_info[2]) - 1,
                return_station=int(order_info[3]) - 1,
                pick_time=datetime.strptime(order_info[4], "%Y/%m/%d %H:%M"),
                return_time=datetime.strptime(order_info[5], "%Y/%m/%d %H:%M"),
            )
            order.return_int = int(
                (order.return_time - starting_time).total_seconds() // 1800
            )
            order.pick_int = int((order.pick_time - starting_time).total_seconds() // 1800)
            order.hours = (order.return_int - order.pick_int) / 2
            orders.append(order)

        f.readline()
        f.readline()

        for i in range(n_S * n_S):
            mov_info = f.readline().strip().split(",")
            T[int(mov_info[0]) - 1][int(mov_info[1]) - 1] = int(mov_info[2])

    Start = defaultdict(list)
    End = defaultdict(list)
    n_D = paras_dict["n_D"]

    for i, order in enumerate(orders):
        order.id = i
        if order.pick_int >= 48 * n_D:
            raise ValueError(f"Order {i} has pick time beyond the time horizon.")
        Start[(order.pick_station, max(order.pick_int - 1, 0), order.car_level)].append(i)
        End[(order.return_station, order.return_int + 8, order.car_level)].append(i)


    n_S = paras_dict["n_S"]
    n_L = paras_dict["n_L"]
    n_K = paras_dict["n_K"]
    time_grid = range(0, 48 * n_D + 1)
    O = range(n_K)
    L = range(n_L)
    S = range(n_S)

    m = gp.Model("Fleet_Management_Midterm")

    I = m.addVars(S, time_grid, L, vtype=GRB.INTEGER, name="Inventory")

    F = m.addVars(S, S, time_grid, L, vtype=GRB.INTEGER, name="EmptyFlow")

    X_exact = m.addVars(O, vtype=GRB.BINARY, name="X_exact")
    X_up = m.addVars(O, vtype=GRB.BINARY, name="X_up")

    m.setObjective(
        gp.quicksum(
            car_rate[orders[o].car_level]
            * orders[o].hours
            * (3 * X_exact[o] + 3 * X_up[o] - 2)
            for o in O
        ),
        GRB.MAXIMIZE,
    )

    m.addConstrs(
        (X_up[o] == 0 for o in O if orders[o].car_level == n_L - 1),
        name="No_Max_Level_Upgrades",
    )

    m.addConstrs((X_exact[o] + X_up[o] <= 1 for o in O), name="Fulfillment_Limit")


    m.addConstr(
        gp.quicksum(
            F[i, j, t, l] * T[i][j]
            for i in S
            for j in S
            if i != j
            for t in time_grid
            for l in L
        )
        <= paras_dict["B"],
        name="max moving time",
    )

    # Use m.addConstrs to loop over all s, t, l combinations
    m.addConstrs(
        (
            (
                I[s, t, l]
                == (I[s, t - 1, l] if t > 0 else init_cars[s][l])
                # + Incoming Empty Cars: Only add if departure time >= 0
                + gp.quicksum(
                    F[i, s, t - (T[i][s] // 30), l]
                    for i in S
                    if i != s and (t - (T[i][s] // 30)) >= 0
                )
                # - Outgoing Empty Cars
                - gp.quicksum(F[s, j, t, l] for j in S if j != s)
                # + Orders Returning (Exact Match)
                + gp.quicksum(X_exact[o] for o in End[(s, t, l)])
                # + Orders Returning (Upgraded from l-1): Only if l > 0
                + gp.quicksum(X_up[o] for o in End[(s, t, l - 1)])
                # - Orders Starting (Exact Match)
                - gp.quicksum(X_exact[o] for o in Start[(s, t, l)])
                # - Orders Starting (Upgraded from l-1): Only if l > 0
                - gp.quicksum(X_up[o] for o in Start[(s, t, l - 1)])
            )
            for s in S
            for t in time_grid
            for l in L
        ),
        name="Flow_Conservation",
    )
    m.optimize()
    # First, ensure the model actually found a mathematical solution
    if m.status == GRB.OPTIMAL:
        print("\n=== INVENTORY TRACKER ===")

        # Loop through your sets
        for o in O:
            if X_exact[o].X > 0.5:  # If the order is fulfilled exactly
                print(
                    f"Order {o} fulfilled exactly: Car Level {orders[o].car_level}, "
                    f"Pick Station {orders[o].pick_station}, Return Station {orders[o].return_station}, "
                    f"Pick Time {orders[o].pick_time}, Return Time {orders[o].return_time}"
                )
            elif X_up[o].X > 0.5:  # If the order is fulfilled with an upgrade
                print(
                    f"Order {o} fulfilled with upgrade: Car Level {orders[o].car_level}, "
                    f"Pick Station {orders[o].pick_station}, Return Station {orders[o].return_station}, "
                    f"Pick Time {orders[o].pick_time}, Return Time {orders[o].return_time}"
                )
        return m.ObjVal
    elif m.status == GRB.INFEASIBLE:
        print("Model is Infeasible. No inventory to print.")
        return -math.inf


test_case = [("instance01.txt", 27900), ("instance02.txt", 49500),
             ("instance03.txt", 50000), ("instance04.txt", 82300), ("instance05.txt", 106800)]

for (file_name, expected_val) in test_case:
    assert problem1(file_name) == expected_val, f"Failed on {file_name}"
