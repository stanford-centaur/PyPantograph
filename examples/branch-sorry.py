#!/usr/bin/env python3

from pantograph.server import Server
from pantograph.expr import TacticHave

# This example shows what happens when a tactic generates a sorry.
if __name__ == '__main__':
    server = Server(imports=['Init'])
    state0 = server.goal_start("1 = 0")
    state1 = server.goal_tactic(state0, tactic=TacticHave("1 = 0"))
    print(state1)
    state1b = server.goal_tactic(state1, tactic="apply?")
    print(state1b)
