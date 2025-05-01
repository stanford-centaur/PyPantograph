#!/usr/bin/env python3

import subprocess
from pathlib import Path
from pantograph.server import Server

if __name__ == '__main__':
    project_path = Path(__file__).parent.resolve() / 'Example'
    print(f"$PWD: {project_path}")
    server = Server(imports=['Example'], project_path=project_path)
    state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
    state1 = server.goal_tactic(state0, goal_id=0, tactic="aesop")
    assert state1.is_solved
