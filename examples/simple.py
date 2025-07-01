from pantograph.server import Server

if __name__ == '__main__':
    server = Server(imports=['Init'])
    state0 = server.goal_start("forall (p q: Prop), Or p q -> Or q p")
    state1 = server.goal_tactic(state0, tactic="intro")
    print(state1)
