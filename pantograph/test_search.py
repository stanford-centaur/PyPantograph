from .search import *
import unittest

class TestSearch(unittest.TestCase):

    def test_solve(self):

        server = Server()
        agent = DumbAgent()
        goal_state = server.goal_start("∀ (p q: Prop), p -> p")
        flag = agent.search(
            server=server,
            goal_state=goal_state,
            verbose=False)
        #flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)
    def test_solve_big(self):

        server = Server()
        agent = DumbAgent()
        goal_state = server.goal_start("∀ (p q: Prop), Or p q -> Or q p")
        flag = agent.search(
            server=server,
            goal_state=goal_state,
            verbose=False)
        self.assertTrue(flag)

class TestMCTSSearch(unittest.TestCase):

    def test_solve(self):

        server = Server()
        agent = DumbMCTSAgent()
        goal_state = server.goal_start("∀ (p q: Prop), p -> p")
        flag = agent.search(
            server=server,
            goal_state=goal_state,
            verbose=False)
        #flag = agent.search(server=server, target="∀ (p q: Prop), Or p q -> Or q p", verbose=True)
        self.assertTrue(flag)
    def test_solve_big(self):

        server = Server()
        agent = DumbMCTSAgent()
        goal_state = server.goal_start("∀ (p q: Prop), Or p q -> Or q p")
        flag = agent.search(
            server=server,
            goal_state=goal_state,
            max_steps=200,
            verbose=False)
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
