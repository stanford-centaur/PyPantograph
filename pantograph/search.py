import random
from abc import abstractmethod
import time
from dataclasses import dataclass
from typing import Optional, List
import collections, unittest
from math import log, sqrt
from pantograph.server import Server, TacticFailure, ServerError
from pantograph.expr import Expr, Tactic, GoalState, Site


@dataclass
class SearchState:

    goal_state: GoalState
    parent: Optional["SearchState"]
    parent_goal_id: Optional[int]
    priorities: list[float]
    children: Optional[List["SearchState"]] = None
    tested_tactics: Optional[List[Tactic]] = None
    total_value: Optional[float] = None
    tactic_feedback: Optional[str] = None

    def __post_init__(self):
        assert len(self.priorities) == len(self.goal_state.goals)
        self.solved = [False for _ in self.goal_state.goals]
        self.trials = [0 for _ in self.goal_state.goals]
        self.tested_tactics = [] if self.tested_tactics is None else self.tested_tactics
        self.children = [] if self.children is None else self.children
        self.visit_count = 1
        self.exhausted = False
        self.subtree_exhausted = False

    @property
    def next_goal_id(self) -> int:
        goal_id, _ = max(
            ((i, prio) for i, prio in enumerate(self.priorities) if not self.solved[i]),
            key=lambda x: x[1])
        return goal_id

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_solved(self) -> bool:
        return all(self.solved)

@dataclass(frozen=True)
class SearchResult:

    n_goals_root: int
    duration: float
    success: bool
    steps: int

class Agent:
    """
    An agent interface for proof search
    """

    @abstractmethod
    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
        ) -> Optional[Tactic]:
        """
        Implement this function to generate the next tactic for a goal
        """

    @abstractmethod
    def guidance(self, state: GoalState) -> list[float]:
        """
        Return a list of priorities determining which goal should be searched
        first. This will not be called on states with one or zero goals.
        """
        return [0.0 for _ in state.goals]
    @abstractmethod
    def reset(self):
        """
        Called after search
        """

    def search(self,
               server: Server,
               goal_state: GoalState,
               max_steps: int = 100,
               max_trials_per_goal: int = 5,
               verbose: bool = False) -> SearchResult:
        """
        Executes proof search on this state
        """

        assert server.is_automatic(), "Search must be run in automatic mode"

        n_goals_root = len(goal_state.goals)
        time_start = time.time()

        initial_state = SearchState(
            goal_state,
            parent=None,
            parent_goal_id=None,
            priorities=[0.0 for _ in goal_state.goals]
        )
        search_stack = [initial_state]
        for i_step in range(max_steps):
            assert search_stack, "No states in search stack"

            if verbose:
                print(f"I={i_step}: len(S) = {len(search_stack)}")
            search_state = search_stack[-1]

            assert isinstance(search_state, SearchState)

            if search_state.is_solved:
                return SearchResult(
                    n_goals_root=n_goals_root,
                    duration=time.time() - time_start,
                    success=True,
                    steps=i_step,
                )

            # Find the unsolved goal with the highest priority
            goal_id = search_state.next_goal_id

            if search_state.trials[goal_id] > max_trials_per_goal:
                # force halt the search
                tactic = None
            else:
                # Generate tactic for this goal
                tactic = self.next_tactic(search_state.goal_state, goal_id)

            if verbose:
                print(f"Next tactic: {tactic}")
            if not tactic:
                # resets the feedback
                search_state.tactic_feedback = None
                # pop the current state and continue to the next
                search_stack.pop(-1)
                if not search_stack:
                    if verbose:
                        print("Search stack has been exhausted")
                    self.reset()
                    return SearchResult(
                        n_goals_root=n_goals_root,
                        duration=time.time() - time_start,
                        success=False,
                        steps=i_step,
                    )
                continue

            try:
                search_state.trials[goal_id] += 1
                goal_state = search_state.goal_state
                if verbose:
                    print(f"{goal_state.state_id}.{goal_id}: {tactic} on {goal_state.goals[goal_id]}")
                next_goal_state = server.goal_tactic(goal_state, tactic, site=Site(goal_id, auto_resume=False))
                # Generate priorities for the next goal state
                priorities = [0.0 for _ in next_goal_state.goals] \
                    if len(next_goal_state.goals) <= 1 else \
                    self.guidance(next_goal_state)
                parent = len(search_stack) - 1
                next_state = SearchState(
                    goal_state=next_goal_state,
                    parent=search_state,
                    parent_goal_id=goal_id,
                    priorities=priorities
                )
                search_stack.append(next_state)

            except TacticFailure as t:
                if verbose:
                    print(f"Tactic failed: {t}")
                search_state.tactic_feedback = str(t)
                # try the next tactic. this one failed
            except ServerError as e:
                raise RuntimeError(f"While executing tactic: {tactic}") from e

        if verbose:
            print("Search iteration limit exhausted")

        self.reset()
        return SearchResult(
            n_goals_root=n_goals_root,
            duration=time.time() - time_start,
            success=False,
            steps=max_steps,
        )


class MCTSAgent(Agent):
    """
    An agent interface for proof search using monte carlo tree search
    """

    @abstractmethod
    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
            tested: Optional[List[Tactic]] = None,
        ) -> Optional[Tactic]:
        """
        Implement this function to generate the next tactic for a goal given tactics already tested
        """

    @abstractmethod
    def reset(self):
        """
        Called after search
        """

    @abstractmethod
    def estimate(self, state: SearchState) -> SearchState:
        """
        Implement this function to estimate the value of a state
        """

    @abstractmethod
    def select(self, state: SearchState) -> list[SearchState]:
        """
        Implement this function to select the best node within the subtree of the state.
        Returns the path to the selected node from the given state.
        """

    def backup(self, states: list[SearchState], value: float):
        """
        Backup value of the state at the end of the states list.
        """
        for state in states:
            state.total_value += value
            state.visit_count += 1
            state.subtree_exhausted = all(child.subtree_exhausted for child in state.children) and state.exhausted

    def search(self,
               server: Server,
               goal_state: GoalState,
               max_steps: int = 100,
               max_trials_per_goal: int = 5,
               verbose: bool = False) -> SearchResult:
        """
        Executes proof search on this state
        """

        assert server.is_automatic(), "Search must be run in automatic mode"

        n_goals_root = len(goal_state.goals)
        time_start = time.time()

        initial_state = SearchState(
            goal_state=goal_state,
            parent=None,
            parent_goal_id=None,
            priorities=[0.0 for _ in goal_state.goals]
        )
        initial_state = self.estimate(initial_state)
        search_root = initial_state

        for i_step in range(max_steps):
            search_trajectory = self.select(search_root)
            search_state = search_trajectory[-1]
            assert isinstance(search_state, SearchState)

            if search_state.is_solved:
                return SearchResult(
                    n_goals_root=n_goals_root,
                    duration=time.time() - time_start,
                    success=True,
                    steps=i_step,
                )

            # Find the unsolved goal with the highest priority
            goal_id = search_state.next_goal_id

            if search_state.trials[goal_id] > max_trials_per_goal:
                # force halt the search
                tactic = None
            else:
                # Generate tactic for this goal
                tactic = self.next_tactic(search_state.goal_state, goal_id, search_state.tested_tactics)

            if verbose:
                print(f"Next tactic: {tactic}")
            if not tactic:
                # resets the feedback
                search_state.tactic_feedback = None
                search_state.exhausted = True
                search_state.subtree_exhausted = all(child.subtree_exhausted for child in search_state.children)
                continue
            assert tactic not in search_state.tested_tactics, "Tactic already seen!"
            search_state.tested_tactics.append(tactic)

            try:
                search_state.trials[goal_id] += 1
                state = search_state.goal_state
                if verbose:
                    print(f"{state.state_id}.{goal_id}: {tactic} on {search_state.goal_state.goals[goal_id]}")
                next_goal_state = server.goal_tactic(goal_state, tactic, site=Site(goal_id, auto_resume=False))
                # Generate priorities for the next goal state
                priorities = [0.0 for _ in next_goal_state.goals] \
                    if len(next_goal_state.goals) <= 1 else \
                    self.guidance(next_goal_state)
                parent = -1
                next_state = SearchState(
                    goal_state=next_goal_state,
                    parent=parent,
                    parent_goal_id=goal_id,
                    priorities=priorities
                )
                next_state = self.estimate(next_state)
                search_state.children.append(next_state)
                self.backup(search_trajectory, next_state.total_value)
            except TacticFailure as t:
                if verbose:
                    print(f"Tactic failed: {t}")
                search_state.tactic_feedback = str(t)
                # try the next tactic. this one failed
            except ServerError as e:
                raise RuntimeError(f"While executing tactic: {tactic}") from e

        if verbose:
            print("Search iteration limit exhausted")

        self.reset()
        return SearchResult(
            n_goals_root=n_goals_root,
            duration=time.time() - time_start,
            success=False,
            steps=max_steps,
        )


class DumbAgent(Agent):

    def __init__(self):
        super().__init__()

        self.goal_tactic_id_map = collections.defaultdict(lambda : 0)
        self.intros = [
            "intro",
        ]
        self.tactics = [
            "intro h",
            "cases h",
            "apply Or.inl",
            "apply Or.inr",
        ]
        self.no_space_tactics = [
            "assumption",
        ]

    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
    ) -> Optional[Tactic]:
        key = (state.state_id, goal_id)
        i = self.goal_tactic_id_map[key]

        target = state.goals[goal_id].target
        if target.startswith('∀'):
            tactics = self.intros
        elif ' ' in target:
            tactics = self.tactics
        else:
            tactics = self.no_space_tactics

        if i >= len(tactics):
            return None

        self.goal_tactic_id_map[key] = i + 1
        return tactics[i]

class DumbMCTSAgent(MCTSAgent):
    def __init__(self):
        super().__init__()

        self.goal_tactic_id_map = collections.defaultdict(lambda : 0)
        self.intros = [
            "intro",
        ]
        self.tactics = [
            "intro h",
            "cases h",
            "apply Or.inl",
            "apply Or.inr",
        ]
        self.no_space_tactics = [
            "assumption",
        ]
        self.c = 0.6

    def estimate(self, state: SearchState) -> SearchState:
        state.total_value = random.random()
        return state

    def select(self, state: SearchState) -> list[SearchState]:
        """
        UCB scoring with taking the current state as one option, i.e. one child
        """
        state_trajectory = [state]
        current_state = state
        current_state_ucb = (state.total_value / state.visit_count) + self.c * sqrt((log(state.visit_count) / state.visit_count))
        while current_state.children:
            avg_val = [child.total_value / child.visit_count for child in current_state.children]
            visit_portions = [sqrt(log(current_state.visit_count) / child.visit_count) for child in current_state.children]
            ucbs = [avg + self.c * visit for avg, visit in zip(avg_val, visit_portions, strict=True)]
            child_idcs = [idx for idx in range(len(current_state.children)) if not current_state.children[idx].subtree_exhausted]
            if not child_idcs:
                return state_trajectory
            child_idx = child_idcs[0]
            for i in child_idcs:
                if ucbs[i] > ucbs[child_idx]:
                    child_idx = i
            if ucbs[child_idx] < current_state_ucb and not current_state.exhausted:
                return state_trajectory
            current_state_ucb = ucbs[child_idx]
            current_state = current_state.children[child_idx]
            state_trajectory.append(current_state)
        return state_trajectory

    def next_tactic(
            self,
            state: GoalState,
            goal_id: int,
            tested: Optional[List[Tactic]] = None
    ) -> Optional[Tactic]:
        key = (state.state_id, goal_id)
        i = self.goal_tactic_id_map[key]
        target = state.goals[goal_id].target
        if target.startswith('∀'):
            tactics = self.intros
        elif ' ' in target:
            tactics = self.tactics
        else:
            tactics = self.no_space_tactics

        if i >= len(tactics):
            return None
        self.goal_tactic_id_map[key] = i + 1
        while tactics[i] in tested:
            i += 1
            if i >= len(tactics):
                return None
        return tactics[i]


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
