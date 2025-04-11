"""
Data structuers for expressions and goals
"""
from dataclasses import dataclass, field
from typing import Optional, TypeAlias

Expr: TypeAlias = str

def parse_expr(payload: dict) -> Expr:
    """
    :meta private:
    """
    return payload["pp"]

@dataclass(frozen=True)
class Variable:
    t: Expr
    v: Optional[Expr] = None
    name: Optional[str] = None

    @staticmethod
    def parse(payload: dict):
        name = payload.get("userName")
        t = parse_expr(payload["type"])
        v = payload.get("value")
        if v:
            v = parse_expr(v)
        return Variable(t, v, name)

    def __str__(self):
        """
        :meta public:
        """
        result = self.name if self.name else "_"
        result += f" : {self.t}"
        if self.v:
            result += f" := {self.v}"
        return result

@dataclass(frozen=True)
class Goal:
    id: str
    variables: list[Variable]
    target: Expr
    sibling_dep: list[int] = field(default_factory=lambda: [])
    name: Optional[str] = None
    is_conversion: bool = False

    @staticmethod
    def sentence(target: Expr):
        """
        :meta public:
        """
        return Goal(id=None, variables=[], target=target)

    @staticmethod
    def parse(payload: dict, sibling_map: dict[str, int]):
        id = payload["name"]
        name = payload.get("userName")
        variables = [Variable.parse(v) for v in payload["vars"]]
        target = parse_expr(payload["target"])
        is_conversion = payload["isConversion"]

        dependents = payload["target"]["dependentMVars"]
        sibling_dep = [sibling_map[d] for d in dependents if d in sibling_map]

        return Goal(id, variables, target, sibling_dep, name, is_conversion)

    def __str__(self):
        head = f"{self.name}\n" if self.name else ""
        front = "|" if self.is_conversion else "⊢"
        return head +\
            "\n".join(str(v) for v in self.variables) +\
            f"\n{front} {self.target}"

@dataclass(frozen=True)
class GoalState:
    state_id: int
    goals: list[Goal]

    _sentinel: list[int]

    def __del__(self):
        self._sentinel.append(self.state_id)

    @property
    def is_solved(self) -> bool:
        """
        WARNING: Does not handle dormant goals.

        :meta public:
        """
        return not self.goals

    @staticmethod
    def parse_inner(state_id: int, goals: list, _sentinel: list[int]):
        assert _sentinel is not None
        goal_names = { g["name"]: i for i, g in enumerate(goals) }
        goals = [Goal.parse(g, goal_names) for g in goals]
        return GoalState(state_id, goals, _sentinel)
    @staticmethod
    def parse(payload: dict, _sentinel: list[int]):
        return GoalState.parse_inner(payload["nextStateId"], payload["goals"], _sentinel)

    def __str__(self):
        """
        :meta public:
        """
        return "\n".join([str(g) for g in self.goals])

@dataclass(frozen=True)
class TacticHave:
    """
    The `have` tactic, equivalent to
    ```lean
    have {binder_name} : {branch} := ...
    ```
    """
    branch: str
    binder_name: Optional[str] = None
@dataclass(frozen=True)
class TacticLet:
    """
    The `let` tactic, equivalent to
    ```lean
    let {binder_name} : {branch} := ...
    ```
    """
    branch: str
    binder_name: Optional[str] = None
@dataclass(frozen=True)
class TacticCalc:
    """
    The `calc` tactic, equivalent to
    ```lean
    calc {step} := ...
    ```
    You can use `_` in the step.
    """
    step: str
@dataclass(frozen=True)
class TacticExpr:
    """
    Assigns an expression to the current goal
    """
    expr: str
@dataclass(frozen=True)
class TacticDraft:
    """
    Assigns an expression to the current goal
    """
    expr: str

Tactic: TypeAlias = str | TacticHave | TacticLet | TacticCalc | TacticExpr | TacticDraft
