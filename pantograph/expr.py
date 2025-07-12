"""
Data structuers for expressions and goals
"""
from pantograph.message import Message

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypeAlias

Expr: TypeAlias = str

def parse_expr(payload: dict) -> Expr:
    """
    :meta private:
    """
    return payload["pp"]

class TacticMode(Enum):
    """
    Current execution mode
    """
    TACTIC = 1
    CONV = 2
    CALC = 3

    def serial(self):
        match self:
            case TacticMode.TACTIC: return "tactic"
            case TacticMode.CONV: return "conv"
            case TacticMode.CALC: return "calc"

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
    sibling_dep: Optional[set[int]] = field(default_factory=lambda: None)
    name: Optional[str] = None
    mode: TacticMode = TacticMode.TACTIC

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
        mode = TacticMode[payload["fragment"].upper()]

        sibling_dep = None
        for e in [payload["target"]] \
                + [v["type"] for v in payload["vars"]] \
                + [v["value"] for v in payload["vars"] if "value" in v]:
            dependents = e.get("dependentMVars")
            if dependents is None:
                continue
            deps = [sibling_map[d] for d in dependents if d in sibling_map]
            if sibling_dep:
                sibling_dep = { *sibling_dep, *deps }
            else:
                sibling_dep = { *deps }

        return Goal(id, variables, target, sibling_dep, name, mode)

    def __str__(self):
        head = f"{self.name}\n" if self.name else ""
        front = "|" if self.mode == TacticMode.CONV else "⊢"
        return head +\
            "\n".join(str(v) for v in self.variables) +\
            f"\n{front} {self.target}"

@dataclass(frozen=True)
class GoalState:
    state_id: int
    goals: list[Goal]
    messages: list[Message]

    # For tracking memory usage
    _sentinel: list[int]

    def __del__(self):
        self._sentinel.append(self.state_id)
    def __repr__(self):
        cls = self.__class__.__name__
        messages = f"messages={repr(self.messages)}, " if self.messages else ""
        return f"{cls}(#{self.state_id}, goals={repr(self.goals)}{messages}, _sentinel=#{len(self._sentinel)}"

    @property
    def is_solved(self) -> bool:
        """
        WARNING: Does not handle dormant goals.

        :meta public:
        """
        return not self.goals

    @staticmethod
    def parse_inner(state_id: int, goals: list, messages: list[dict], _sentinel: list[int]):
        assert _sentinel is not None
        goal_names = { g["name"]: i for i, g in enumerate(goals) }
        goals = [Goal.parse(g, goal_names) for g in goals]
        messages = [Message.parse(m) for m in messages]
        return GoalState(state_id, goals, messages, _sentinel)
    @staticmethod
    def parse(payload: dict, messages: list[dict], _sentinel: list[int]):
        return GoalState.parse_inner(payload["nextStateId"], payload["goals"], messages, _sentinel)

    def __str__(self):
        """
        :meta public:
        """
        return "\n".join([str(g) for g in self.goals])

@dataclass(frozen=True)
class Site:
    """
    Acting area of a tactic
    """
    goal_id: Optional[int] = None
    auto_resume: Optional[bool] = None

    def serial(self) -> dict:
        result = {}
        if self.goal_id is not None:
            result["goalId"] = self.goal_id
        if self.auto_resume is not None:
            result["autoResume"] = self.auto_resume
        return result

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

Tactic: TypeAlias = str | TacticHave | TacticLet | TacticExpr | TacticDraft | TacticMode
