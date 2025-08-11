from pantograph.message import Severity, Message
from pantograph.expr import GoalState

from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

@dataclass(frozen=True)
class TacticInvocation:
    """
    One tactic invocation with the before/after goals extracted from Lean source
    code.
    """
    before: str
    after: str
    tactic: str
    used_constants: list[str]

    @staticmethod
    def parse(payload: dict):
        return TacticInvocation(
            before=payload["goalBefore"],
            after=payload["goalAfter"],
            tactic=payload["tactic"],
            used_constants=payload.get('usedConstants', []),
        )

@dataclass(frozen=True)
class CompilationUnit:

    # Byte boundaries [begin, end[ of each compilation unit.
    i_begin: int
    i_end: int

    messages: list[Message] = field(default_factory=lambda: [])

    invocations: Optional[list[TacticInvocation]] = None
    # If `goal_state` is none, maybe error has occurred. See `messages`
    goal_state: Optional[GoalState] = None
    goal_src_boundaries: Optional[list[Tuple[int, int]]] = None

    new_constants: Optional[list[str]] = None

    @staticmethod
    def parse(payload: dict, goal_state_sentinel=None, invocations=None):
        i_begin = payload["boundary"][0]
        i_end = payload["boundary"][1]
        messages = [Message.parse(m) for m in payload["messages"]]

        if invocations:
            invocations = [
                TacticInvocation.parse(i) for i in invocations
            ]
        else:
            invocations = None

        if (state_id := payload.get("goalStateId")) is not None:
            goal_state = GoalState.parse_inner(int(state_id), payload["goals"], [], goal_state_sentinel)
            goal_src_boundaries = payload["goalSrcBoundaries"]
        else:
            goal_state = None
            goal_src_boundaries = None

        new_constants = payload.get("newConstants")

        return CompilationUnit(
            i_begin,
            i_end,
            messages,
            invocations,
            goal_state,
            goal_src_boundaries,
            new_constants
        )

@dataclass
class CheckTrackResult:

    src_messages : list[Message]
    dst_messages : list[Message]
    failure : Optional[str] = None

    @property
    def hasSrcError(self):
        return any(m.severity == Severity.ERROR for m in self.src_messages)
    @property
    def hasDstError(self):
        return any(m.severity == Severity.ERROR for m in self.dst_messages)
    @property
    def succeeded(self):
        return not self.hasSrcError and not self.hasDstError and self.failure is None
    @property
    def feedback(self):
        """
        Feedback based on the dst
        """
        result = [str(s) for s in self.dst_messages]
        if self.failure:
            result.append(self.failure)
        return result

@dataclass(frozen=True)
class SearchTarget:
    goal_state: GoalState

    @staticmethod
    def parse(payload: dict, goal_state_sentinel=None):
        state_id = payload.get("stateId")
        goal_state = GoalState.parse_inner(int(state_id), payload["goals"], [], goal_state_sentinel)

        return SearchTarget(
            goal_state,
        )
