from dataclasses import dataclass
from enum import Enum
from typing import Optional, Self

class Severity(Enum):
    INFORMATION = 1
    WARNING = 2
    ERROR = 3

@dataclass(frozen=True)
class Message:
    data: str
    severity: Severity = Severity.ERROR
    kind: Optional[str] = None

    def __str__(self):
        return f"{self.kind}/{self.severity}: {self.data}"

    @staticmethod
    def parse(d: dict) -> Self:
        kind = None if d["kind"] == "[anonymous]" else d["kind"]
        return Message(
            severity=Severity[d["severity"].upper()],
            kind=kind,
            data=d["data"],
        )


class TacticFailure(Exception):
    """
    Indicates a tactic failed to execute
    """
class ServerError(Exception):
    """
    Indicates a logical error in the server.
    """
