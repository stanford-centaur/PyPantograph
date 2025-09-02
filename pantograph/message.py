from dataclasses import dataclass
from enum import Enum
from typing import Optional, Self

class Severity(Enum):
    INFORMATION = 1
    WARNING = 2
    ERROR = 3

    def __str__(self):
        cls = self.__class__.__name__
        return super(Severity, self).__str__()[len(cls)+1:].lower()

@dataclass(frozen=True)
class Position:
    """
    Position in a file
    """
    line: int
    column: int

    @staticmethod
    def parse(d: Optional[dict]) -> Optional[Self]:
        if d is None:
            return None
        return Position(line=d["line"], column=d["column"])

@dataclass(frozen=True)
class Message:
    data: str
    pos: Position
    pos_end: Optional[Position] = None
    severity: Severity = Severity.ERROR
    kind: Optional[str] = None

    @staticmethod
    def parse(d: dict) -> Self:
        kind = None if d["kind"] == "[anonymous]" else d["kind"]
        return Message(
            severity=Severity[d["severity"].upper()],
            pos=Position.parse(d["pos"]),
            pos_end=Position.parse(d.get("endPos")),
            kind=kind,
            data=d["data"],
        )

    def __str__(self) -> str:
        if self.pos_end is not None:
            pos_end = f"-{self.pos_end.line}:{self.pos_end.column}"
        else:
            pos_end = ""
        match self.severity:
            case Severity.INFORMATION:
                prefix = ""
            case Severity.WARNING:
                prefix = "warning: "
            case Severity.ERROR:
                prefix = "error: "
        return f"{self.pos.line}:{self.pos.column}{pos_end}: {prefix}{self.data}"


class TacticFailure(Exception):
    """
    Indicates a tactic failed to execute
    """
class ParseError(Exception):
    """
    Indicates a logical error in the server.
    """
class ServerError(Exception):
    """
    Indicates a logical error in the server.
    """
