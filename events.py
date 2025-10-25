from dataclasses import dataclass, asdict

@dataclass
class Event:
    type: str            # "key" or "mouse"
    time: float          # Time interval, in seconds
    data: dict           # Event-specific data

    def to_json(self):
        return asdict(self)

    @staticmethod
    def from_json(obj):
        return Event(**obj)
