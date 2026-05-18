from typing import Optional
from dataclasses import dataclass, field
import json 
from pathlib import Path

@dataclass
class TestCase:
    input: str
    expected: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
@dataclass
class Dataset:
    cases: list[TestCase]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        cases=[]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                cases.append(TestCase(
                    input=data["input"],
                    expected=data.get("expected"),
                    metadata=data.get("metadata", {}),
                ))
        return cls(cases=cases)