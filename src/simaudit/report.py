"""Strict report envelope. Evidence is an intentionally open JSON object."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ABSTAIN = "ABSTAIN"
    SKIPPED = "SKIPPED"
    PARTIAL = "PARTIAL"
    BLOWUP = "BLOWUP"


COVERAGE = {"tested", "disabled_for_test", "present_not_tested", "unsupported"}
FEATURES = {"dynamics", "parameters", "observations", "gravity", "springs", "contacts",
            "joint_limits", "actuators", "damping", "friction", "fluid", "tendons",
            "equality_constraints", "plugins", "wrappers", "flex", "mocap", "callbacks"}


@dataclass
class Feature:
    name: str
    status: str
    note: str
    present: bool | None = None

    def __post_init__(self):
        if not self.name or self.status not in COVERAGE or not self.note:
            raise ValueError("Invalid feature inventory entry")
        if self.present is not None and type(self.present) is not bool:
            raise ValueError("Feature presence must be boolean or null")
        if self.name not in FEATURES:
            self.status = "unsupported"


@dataclass
class Modification:
    name: str
    before: object
    after: object
    reason: str

    def __post_init__(self):
        if not self.name or not self.reason:
            raise ValueError("Modification needs a name and reason")


@dataclass
class Result:
    check: str
    status: Status
    reason: str
    scope: str
    limitations: list[str]
    coverage: list[Feature]
    modifications: list[Modification] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    fault_class: str | None = None

    def __post_init__(self):
        self.status = Status(self.status)
        if not all(isinstance(v, str) and v for v in (self.check, self.reason, self.scope)):
            raise ValueError("Results require check, reason and scope")
        if not self.coverage or not all(isinstance(v, Feature) for v in self.coverage):
            raise ValueError("Coverage is mandatory")
        if not self.limitations or not all(isinstance(v, str) and v for v in self.limitations):
            raise ValueError("Limitations are mandatory")
        if not isinstance(self.evidence, dict) or not isinstance(self.versions, dict):
            raise ValueError("Evidence and versions must be objects")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.versions.items()):
            raise ValueError("Versions must map names to strings")
        if not all(isinstance(m, Modification) for m in self.modifications):
            raise ValueError("Invalid modification ledger")
        if self.fault_class is not None and not isinstance(self.fault_class, str):
            raise ValueError("Invalid fault class")
        json.dumps(asdict(self), allow_nan=False)


@dataclass
class Report:
    results: list[Result]
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError("Unsupported report schema version")
        if not isinstance(self.results, list) or not all(isinstance(r, Result) for r in self.results):
            raise ValueError("Invalid results")

    @property
    def counts(self):
        return {s.value: sum(r.status == s for r in self.results) for s in Status}

    def exit_code(self, strict=False):
        return int(not self.results or any(r.status == Status.FAIL or
                   (strict and r.status != Status.PASS) for r in self.results))

    def dumps(self):
        return json.dumps(asdict(self), indent=2, allow_nan=False) + "\n"

    def write(self, path):
        Path(path).write_text(self.dumps(), encoding="utf-8")

    @classmethod
    def loads(cls, text):
        def pairs(items):
            result = {}
            for k, v in items:
                if k in result:
                    raise ValueError(f"Duplicate JSON key: {k}")
                result[k] = v
            return result
        def constant(value):
            raise ValueError(f"Nonfinite JSON number: {value}")
        data = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        if not isinstance(data, dict) or set(data) != {"schema_version", "results"}:
            raise ValueError("Invalid report fields")
        if data['schema_version'] != '1.0' or not isinstance(data['results'], list):
            raise ValueError("Unsupported report schema or results")
        results = []
        for row in data['results']:
            if not isinstance(row, dict) or set(row) != set(Result.__dataclass_fields__):
                raise ValueError("Invalid result fields")
            row = dict(row)
            for key, constructor in (("coverage", Feature), ("modifications", Modification)):
                if not isinstance(row[key], list):
                    raise ValueError(f"{key} must be a list")
                entries = []
                for entry in row[key]:
                    if not isinstance(entry, dict) or set(entry) != set(constructor.__dataclass_fields__):
                        raise ValueError(f"Invalid {key} fields")
                    entries.append(constructor(**entry))
                row[key] = entries
            results.append(Result(**row))
        return cls(results, data['schema_version'])

    def summary(self):
        lines = []
        for r in self.results:
            lines.append(f"{r.status.value} {r.check}: {r.reason} [{r.scope}]")
            lines.extend(f"  Limit: {v}" for v in r.limitations)
            lines.extend(f"  Coverage: {f.name}: {f.status} (present={f.present}): {f.note}" for f in r.coverage)
            lines.extend(f"  Modified: {m.name}: {m.before} -> {m.after}: {m.reason}" for m in r.modifications)
        lines.append("Counts: " + ", ".join(f"{k}={v}" for k, v in self.counts.items()))
        if not self.results or any(r.status not in (Status.PASS, Status.FAIL) for r in self.results):
            lines.append("WARNING: some requested behaviour was not assessed; this is not an all-clear.")
        return "\n".join(lines)
