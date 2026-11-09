from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from llmsys.data import ParameterSpec, Scalar, SearchSpace, SystemSpec


def derived_seed(seed: int, algorithm: str) -> int:
    payload = f"{seed}:{algorithm}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def objective_loss(spec: SystemSpec, objective: float) -> float:
    return -float(objective) if spec.maximize else float(objective)


@dataclass(frozen=True)
class ExternalSpaceCodec:
    space: SearchSpace

    def uses_tokens(self, parameter: ParameterSpec) -> bool:
        return parameter.kind == "categorical" or parameter.lower == parameter.upper

    def choices(self, parameter: ParameterSpec) -> tuple[Scalar, ...]:
        if parameter.kind == "categorical":
            return parameter.choices
        value: Scalar
        if parameter.kind == "integer":
            value = int(parameter.lower)
        else:
            value = float(parameter.lower)
        return (value,)

    def tokens(self, parameter: ParameterSpec) -> tuple[str, ...]:
        return tuple(f"choice_{index}" for index, _ in enumerate(self.choices(parameter)))

    def decode(self, values: Mapping[str, object]) -> dict[str, Scalar]:
        decoded: dict[str, Scalar] = {}
        for parameter in self.space.parameters:
            value = values[parameter.name]
            if self.uses_tokens(parameter):
                index = self.tokens(parameter).index(str(value))
                decoded[parameter.name] = self.choices(parameter)[index]
            elif parameter.kind == "integer":
                decoded[parameter.name] = int(value)
            else:
                decoded[parameter.name] = float(value)
        return decoded
