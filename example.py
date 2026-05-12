import random

from pathlib import Path

from llmsys_hpobench import Benchmark

b = Benchmark(system="vLLM", root="experiment-data")

X = b.get_config_space()
Z = b.get_fidelity_space()

z = Z.sample(random_state=0)

budget = 10.0
t = 0.0
rng = random.Random(0)
m = None
while t < budget:
    x = X.sample(fidelity=z, random_state=rng)
    m = b.evaluate(config=x, fidelity=z)
    cost_values = [value for value in m["cost"].values() if isinstance(value, (int, float))]
    cost = sum(cost_values) if cost_values else 0.0
    t = t + cost

fidelity_dir = Path(m["fidelity"]["path"]).parent

log_file = fidelity_dir / m["log"]["file"]

print(m["perf"])
print(m["cost"])
print(m["hardware"])
print(log_file)
print(log_file.exists())
