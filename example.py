from pathlib import Path
from llmsys_hpobench import Benchmark

b = Benchmark(system="vLLM", root="experiment-data")

X = b.get_config_space()
Z = b.get_fidelity_space()

z = Z.sample(random_state=0)
x = X.sample(fidelity=z, random_state=0)
m = b.evaluate(config=x, fidelity=z)

fidelity_dir = Path(m["fidelity"]["path"]).parent

log_file = fidelity_dir / m["log"]["file"]

print(m["perf"])
print(m["cost"])
print(m["hardware"])
print(log_file)
print(log_file.exists())
