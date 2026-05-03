from pathlib import Path
from llmsys_hpobench import Benchmark

b = Benchmark(system="vLLM", root="experiment-data")

X = b.get_config_space()
Z = b.get_fidelity_space()

z = Z.sample(random_state=0)
x = X.sample(fidelity=z, random_state=0)
m = b.evaluate(config=x, fidelity=z)

fidelity_dir = Path(m["fidelity"]["path"]).parent

client_log = fidelity_dir / m["log"]["client-file"]
server_log = fidelity_dir / m["log"]["server-file"]

print(m["perf"])
print(m["cost"])
print(m["hardware"])
print(client_log)
print(server_log)
print(client_log.exists(), server_log.exists())
