"""Run all five nonlinear manufactured examples and save the raw result records.

Usage (from the code directory): python run_all_benchmarks.py
This can take substantial time because it includes multi-seed PINN and nonlinear fits.
The off-lattice distance check is also written to the output directory.
"""
import json, os
import muntz_pinn_extended as m

os.makedirs(m.OUT, exist_ok=True)
results = []
for i in range(1, 6):
    path = os.path.join(m.OUT, f"ex{i}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
        print(f"Using cached {path}")
    else:
        result = m.experiment(i)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    results.append(result)

dist = m.distance_check()
with open(os.path.join(m.OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump({"results": results, "dist": dist}, f, indent=2)
m.write_tables(results, dist)
m.make_figures(results, dist)
print(f"Finished. Outputs are in: {m.OUT}")
