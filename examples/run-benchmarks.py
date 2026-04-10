"""Run the river benchmarks for all models on all tracks.

This script is a wrapper around the `run_track` function in `river.benchmarks.run`. It runs the
benchmarks for all models on all tracks specified in config.py.

Current config.py depends on modified version of following files:
`river.benchmarks.run`
        47+        res["Prediction"] = i["Prediction"]
`river.evaluate.track`
        43+            yield_predictions=True,
`river.evaluate.progressive_validation`
        69+    qa = "q"
        73+        if qa == "q":
        79+            qa = "a"
        86+        if y_pred != {} and y_pred is not None and y is not None:
        103+        qa = "q"
"""

import importlib.util
import json
import logging
import sys

from config import MODELS, TRACKS

# Load the module by its path
module_path = "../river/benchmarks/run.py"
spec = importlib.util.spec_from_file_location("run", module_path)
if spec is not None and spec.loader is not None:
    run = importlib.util.module_from_spec(spec)
    sys.modules["run"] = run
    spec.loader.exec_module(run)
else:
    raise ImportError(f"Could not load {module_path}")

from run import run_track

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    details: dict[str, dict[str, dict[str, str]]] = {}
    # Create details for each model
    for i, track in enumerate(TRACKS):
        details[track.name] = {"Dataset": {}, "Model": {}}
        for dataset in track.datasets:
            details[track.name]["Dataset"][dataset.__class__.__name__] = repr(
                dataset
            )
        for model_name, model in MODELS[track.name].items():
            details[track.name]["Model"][model_name] = repr(model)
        with open("details.json", "w") as f:
            json.dump(details, f, indent=2)
        run_track(models=MODELS[track.name].keys(), no_track=i, n_workers=1)
