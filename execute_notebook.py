"""Run every notebook cell and save the outputs."""

import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
for folder in [".cache/matplotlib", ".cache/ipython", ".cache/jupyter"]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache/jupyter"))
path = ROOT / "floodscope.ipynb"
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
started = time.perf_counter()
client = NotebookClient(notebook, km=manager, timeout=3600,
                        resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)


def on_cell_complete(cell, cell_index, **kwargs):
    if cell.cell_type == "code":
        print(f"Completed code cell at notebook index {cell_index}", flush=True)


client.on_cell_executed = on_cell_complete
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
    manager.cleanup_resources()
report = {
    "executed_code_cells": sum(cell.cell_type == "code" for cell in notebook.cells),
    "errors": sum(output.output_type == "error" for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs),
    "elapsed_seconds": time.perf_counter() - started,
    "method": "nbclient with a real IPython kernel using the project Python interpreter",
}
(ROOT / "results/notebook_execution.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
