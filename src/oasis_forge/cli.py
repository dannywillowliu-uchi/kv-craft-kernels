"""oasis-forge CLI.

  oasis-forge problems              list kernel problems + harvest status
  oasis-forge ledger [problem]      show attempts / best kept port
  oasis-forge solve <problem>       build prompt + launch agent (needs target box)
  oasis-forge harvest <problem>     dump golden tensors from a rollout (needs GPU box)

solve/harvest are stubs until hardware is wired; problems/ledger work now.
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from .config import PortConfig
from .ledger import Ledger

console = Console()
ROOT = Path(__file__).resolve().parents[2]


def _problems() -> list[dict]:
	out = []
	for p in sorted((ROOT / "problems").glob("*/problem.yaml")):
		data = yaml.safe_load(p.read_text())
		data["_dir"] = p.parent
		out.append(data)
	return out


@click.group()
def main() -> None:
	"""Port-and-optimize world-model kernels from B300/sm_100 to H100/sm_90."""


@main.command()
def problems() -> None:
	"""List kernel problems and whether their golden tensors have been harvested."""
	t = Table(title="oasis-h100-port problems")
	t.add_column("kernel")
	t.add_column("tier")
	t.add_column("harvested")
	t.add_column("shapes")
	for prob in _problems():
		tf = prob["_dir"] / "task_files"
		harvested = "yes" if (tf / "golden.npz").exists() else "no (stub)"
		shapes = prob.get("shapes_status", "from harvest")
		t.add_row(prob["name"], prob.get("default_tier", "structural"), harvested, str(shapes))
	console.print(t)


@main.command()
@click.argument("problem", required=False)
def ledger(problem: str | None) -> None:
	"""Show port attempts (optionally for one problem) and the best kept port."""
	cfg = PortConfig()
	led = Ledger(ROOT / cfg.ledger_path)
	rows = [a for a in led.all() if problem is None or a.problem == problem]
	if not rows:
		console.print("[yellow]no attempts logged yet[/yellow]")
		return
	t = Table(title="attempts")
	for c in ("problem", "tier", "compiled", "t1", "t2", "speedup", "approach"):
		t.add_column(c)
	for a in rows:
		t.add_row(
			a.problem,
			a.tier,
			"y" if a.compiled else "n",
			"y" if a.tier1_pass else "n",
			{True: "y", False: "n", None: "-"}[a.tier2_pass],
			f"{a.speedup:.2f}x" if a.speedup else "-",
			a.approach[:48],
		)
	console.print(t)
	for prob in {a.problem for a in rows}:
		best = led.best(prob)
		if best:
			console.print(f"[green]best {prob}: {best.speedup:.2f}x — {best.approach}[/green]")


@main.command()
@click.argument("problem")
def solve(problem: str) -> None:
	"""Launch the optimizer agent for a problem. STUB: needs a target H100 box."""
	console.print(
		f"[red]solve {problem} is a stub.[/red] Wire H100Remote (src/oasis_forge/remote.py) "
		"and populate task_files via `harvest` first. See agents/agent_prompt.md."
	)


@main.command()
@click.argument("problem")
def harvest(problem: str) -> None:
	"""Dump golden input/output tensors from a rollout. STUB: needs a GPU box + model."""
	console.print(
		f"[red]harvest {problem} is a stub.[/red] Implement harvest/hooks.py against a live "
		"Oasis-500M rollout to emit task_files/golden.npz."
	)


if __name__ == "__main__":
	main()
