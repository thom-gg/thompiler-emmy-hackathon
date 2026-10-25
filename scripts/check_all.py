"""Check our compiler on every problem, and stop at the first failure.

For each problem folder (problems/ and tests/problems/):
  1. compile: run compile.py on each kernel's .ir.json, the way the
     evaluation calls it (`python compile.py <ir.json>`, PTX on stdout);
  2. with --gpu: submit the problem with submit.py --compiler compile.py:
     the hosted runner assembles the PTX with ptxas, runs it on its GPU, and
     the outputs are compared with expected.npz.
The first compile or GPU failure stops the run, with its full output.

Usage:
    venv/bin/python scripts/check_all.py                    # compile every problem
    venv/bin/python scripts/check_all.py --gpu              # compile and run every problem
    venv/bin/python scripts/check_all.py --gpu r01_array_sum m01_matmul_512
Exit code 1 at the first failure.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEM_ROOTS = [ROOT / "problems", ROOT / "tests" / "problems"]


def problem_dirs(names: list[str]) -> list[Path]:
    """Every problem folder (one holding a meta.json), sorted; or only those named in `names`."""
    dirs = sorted(d for root in PROBLEM_ROOTS for d in root.iterdir() if (d / "meta.json").exists())
    if not names:
        return dirs
    by_name = {d.name: d for d in dirs}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        sys.exit(f"unknown problem(s): {', '.join(unknown)}")
    return [by_name[n] for n in names]


def compile_problem(pdir: Path) -> str | None:
    """Compile every kernel of the problem: None if all compile, else the error."""
    for ir in sorted(pdir.glob("*.ir.json")):
        result = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.startswith(".version"):
            return f"{ir.name}:\n{result.stderr.strip() or 'no PTX on stdout'}"
    return None


def run_problem(pdir: Path) -> tuple[bool, str]:
    """Submit the problem to the GPU runner: (passed, verdict or full output)."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "submit.py"), str(pdir), "--compiler", str(ROOT / "compile.py")],
        capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stdout + result.stderr
    verdicts = [line for line in result.stdout.splitlines() if " PASS " in line]
    return True, verdicts[-1].split("PASS", 1)[1].strip() if verdicts else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", help="problem folder names (default: all)")
    ap.add_argument("--gpu", action="store_true", help="also run each problem on the hosted GPU runner")
    args = ap.parse_args()

    dirs = problem_dirs(args.problems)
    for done, pdir in enumerate(dirs):
        error = compile_problem(pdir)
        if error is not None:
            print(f"FAIL  {pdir.name} (compile) {error}")
            print(f"\nstopped: {done} of {len(dirs)} passed before {pdir.name}")
            return 1
        if not args.gpu:
            print(f"ok    {pdir.name}")
            continue
        passed, output = run_problem(pdir)
        if not passed:
            print(f"FAIL  {pdir.name} (GPU)\n\n{output}")
            print(f"stopped: {done} of {len(dirs)} passed before {pdir.name}")
            return 1
        print(f"PASS  {pdir.name:22} {output}")
    print(f"\nall {len(dirs)} {'passed on the GPU' if args.gpu else 'compiled'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
