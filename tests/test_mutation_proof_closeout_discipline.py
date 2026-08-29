from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _disposable_harness(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "disposable-repo"
    ops = root / "ops"
    ops.mkdir(parents=True)
    source = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "mutation-proof-closeout-discipline.sh"
    )
    script = ops / source.name
    shutil.copy2(source, script)
    target = root / "sample.py"
    target.write_text("STATE = 'restored'\n", encoding="utf-8")
    return script, target


@pytest.mark.parametrize(
    ("mutated_rc", "restored_rc", "expected_rc"),
    ((1, 0, 0), (0, 0, 1), (1, 1, 1)),
)
def test_mutation_proof_accepts_exactly_fails_then_passes(
    tmp_path: Path, mutated_rc: int, restored_rc: int, expected_rc: int
) -> None:
    script, target = _disposable_harness(tmp_path)
    command = r'''
source "$1"
install_cleanup_traps
run_test() {
  if grep -q mutated "$ACTIVE_FILE"; then
    return "$MUTATED_RC"
  fi
  return "$RESTORED_RC"
}
mutate "$2" "STATE = 'restored'" "STATE = 'mutated'" "synthetic-test" "synthetic probe"
mutation_proof_result
'''
    env = os.environ.copy()
    env.update(MUTATED_RC=str(mutated_rc), RESTORED_RC=str(restored_rc))
    completed = subprocess.run(
        ["bash", "-c", command, "mutation-test", str(script), str(target)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert completed.returncode == expected_rc, completed.stdout + completed.stderr
    assert target.read_text(encoding="utf-8") == "STATE = 'restored'\n"
    expected = "PASSED" if expected_rc == 0 else "FAILED"
    assert f"MUTATION PROOF {expected}" in completed.stdout


def test_mutation_proof_signal_trap_restores_the_active_backup(tmp_path: Path) -> None:
    script, target = _disposable_harness(tmp_path)
    command = r'''
source "$1"
install_cleanup_traps
run_test() {
  kill -TERM "$$"
  return 1
}
mutate "$2" "STATE = 'restored'" "STATE = 'mutated'" "synthetic-test" "signal probe"
'''
    completed = subprocess.run(
        ["bash", "-c", command, "mutation-test", str(script), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 143, completed.stdout + completed.stderr
    assert target.read_text(encoding="utf-8") == "STATE = 'restored'\n"
