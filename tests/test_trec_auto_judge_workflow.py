"""Minimum-compatibility checks for the TREC AutoJudge integration.

Adapted from the auto-judge-starter-kit test suite for this repo's layout:
the judge lives in trec-auto-judge/ and its `judges.ir_axioms.*` module path
only exists inside the Docker container (see trec-auto-judge/Dockerfile), so
the class reference is checked against the source file rather than imported.
Framework packages (autojudge-base, tira) are provided by the container base
image; when they happen to be installed locally, their versions are checked
against the upstream template's minimum pins.
"""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml needed for workflow checks")

REPO = Path(__file__).parent.parent
WORKFLOW = REPO / "trec-auto-judge" / "workflow.yml"

# Fallback pins mirroring the upstream template (trec-auto-judge/auto-judge-starter-kit
# pyproject.toml); bump when the template raises its requirements. A live value is
# preferred when a `starterkit` (or `upstream`) git remote exists — add one with:
#   git remote add starterkit git@github.com:trec-auto-judge/auto-judge-starter-kit.git
TEMPLATE_MINIMUMS = {"autojudge-base": "0.3.18", "tira": "0.0.100"}


def _template_minimum(package: str, fallback: str) -> str:
    """The template's current pin from a starterkit/upstream remote, else the fallback."""
    import subprocess
    for remote in ("starterkit", "upstream"):
        try:
            subprocess.run(
                ["git", "fetch", "--quiet", remote, "main"],
                cwd=REPO, capture_output=True, timeout=10, check=False,
            )
            out = subprocess.run(
                ["git", "show", f"{remote}/main:pyproject.toml"],
                cwd=REPO, capture_output=True, text=True, check=True,
            ).stdout
            m = re.search(rf'"{re.escape(package)}\s*>=\s*([0-9][0-9a-zA-Z.]*)"', out)
            if m:
                return m.group(1)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError):
            continue
    return fallback


def test_workflow_parses():
    """workflow.yml parses and declares a judge_class — what auto-judge run needs."""
    cfg = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert cfg.get("judge_class"), "workflow.yml declares no judge_class"
    measures = cfg.get("settings", {}).get("evaluation_measures", {})
    assert measures, "workflow.yml declares no evaluation_measures"
    undescribed = [name for name, spec in measures.items()
                   if not (spec or {}).get("description")]
    assert not undescribed, f"measures without description: {undescribed}"


def test_judge_class_source_exists():
    """The class named in judge_class exists in the judge source file.

    The module path (judges.ir_axioms.axiom_judge) is container-only, so the
    check resolves against trec-auto-judge/axiom_judge.py instead of importing.
    """
    cfg = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    module_path, _, class_name = cfg["judge_class"].partition(":")
    source = REPO / "trec-auto-judge" / (module_path.rsplit(".", 1)[-1] + ".py")
    assert source.exists(), f"judge source {source.name} not found next to workflow.yml"
    assert re.search(rf"^class {re.escape(class_name)}\b",
                     source.read_text(encoding="utf-8"), re.M), (
        f"{class_name} (from judge_class) not defined in {source.name}"
    )


@pytest.mark.parametrize("package,minimum", sorted(TEMPLATE_MINIMUMS.items()))
def test_framework_version_if_installed(package, minimum):
    """Locally installed framework packages meet the upstream template's pins."""
    packaging_version = pytest.importorskip("packaging.version")
    try:
        installed = packaging_version.Version(version(package))
    except PackageNotFoundError:
        pytest.skip(f"{package} not installed locally (provided by the container base image)")
    minimum = _template_minimum(package, minimum)
    assert installed >= packaging_version.Version(minimum), (
        f"installed {package} {installed} < {minimum} required by the upstream "
        f"starter-kit template — run: pip install --upgrade {package}"
    )
