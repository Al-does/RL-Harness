"""Dependency-direction safeguards for the generic harness."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROHIBITED_IMPORTS = {
    "harness": ("experiments", "envs"),
    "analysis": ("experiments", "envs"),
    "learners": ("experiments", "envs", "analysis"),
    "losses": ("experiments", "envs"),
    "envs": (
        "harness",
        "experiments",
        "learners",
        "losses",
        "analysis",
    ),
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def mutates_sys_path(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr in {"insert", "append"}
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "path"
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "sys"
        ):
            return True
    return False


def test_generic_package_dependencies_point_inward_only():
    violations = []
    for package, prohibited in PROHIBITED_IMPORTS.items():
        for path in (ROOT / package).rglob("*.py"):
            if "tests" in path.parts:
                continue
            invalid = imported_roots(path) & set(prohibited)
            if invalid:
                violations.append(
                    f"{path.relative_to(ROOT)} imports {sorted(invalid)}"
                )

    assert violations == []


def test_python_sources_do_not_mutate_sys_path():
    offenders = []
    for package in (
        "analysis",
        "devops",
        "envs",
        "harness",
        "learners",
        "losses",
        "tests",
    ):
        for path in (ROOT / package).rglob("*.py"):
            if mutates_sys_path(path):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_historical_root_results_archive_is_ignored():
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert "/results/" in ignore
