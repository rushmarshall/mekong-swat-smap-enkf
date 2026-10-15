"""Every number the documentation states about this repository, checked against it.

Prose drifts silently: the test count in the README was wrong three times while the package
was being built, and it claimed nine modules beside its own list of eleven. A number a
reader can check is a number that should check itself.
"""

import ast
import re
from pathlib import Path

import pytest

from mekong_da.config import MEKONG_2024, OPTIMAL_CONFIG, REPO_ROOT, RESOLUTIONS, STATIONS

README = (REPO_ROOT / "README.md").read_text()


def claimed(pattern: str) -> int:
    """Pull the number the README states, so the test reads the same source a reader does."""
    m = re.search(pattern, README)
    assert m, f"README no longer states this claim: {pattern}"
    return int(m.group(1))


def test_fortran_translation_unit_count_is_what_the_readme_says():
    src = REPO_ROOT / "fortran" / "src"
    actual = len(list(src.glob("*.f"))) + len(list(src.glob("*.f90")))
    assert actual == claimed(r"(\d+) translation units")


def test_module_count_is_what_the_readme_says():
    words = {"Nine": 9, "Ten": 10, "Eleven": 11, "Twelve": 12, "Thirteen": 13}
    m = re.search(r"The package\. (\w+) modules", README)
    assert m, "the README no longer states a module count"
    assert m.group(1) in words, f"unrecognised module count word: {m.group(1)}"
    actual = len([p for p in (REPO_ROOT / "src" / "mekong_da").glob("*.py")
                  if p.name != "__init__.py"])
    assert actual == words[m.group(1)]


def test_document_count_is_what_the_readme_says():
    words = {"Seven": 7, "Eight": 8, "Nine": 9, "Ten": 10}
    m = re.search(r"docs/\s+(\w+) documents", README)
    assert m, "the README no longer states a document count"
    assert len(list((REPO_ROOT / "docs").glob("*.md"))) == words[m.group(1)]


def test_test_count_is_what_the_readme_says(request):
    """The README tells a reader how many tests to expect. It has to be true.

    Only meaningful when the whole suite was collected, so running one file does not fail
    on a count it was never going to reach.
    """
    collected = {item.nodeid.split("::")[0] for item in request.session.items}
    expected = {f"tests/{p.name}" for p in sorted((REPO_ROOT / "tests").glob("test_*.py"))}
    if collected != expected:
        pytest.skip("counts the suite only when the whole suite is collected")
    assert len(request.session.items) == claimed(r"(\d+) tests, Fortran build included")


def test_run_geometry_is_internally_consistent():
    """n_days must be the calendar, and the HRU-day total must follow from the geometry."""
    years = range(MEKONG_2024.start_year, MEKONG_2024.start_year + MEKONG_2024.n_years)
    calendar = sum(366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365 for y in years)
    assert MEKONG_2024.n_days == calendar
    assert MEKONG_2024.n_days * MEKONG_2024.n_hru == 55_276_510


def test_the_fixed_study_constants_have_not_moved():
    assert len(STATIONS) == 5
    assert len(RESOLUTIONS) == 3
    assert OPTIMAL_CONFIG.ensemble_size == 200
    assert OPTIMAL_CONFIG.seed == 42
    assert (OPTIMAL_CONFIG.inflation_factor, OPTIMAL_CONFIG.persistence_phi) == (1.02, 0.50)


@pytest.mark.parametrize("doc", sorted((Path(REPO_ROOT) / "docs").glob("*.md")),
                         ids=lambda p: p.name)
def test_no_em_dash_anywhere_in_the_documentation(doc):
    assert "—" not in doc.read_text(), f"{doc.name} contains an em dash"


def test_every_module_parses_and_carries_a_docstring():
    for path in sorted((REPO_ROOT / "src" / "mekong_da").glob("*.py")):
        tree = ast.parse(path.read_text())
        assert ast.get_docstring(tree), f"{path.name} has no module docstring"


def test_no_text_io_depends_on_the_locale():
    """Text I/O without an explicit encoding reads differently on different machines.

    Python picks the encoding from the locale: UTF-8 on a typical Linux or macOS box,
    cp1252 on a typical Windows one. A reach file that parses here would then raise a
    UnicodeDecodeError there, on the same bytes. Pin it, so the behaviour is the same
    everywhere and a genuinely corrupt byte fails identically on every platform.
    """
    import re

    offenders = []
    roots = [REPO_ROOT / "src" / "mekong_da", REPO_ROOT / "scripts",
             REPO_ROOT / "analyses"]
    # binary mode has no encoding, and webbrowser.open is not file I/O
    pattern = re.compile(r"(?<![\w.])(open|read_text|write_text)\s*\(")
    binary = re.compile(r"""["'][rwax]b\+?["']""")
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            lines = path.read_text(encoding="utf-8").splitlines(True)
            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#") or "subprocess" in stripped:
                    continue
                if "webbrowser.open" in stripped or binary.search(stripped):
                    continue
                if not pattern.search(stripped):
                    continue
                # a call split across lines carries its encoding on a later line
                block = "".join(lines[lineno - 1:lineno + 8])
                if "encoding=" in block.split(")")[0] or "encoding=" in block:
                    continue
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {stripped}")
    assert not offenders, "text I/O without an explicit encoding:\n  " + "\n  ".join(offenders)


def test_the_model_is_not_invoked_through_a_posix_only_path():
    """`./swat_da` works on POSIX and not on Windows; an absolute path works on both.

    The Python half of this package is meant to be platform-neutral even though the model
    itself has only been executed on macOS and Linux. Launching by a relative shell path is
    the one thing that would have made that claim false.
    """
    source = (REPO_ROOT / "src" / "mekong_da" / "fortran.py").read_text(encoding="utf-8")
    assert 'f"./{' not in source, "the model is launched through a POSIX-only relative path"
    assert "[str(local_exe)]" in source, "the model should be launched by absolute path"


def test_every_tracked_directory_is_described_in_the_readme():
    """A directory a reader can see and the README never mentions is a question mark.

    `scripts/` sat unexplained in the layout for exactly that reason: two useful tools that
    a newcomer had no way to know were there.
    """
    import re
    import subprocess

    tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.split()
    directories = {path.split("/")[0] for path in tracked if "/" in path}
    block = re.search(r"## Repository layout(.*?)\n---", README, re.S)
    assert block, "the README no longer has a Repository layout section"
    undocumented = sorted(d for d in directories if f"{d}/" not in block.group(1))
    assert not undocumented, f"top-level directories missing from the layout: {undocumented}"
