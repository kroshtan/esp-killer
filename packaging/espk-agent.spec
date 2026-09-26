# PyInstaller spec for the one-file agent binary: dist/espk-agent (Linux) or dist/espk-agent.exe (Windows).
# Build from the repo root with `make agent-build`
# (= uv run --group build pyinstaller packaging/espk-agent.spec --noconfirm --clean).
#
# The repo is one uv project, so the venv also holds the server's heavy dependencies. The agent never imports
# them, but they are excluded explicitly too, so a stray import fails loudly in the smoke test instead of
# silently adding tens of MB to the binary.
from pathlib import Path

ROOT = Path(SPECPATH).parent  # SPECPATH is injected by PyInstaller: the directory containing this file

EXCLUDES = [
    # our own non-agent code
    "server",
    "tools",
    "tests",
    # server / analysis stack
    "numpy",
    "pandas",
    "matplotlib",
    "PIL",
    "fastapi",
    "starlette",
    "uvicorn",
    "yaml",
    # dev tooling that lives in the same venv
    "pytest",
    "_pytest",
    "hypothesis",
    "mypy",
    "ruff",
    "pydoclint",
    "PyInstaller",
    # pulled in only by PyInstaller's own setuptools runtime hook; the agent never uses them
    "setuptools",
    "pkg_resources",
    "_distutils_hack",
    "distutils",
    # stdlib parts the agent does not use
    "tkinter",
    "unittest",
    "pydoc",
    "doctest",
    "lib2to3",
    "idlelib",
]

a = Analysis(
    [str(ROOT / "agent" / "__main__.py")],
    pathex=[str(ROOT)],
    # pydantic-settings imports tomllib inside a function; list it so it is always collected. pydantic,
    # pydantic-core and rich (incl. its lazily imported unicode tables) are handled by pyinstaller-hooks-contrib.
    hiddenimports=["tomllib"],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,  # keep asserts and docstrings: typer builds --help text from docstrings
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="espk-agent",  # PyInstaller appends .exe on Windows; the release workflow adds version and OS
    debug=False,
    strip=False,
    upx=False,  # UPX-packed executables trigger more antivirus false positives on Windows
    runtime_tmpdir=None,
    console=True,  # a console program: server admins run it from a terminal or as a service
)
