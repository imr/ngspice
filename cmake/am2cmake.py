#!/usr/bin/env python3
"""
am2cmake.py -- extract the source lists of the autotools build into a
CMake include file.

The autotools build system remains the reference.  Rather than duplicating
~90 source lists by hand (and letting them rot), this script parses every
src/**/Makefile.am and emits cmake/NgspiceSourceLists.cmake, which the
CMake build includes.

Run it whenever a Makefile.am changes:

    python3 cmake/am2cmake.py

The generated file is committed to the repository so that building ngspice
with CMake needs no Python.

Automake conditionals are translated into CMake if() blocks using the
mapping in AM_COND_MAP below.  An unknown conditional aborts the run --
that is deliberate, so a new automake conditional cannot silently drop
sources from the CMake build.
"""

import os
import re
import sys

TOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(TOP, "src")
OUT = os.path.join(TOP, "cmake", "NgspiceSourceLists.cmake")

# automake conditional -> CMake condition
AM_COND_MAP = {
    "XSPICE_WANTED":   "NGSPICE_ENABLE_XSPICE",
    "OSDI_WANTED":     "NGSPICE_ENABLE_OSDI",
    "CIDER_WANTED":    "NGSPICE_ENABLE_CIDER",
    "NUMDEV_WANTED":   "NGSPICE_ENABLE_CIDER",
    "NDEV_WANTED":     "NGSPICE_ENABLE_NDEV",
    "KLU_WANTED":      "NGSPICE_ENABLE_KLU",
    "PSS_WANTED":      "NGSPICE_ENABLE_PSS",
    "SP_WANTED":       "NGSPICE_ENABLE_RFSPICE",
    "SENSE2_WANTED":   "NGSPICE_ENABLE_SENSE2",
    "CMATHTESTS":      "NGSPICE_ENABLE_CMATHTESTS",
    "NO_X":            "NOT NGSPICE_ENABLE_X11",
    "NO_HELP":         "NOT NGSPICE_ENABLE_HELP",
    "WINGUI":          "NGSPICE_WINGUI",
    "WINCONSOLE":      "WIN32 AND NOT NGSPICE_WINGUI",
    "WINRESOURCE":     "WIN32",
    "SHWIN":           "WIN32 AND NGSPICE_BUILD_SHARED",
    "SHCYG":           "FALSE",
    "TCLWIN":          "FALSE",
    "TCLCYG":          "FALSE",
    "TCL_MODULE":      "FALSE",
    "SHARED_MODULE":   "NGSPICE_BUILD_SHARED",
    "OLDAPPS":         "NGSPICE_ENABLE_OLDAPPS",
    "CROSS_COMPILING": "CMAKE_CROSSCOMPILING",
    "ADMS_WANTED":     "NGSPICE_ENABLE_ADMS",
    "RELPATH":         "NGSPICE_ENABLE_RELPATH",
    "DLIBS_FULLY_RESOLVED": "NOT APPLE",
    "MAINTAINER_MODE": "FALSE",
    "DEBUG_WANTED":    "FALSE",
}

# only these variables carry source files we care about
VAR_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+)_SOURCES\s*(?P<op>\+?=)\s*(?P<val>.*)$")
COND_RE = re.compile(r"^(if|else|endif)\s*(!?\s*[A-Za-z0-9_]*)")

# extensions that produce compilable output
KEEP_EXT = (".c", ".cpp", ".cc", ".y", ".rc")


def negate(cond):
    if cond.startswith("NOT ") and " AND " not in cond and " OR " not in cond:
        return cond[4:]
    return "NOT (%s)" % cond


def parse_makefile_am(path):
    """Return list of (target, condition_or_None, [files])."""
    with open(path, "r", errors="replace") as fh:
        raw = fh.read()

    # join continuation lines
    raw = raw.replace("\\\n", " ")
    out = []
    stack = []  # list of CMake conditions currently active

    for line in raw.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        m = COND_RE.match(line)
        if m and line.split()[0] in ("if", "else", "endif"):
            kw = m.group(1)
            arg = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            if kw == "if":
                neg = arg.startswith("!")
                name = arg.lstrip("!").strip()
                if name not in AM_COND_MAP:
                    sys.exit("%s: unknown automake conditional '%s' -- "
                             "add it to AM_COND_MAP" % (path, name))
                cond = AM_COND_MAP[name]
                stack.append(negate(cond) if neg else cond)
            elif kw == "else":
                if stack:
                    stack[-1] = negate(stack[-1])
            else:  # endif
                if stack:
                    stack.pop()
            continue

        m = VAR_RE.match(line.strip())
        if not m:
            continue

        files = [f for f in m.group("val").split()
                 if f.endswith(KEEP_EXT) and not f.startswith("$")]
        if not files:
            continue

        cond = " AND ".join("(%s)" % c for c in stack) if stack else None
        out.append((m.group("name"), cond, files))

    return out


def main():
    entries = {}          # varname -> list of (cond, [relative paths])
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d not in (".git", "icm", "adms")]
        if "Makefile.am" not in filenames:
            continue
        rel = os.path.relpath(dirpath, SRC).replace(os.sep, "/")
        if rel == ".":
            rel = ""
        for target, cond, files in parse_makefile_am(
                os.path.join(dirpath, "Makefile.am")):
            paths = [(rel + "/" + f if rel else f) for f in files]
            key = "NGSPICE_SRC_" + (rel.replace("/", "_") + "_" if rel else "") + target
            entries.setdefault(key, []).append((cond, paths))

    lines = [
        "# Generated by cmake/am2cmake.py -- DO NOT EDIT BY HAND.",
        "# Regenerate after changing any src/**/Makefile.am:",
        "#     python3 cmake/am2cmake.py",
        "",
    ]
    for key in sorted(entries):
        lines.append("set(%s" % key)
        # unconditional part first
        for cond, paths in entries[key]:
            if cond is None:
                for p in paths:
                    lines.append("    %s" % p)
        lines.append(")")
        for cond, paths in entries[key]:
            if cond is not None:
                lines.append("if(%s)" % cond)
                lines.append("    list(APPEND %s" % key)
                for p in paths:
                    lines.append("        %s" % p)
                lines.append("    )")
                lines.append("endif()")
        lines.append("")

    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s (%d source lists)" % (OUT, len(entries)))


if __name__ == "__main__":
    main()
