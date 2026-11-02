"""Fail the build if a component restricts commercial use of generated output.

Prints an inventory of installed packages and licences to stdout.
"""
import importlib.metadata as md
import re
import sys

BANNED = {"nvdiffrast", "nvdiffrec"}
RESTRICTED = re.compile(r"non-?commercial|research[- ]only|evaluation[- ]only|\bCC[- ]BY-NC", re.I)

failures = []
for dist in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    name = dist.metadata["Name"]
    meta = dist.metadata
    # Only short licence fields: a full licence text can mention these words in passing.
    fields = [meta.get("License-Expression") or "", (meta.get("License") or "")[:200]]
    fields += [c for c in meta.get_all("Classifier", []) if c.startswith("License")]
    text = "; ".join(f for f in fields if f)
    print(f"{name}=={dist.version}\t{text or 'UNKNOWN'}")
    if name.lower() in BANNED:
        failures.append(f"{name}: banned, its licence forbids commercial use")
    elif RESTRICTED.search(text):
        failures.append(f"{name}: licence metadata restricts use: {text}")

if failures:
    sys.exit("Licence check failed:\n  " + "\n  ".join(failures))
