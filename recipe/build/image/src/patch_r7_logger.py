#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

import ast
import hashlib
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
TARGET = SP + "/vllm_inproj_ba_gemv.py"
REFERENCE = SP + "/vllm_ple_mmap.py"

OLD_LOGGER = 'logger = init_logger(__name__)'
NEW_LOGGER = 'logger = logging.getLogger("vllm.inproj_ba_gemv")'
OLD_IMPORT = '\nimport os\n'
NEW_IMPORT = '\nimport logging\nimport os\n'
DONE_MARK = NEW_LOGGER

src = open(TARGET).read()
before_sha = hashlib.sha256(src.encode()).hexdigest()

if DONE_MARK in src:
    print("patch_r7_logger.py: already applied, skipping", file=sys.stderr)
    sys.exit(0)

# The reference binding must be exactly what this patch is copying. If
# vllm_ple_mmap.py ever stops using the plain stdlib logger, the premise of this
# patch -- "bind it the way the module whose lines DO reach the log binds it" --
# is void and the build must stop rather than guess.
ref = open(REFERENCE).read()
assert 'logger = logging.getLogger("vllm.ple_mmap")' in ref, (
    "vllm_ple_mmap.py no longer binds logging.getLogger(\"vllm.ple_mmap\"): "
    "the reference this patch copies has changed")
assert "\nimport logging\n" in ref, "vllm_ple_mmap.py no longer imports logging"

# Both anchors must be present exactly once.
for anchor, name in ((OLD_LOGGER, "logger binding"), (OLD_IMPORT, "import os block")):
    n = src.count(anchor)
    assert n == 1, "%s: expected exactly 1 %s, found %d -- anchor moved" % (
        TARGET, name, n)

# The install and disable lines this fix exists to surface must still be there.
for anchor in ('"iter6 R7 installed: %d in_proj_ba layers ',
               'logger.info("iter6 R7 disabled")'):
    assert anchor in src, "%s: R7 log call site moved -- %r" % (TARGET, anchor)

src = src.replace(OLD_IMPORT, NEW_IMPORT, 1)
src = src.replace(OLD_LOGGER, NEW_LOGGER, 1)

ast.parse(src)                       # refuse to write a file that will not parse
open(TARGET, "w").write(src)
after = open(TARGET).read()
ast.parse(after)
assert NEW_LOGGER in after and OLD_LOGGER not in after
assert after.count("\nimport logging\n") == 1

print("patch_r7_logger.py applied OK", file=sys.stderr)
print("  %s" % TARGET, file=sys.stderr)
print("  sha256 before %s" % before_sha, file=sys.stderr)
print("  sha256 after  %s" % hashlib.sha256(after.encode()).hexdigest(), file=sys.stderr)
