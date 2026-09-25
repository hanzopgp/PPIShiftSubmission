# E003 execution-source recovery

Status: partial file identity recovered; the full dirty working-tree patch is
not reconstructable from the repository objects available on the submission
branch.

## Evidence boundary

The final scientific summary and reviewer red-team record the executed cluster
checkout as commit
`withheld-for-anonymous-review` with deliberate uncommitted
execution-script changes. They also record the SHA-256 of the analysis script
used for the result pack as
`8de1ee230776a077e40bd254b506deaf28c8ea2f5b1586c6dab33c1f98f85f6e`.

The current committed `scripts/analyze_e003_results.py` has that exact
SHA-256, so the analysis-script file identity is recoverable. The committed
file at the recorded clean checkout does not have that SHA-256, confirming
that the executed source differed from the clean checkout. The later E003
evidence-pack commit contains the matching analysis-script file, but the
available evidence does not prove that it captures every file in the original
dirty working-tree patch or its exact uncommitted diff.

No cluster checkout or stash containing the original working tree is available
in this workspace, and the reachable repository history does not contain a
commit or patch object identified as that complete dirty state. Reconstructing
an equivalent-looking patch would therefore overstate provenance.

## Consequence

This document deliberately does not claim exact execution-source recovery and
does not alter any frozen result, registry, panel, checkpoint, or figure-data
artifact. The submission figures are new implementation artifacts derived
from the source-adjacent E003 tables and the canonical measured-result
registry; they are not presented as a retroactive reconstruction of the
cluster working tree.

For reproducibility, the submission should cite the recorded artifact hashes
and distinguish the exact analysis-script identity recovered above from the
unrecovered remainder of the historical dirty patch.
