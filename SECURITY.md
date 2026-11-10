# Security

This project claims that its checkers catch any GEMV or sampling error larger than a
proven bound, and that the slot-code sandbox stops an RL policy from running anything
but a kernel body. A way around either is a security bug here.

## In scope

- A wrong output that the checker passes (a false negative beyond the proven bound), or
  a correct output it rejects (a false alarm).
- A gap between the code and the Lean theorems' hypotheses (rounding direction, error
  constants, overflow).
- Slot code that the sandbox (`src/pck/agent/sandbox.py`) accepts but that runs host
  code, reads the timer or checker, or changes anything outside the kernel body.
- A change to a trusted file that the hash pin (`src/pck/manifest.py`) does not detect.

Errors within the proven bound are expected, not bugs.

## Reporting

Please report privately through GitHub's
[private vulnerability reporting](../../security/advisories/new) rather than a public
issue. Include the shapes, inputs (or a seed) and the command that shows the problem.
I aim to reply within a week.

## Supported versions

Only `main` is supported.
