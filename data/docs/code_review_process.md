# Code Review Process

Every change to a shared repository must go through a pull request; direct
pushes to `main` are disabled at the branch-protection level. A PR needs at
least one approval from someone outside the author's immediate team for
changes touching shared libraries, and one approval from anyone for
service-local changes.

Reviewers are expected to respond within 1 business day of being requested.
If no reviewer responds within 2 business days, the author may ping the
team channel and request a different reviewer. PRs that only fix
typos, formatting, or comments can be self-merged after CI passes, but must
still be opened as a PR for the audit trail.

CI must pass (unit tests, lint, type check) before a PR is eligible for
merge. Reviewers should leave one of three verdicts: "approve," "approve
with comments" (author may merge after addressing), or "request changes"
(author must re-request review after updates). Reviews should focus on
correctness, security, and maintainability first; style nits are optional
and should be marked as such.

Large PRs (over roughly 400 changed lines) should be split where possible.
If a large PR can't reasonably be split, the author should provide a
walkthrough comment summarizing the change's structure before requesting
review. Merged PRs are squash-merged by default, with the PR title becoming
the commit message.