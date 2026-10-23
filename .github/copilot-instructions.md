# PCCL code review instructions

Review correctness and compatibility risks in the changed code through four questions:

- Can communication complete with correct results?
- Is concurrent and asynchronous execution safe, including synchronization and resource lifetimes?
- Are failures reported and propagated so affected work can stop or recover safely?
- Will existing applications and plugins remain compatible?

Apply only checks relevant to this change. Start with applicable entries in the
"AI Review Mandatory Rejections" category, then other relevant categories.
For each finding, explain the triggering conditions, affected code path, and
resulting impact. Keep optional style advice distinct from correctness findings;
do not turn recommendations into mandatory requirements.

When reviewing a pull request, read and apply the relevant checks in
[the shared review rules](../.agents/review_rules.yaml), and follow
[the contribution guidelines](../CONTRIBUTING.md).

The shared review rules are maintained in YAML for use by local skills and other
review platforms. The priorities above summarize that checklist; keep detailed
requirements in the shared file rather than duplicating them here.

If the referenced files cannot be read, report that limitation instead of claiming
their checks were applied. Base findings on the changed code and available
evidence; do not claim a build or test ran without evidence.
