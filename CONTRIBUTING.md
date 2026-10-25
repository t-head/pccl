<!-- Modified for PCCL. -->

# Contributing to PCCL

Thank you for your interest in contributing to PCCL! We appreciate the time and effort you're putting into helping improve the library. This document will help guide you through the contribution process.

## Before You Start

To help ensure your contribution can be accepted smoothly:

- **Open an issue first** for significant changes or new features. This allows us to discuss the approach before you invest significant time.
- **Check existing issues** to see if someone else is already working on something similar.
- **Ask questions** if you're unsure about anything. We're happy to help!
- **License**: Follow the licenses applicable to the files you change and preserve existing copyright and third-party notices. See [LICENSE.txt](LICENSE.txt) and [ThirdPartyNotices.txt](ThirdPartyNotices.txt).

## Getting Started

1. **Fork the Repository**: Fork [https://github.com/t-head/pccl](https://github.com/t-head/pccl) and clone your fork locally.

2. **Create a Branch**: Create a working branch for your change.
   Prefer `<type>/<short-description>`, using lowercase letters, digits, and hyphens to briefly describe the change
   (e.g. `fix/buffer-size`, `perf/allreduce-8-ranks`, or `docs/issue-template`).

   ```bash
   git checkout -b <my-feature-branch>
   ```

3. **Commit your Changes**: Commit your changes and push them to the feature branch
   ```bash
   git add <files-to-commit>
   git commit -s -m "Summarize your change" -m "Explain the problem, solution, and relevant limitations." -m "Refs #<issue-number>"
   git push origin <my-feature-branch> --set-upstream
   ```
## What We're Looking For

We welcome contributions in the following areas:

- **Bug fixes**: Reproducible bugs with clear fixes are always appreciated
- **Performance improvements**: Targeted optimizations, guarded by appropriate checks
- **Platform support**: Extending PCCL to new platforms, network fabrics, or PCI bridges
- **Documentation**: Improvements to README, comments, or usage examples

## Making Your Contribution Successful

To help us review and integrate your contribution efficiently:

### Scope and Focus
- Keep changes focused on a single issue or feature
- Break large changes into smaller, reviewable pieces when possible
- Avoid mixing unrelated changes (e.g., bug fix + formatting changes) in one PR

### Quality Standards
- Ensure your code compiles without warnings
- Test thoroughly on relevant hardware configurations
- Performance claims should be backed by measurements
- Public API changes require discussion (see below)

### Common Challenges

Some types of contributions require extra discussion before we can accept them:

- **Public API changes**: PCCL's API stability is important to users. If your contribution changes public APIs, please open an issue to discuss the design first. We need to ensure backward compatibility and consistency.

- **Architecture-specific code**: PCCL supports specific PPU architectures and network configurations. Contributions targeting unsupported architectures may not be accepted unless there's a clear plan for ongoing maintenance.

- **Workarounds for external issues**: If a change works around a problem in another component (drivers, libraries, frameworks), we may need to address it differently. Let's discuss the root cause first.

- **Incomplete implementations**: Partial features or fixes that don't fully address the problem are difficult to maintain. If you need help completing an implementation, let us know - we're happy to assist!

## New Features

If you're proposing a substantial new feature (e.g., new collective operations, transport mechanisms, algorithms, or significant architectural changes), we follow a more structured process:

1. **Open an issue** describing the feature and use case
2. **Discussion phase**: We'll discuss whether it fits PCCL's direction
3. **Document Design**: Document the proposed architecture, a testing and validation plan as well as any known performance impact.
4. **Review**: The design will be reviewed by PCCL maintainers
5. **Implementation**: We recommend that you wait with the implementation until the review has concluded. Once there's a mutual agreement on the approach, proceed with implementation
6. **Pull request**: Submit your PR referencing the original issue and design doc


## Code Style

PCCL follows these coding conventions:

### General Guidelines
- Use 2 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Follow K&R brace style for C++/HGGC code
- Use clear, descriptive variable names

### Naming Conventions
- Follow the existing naming conventions of the language and module, including type suffixes such as `_t`.
- Keep the `nccl` prefix for internal interfaces ported from NCCL, including hardware-adapted implementations.
- Use the `pccl` prefix for PCCL-specific interfaces and public PCCL APIs and types.

### PPU-Specific
- Minimize register usage in performance-critical kernels
- Avoid warp divergence where possible
- Document kernel launch configurations and occupancy considerations
- Prefer `hggcLaunchKernel` over `<<<>>>` syntax

### Return Codes
- Use `pcclResult_t` for public APIs that report operation status; internal interfaces may use its alias `ncclResult_t`.
- Check returned status with `NCCLCHECK` or an appropriate error-handling helper.
- Check external API errors with helpers appropriate to their return conventions, such as `HGGCCHECK`, `SYSCHECK`, or `PTHREADCHECK`.

### Memory Management
- Use the existing allocation helpers for the required memory domain, such as `ncclCalloc()` for host memory, and the matching deallocation path.
- Use appropriate memory fences for synchronization
- Release resources in dependency order after all users have finished accessing them.

### Comments and Documentation
- Write clear comments for complex algorithms
- Document assumptions and invariants
- Explain "why" not just "what" for non-obvious code

### Code Formatting
- Avoid trailing whitespace
- Try to match the existing style in files you're modifying

## Pull Request Guidelines

Use a clear, descriptive PR title. A type prefix such as `feat:`, `fix:`, or `docs:` is recommended but not required. The 50-character commit-title recommendation below does not apply to PR titles. For title capitalization, follow the recommendation under Commit Messages.

### Before Submitting
Reference at least one open issue in the PR description to satisfy the organization-level check. Use `Refs #123` for partial work and `Closes #123` only when resolving the issue.

1. Consider rebasing onto the latest target branch before submitting. Being behind alone does not block submission; resolve conflicts and address relevant integration issues before merging.
2. Set up the T-Head SAIL SDK and run `./build.sh` as described in [README.md](README.md) to ensure a clean build with no warnings. Use [pccl-tests](https://github.com/t-head/pccl-tests) for relevant functional and performance validation.

Each PR must contain 1 to 5 contribution commits relative to its target branch. Every commit must meet the message and DCO requirements below. Squash excess commits before pushing, preserving existing author attribution and authorized DCO sign-offs.

### Commit Messages

Write concise English messages that describe the complete change.

**Required**

- Use a nonempty imperative title and leave the second line blank.
- Explain the problem, solution, and relevant limitations. For breaking changes, include the impact and migration.
- Reference a real, relevant issue. Use `Refs #123` for related or partial work; use `Closes` or `Fixes` only when resolving it and intending closure. Cross-repository references (`Refs owner/repo#123`) and full GitHub issue URLs are accepted.
- Include an author-matching DCO sign-off as described under [Signed Commits](#signed-commits).

**Recommended**

- Prefer a suitable `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, or `chore` prefix. Aim for 50 characters including the prefix; longer titles are accepted.
- Prefer 1–3 short bullet points describing the main changes and their purpose. Use prose paragraphs when they explain the change more clearly.
- With a prefix, prefer a lowercase summary initial; without one, an initial capital is conventional. Preserve proper nouns, acronyms, and code identifiers.
- Place the issue reference immediately before the sign-off block, with a blank line before this group. A blank line between the reference and sign-offs is also accepted.

These are writing preferences, not additional acceptance requirements.

Example (the issue number and identity are placeholders):

```text
fix: reject invalid buffer sizes

- reject negative sizes before allocating buffers
- return an error instead of continuing with invalid input

Refs #123
Signed-off-by: Your Name <your@email.com>
```

### Signed Commits
All commits must be signed off to certify you have the right to submit the code:
```bash
git commit -s -m "Summarize your change" -m "Explain the problem, solution, and relevant limitations." -m "Refs #<issue-number>"
```
This adds: `Signed-off-by: Your Name <your@email.com>`

At least one `Signed-off-by` line must exactly match the commit author name and email.
GitHub noreply addresses are accepted when both fields match; your GitHub account email is not checked.

This certifies your agreement with the Developer Certificate of Origin (DCO).

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Code Review Process

After you submit a PR:

1. **Maintainer review**: PCCL maintainers will review your code
2. **Discussion**: We may ask questions or request changes
3. **Iteration**: Address feedback and update your PR
4. **Approval**: Once approved, we'll merge your contribution

## Communication

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions and discussion

## Getting Help

If you need help at any stage:
- Comment on your issue or PR
- Ask questions in GitHub Issues
- Refer to [README.md](README.md), examples, and the [PCCL documentation](https://developer.t-head.cn/docs_center/doc_detail/index.html?projectId=35)

We're committed to helping contributors succeed!

## Thank You

We truly appreciate your time and effort in contributing to PCCL.

Happy coding!
