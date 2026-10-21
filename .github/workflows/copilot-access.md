---
name: PCCL Copilot access (gh-aw)
on:
  pull_request:
    branches: [test/module-review-base]
    types: [opened, synchronize, reopened]
  reaction: none
  status-comment: false
if: >-
  github.event_name == 'pull_request' &&
  github.repository == 't-head/pccl' &&
  github.actor == 'honganlxh' &&
  github.event.pull_request.number == 19 &&
  github.event.pull_request.user.login == 'honganlxh' &&
  github.event.pull_request.base.ref == 'test/module-review-base' &&
  github.event.pull_request.head.repo.full_name == github.repository &&
  github.event.pull_request.head.ref == 'test/module-review-pr'
engine: copilot
permissions:
  contents: read
  copilot-requests: write
checkout: false
timeout-minutes: 5
tools:
  github: false
  bash: []
  cli-proxy: false
# Keep this access probe silent, including framework error reports.
safe-outputs:
  staged: true
  report-failure-as-issue: false
  report-failed-jobs: false
  report-incomplete:
    create-issue: false
  missing-tool:
    create-issue: false
  missing-data:
    create-issue: false
  noop:
    report-as-issue: false
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All Rights Reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This isolated test verifies organization-billed Copilot access through GitHub
Agentic Workflows. Complete with a noop output whose message is exactly
`PCCL_GH_AW_COPILOT_ACCESS_OK`.
Do not inspect repository content, use other tools, post comments, or request reviewers.
