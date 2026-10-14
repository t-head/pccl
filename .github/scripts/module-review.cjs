// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
'use strict';

const owners = {
  'Env Check': ['wensuyang'],
  'Dispatch Strategy': ['honganlxh'],
};

function selectedModules(body) {
  // Comments and code samples are not declarations.
  const text = (body || '').replace(/<!--[\s\S]*?(?:-->|$)/g, '');
  let fence = null;
  const lines = text.split(/\r?\n/).filter(line => {
    const marker = line.match(/^\s{0,3}(`{3,}|~{3,})/);
    if (marker) {
      if (!fence) fence = marker[1];
      else if (marker[1][0] === fence[0] && marker[1].length >= fence.length) fence = null;
      return false;
    }
    return !fence;
  });
  const headings = lines.flatMap((line, i) => /^## Affected Modules\s*$/.test(line) ? [i] : []);
  if (headings.length !== 1) throw new Error('Include exactly one "## Affected Modules" section.');
  const section = [];
  for (const line of lines.slice(headings[0] + 1)) {
    if (/^#{1,2}\s/.test(line)) break;
    section.push(line);
  }
  const selected = [...new Set(section.flatMap(line => {
    const match = line.match(/^\s{0,3}- \[[xX]\] (.+?)\s*$/);
    return match ? [match[1]] : [];
  }))];
  if (!selected.length) throw new Error('Select at least one affected module.');
  for (const module of selected) {
    if (!Object.hasOwn(owners, module)) throw new Error(`Unknown module: ${module}`);
  }
  return selected;
}

async function route({github, context, core}) {
  const repo = context.repo;
  const pull_number = context.payload.pull_request.number;
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number});
  if (pr.state !== 'open') return;
  const status = async (state, description) => github.rest.repos.createCommitStatus({
    ...repo, sha: pr.head.sha, state, context: 'PR Modules', description,
    target_url: `${context.serverUrl}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`,
  });
  await status('pending', 'Validating module selection and reviewer routing');
  try {
    const selected = selectedModules(pr.body);
    const desired = [...new Set(selected.flatMap(module => owners[module]))];
    const skipped = desired.filter(login => login.toLowerCase() === pr.user.login.toLowerCase());
    const {data: requested} = await github.rest.pulls.listRequestedReviewers({...repo, pull_number});
    const reviews = await github.paginate(github.rest.pulls.listReviews, {...repo, pull_number, per_page: 100});
    const existing = new Set(requested.users.map(user => user.login.toLowerCase()));
    for (const review of reviews) {
      if (['APPROVED', 'CHANGES_REQUESTED'].includes(review.state) && review.user) {
        existing.add(review.user.login.toLowerCase());
      }
    }
    const reviewers = desired.filter(login => !skipped.includes(login) && !existing.has(login.toLowerCase()));
    if (!pr.draft && reviewers.length) {
      await github.rest.pulls.requestReviewers({...repo, pull_number, reviewers});
    }
    core.info(JSON.stringify({selected, requested: pr.draft ? [] : reviewers, skippedAuthor: skipped, draft: pr.draft}));
    await core.summary.addHeading('Module reviewer routing').addTable([
      [{data: 'Module', header: true}, {data: 'Configured reviewers', header: true}],
      ...selected.map(module => [module, owners[module].join(', ')]),
    ]).addRaw(`\nRequested: ${pr.draft ? 'none (draft)' : reviewers.join(', ') || 'none (already requested/reviewed or PR author)'}.\n`)
      .addRaw(`\nSkipped PR author: ${skipped.join(', ') || 'none'}. Existing review requests are preserved.\n`).write();
    await status('success', pr.draft ? 'Modules valid; reviewer requests deferred until ready' : 'Modules valid; reviewer routing completed');
  } catch (error) {
    await status('failure', 'Module selection or reviewer routing failed; see workflow logs');
    core.setFailed(error.message);
  }
}
module.exports = {owners, selectedModules, route};
