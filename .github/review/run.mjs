// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import {readFileSync, appendFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {validatePolicy, classify, evaluate, reviewerRequests} from './policy.mjs';

export const CHECK = 'PCCL Module Owner Approval';
export class GitHub {
  constructor(token, repository) {
    if (!/^[\w.-]+\/[\w.-]+$/.test(repository)) throw new Error('Invalid repository');
    this.token = token;
    this.prefix = `/repos/${repository}`;
  }
  async request(method, path, body) {
    if (!path.startsWith(this.prefix + '/')) throw new Error('Request outside repository');
    const response = await fetch(`https://api.github.com${path}`, {
      method, redirect: 'error', signal: AbortSignal.timeout(30000),
      headers: {Authorization: `Bearer ${this.token}`, Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28', 'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      const error = new Error(`GitHub ${method} ${path.split('?')[0]} returned ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return response.status === 204 ? null : response.json();
  }
  async list(path) {
    const result = [];
    for (let page = 1; page <= 100; page++) {
      const separator = path.includes('?') ? '&' : '?';
      const data = await this.request('GET', `${this.prefix}${path}${separator}per_page=100&page=${page}`);
      if (!Array.isArray(data)) throw new Error('Expected a paginated array');
      result.push(...data);
      if (data.length < 100) return result;
    }
    throw new Error('Pagination safety limit reached; refusing partial results');
  }
}

export function fingerprint(pr, files, reviews, permissions, policy) {
  return createHash('sha256').update(JSON.stringify({head: pr.head.sha, base: pr.base.sha,
    target: pr.base.ref, body: pr.body, labels: pr.labels.map(l => l.name).sort(),
    draft: pr.draft, state: pr.state, files, reviews, permissions, policy})).digest('hex');
}

async function snapshot(api, policy, number) {
  const pr = await api.request('GET', `${api.prefix}/pulls/${number}`);
  if (pr.changed_files > 3000) throw new Error('GitHub lists at most 3000 files; split this PR');
  const [files, reviews] = await Promise.all([
    api.list(`/pulls/${number}/files`), api.list(`/pulls/${number}/reviews`),
  ]);
  if (files.length !== pr.changed_files) throw new Error('Incomplete file list');
  const scope = classify(policy, files, pr.body, pr.labels.map(l => l.name));
  const users = new Set(scope.modules.flatMap(id => {
    const m = policy.modules[id];
    return [m.maintainer, m.backup].map(p => policy.people[p].github.toLowerCase());
  }));
  reviews.filter(r => r.user?.type === 'User').forEach(r => users.add(r.user.login.toLowerCase()));
  const permissions = {};
  for (const user of users) {
    try {
      const info = await api.request('GET', `${api.prefix}/collaborators/${encodeURIComponent(user)}/permission`);
      permissions[user] = info.permission;
    } catch (error) {
      if (error.status !== 404) throw error;
      permissions[user] = 'none'; // Public reviewers need not be collaborators.
    }
  }
  return {pr, files, reviews, scope, permissions,
    fingerprint: fingerprint(pr, files, reviews, permissions, policy)};
}

function summary(state, result, warnings) {
  const unmapped = state.scope.unmatched.slice(0, 20).map(p => JSON.stringify(p)).join(', ');
  return [
    `PR #${state.pr.number}; HEAD \`${state.pr.head.sha}\`.`,
    'One current-head approval from any affected module\'s Maintainer OR Backup satisfies owner approval.',
    result.approvedBy.length
      ? `Owner approval received from: ${result.approvedBy.join(', ')}.`
      : 'Awaiting one eligible owner approval across the affected modules.',
    ...result.modules.map(m => `- ${m.id}: eligible owners ${m.owners.join(' or ')}` +
      (m.approvedBy.length ? '; approved by ' + m.approvedBy.join(', ') : '')),
    ...result.errors.map(e => `- BLOCKED: ${e}`),
    ...(state.scope.unmatched.length ? [
      `Unmapped paths were routed to the triage module. Verify their semantic scope: ${unmapped}` +
        (state.scope.unmatched.length > 20 ? ' (truncated)' : ''),
    ] : []),
    ...warnings.map(w => `- Notice: ${w}`),
    'Other required CI/review rules still apply. This check does not grant merge permission.',
  ].join('\n\n');
}

export async function reconcile(api, policy, number, openPRs, detailsUrl) {
  let check;
  let pr;
  try {
    pr = await api.request('GET', `${api.prefix}/pulls/${number}`);
    if (pr.state !== 'open' || pr.base.ref !== policy.targetBranch) return;
    check = await api.request('POST', `${api.prefix}/check-runs`, {
      name: CHECK, head_sha: pr.head.sha, status: 'in_progress', details_url: detailsUrl,
      external_id: `pccl-review:${number}:${pr.head.sha}`,
    });
    const state = await snapshot(api, policy, number);
    if (state.pr.head.sha !== pr.head.sha) throw new Error('Head changed; rerun for the new commit');
    const result = evaluate(policy, state.pr, state.scope, state.reviews, state.permissions);
    if (openPRs.filter(p => p.head.sha === pr.head.sha).length !== 1) {
      result.errors.push('Multiple open PRs share this HEAD; use distinct commits to avoid sharing a check result');
      result.pass = false;
    }
    const warnings = [];
    if (!pr.draft) {
      for (const reviewer of reviewerRequests(policy, state.pr, state.scope, state.reviews, state.permissions)) {
        try {
          await api.request('POST', `${api.prefix}/pulls/${number}/requested_reviewers`, {reviewers: [reviewer]});
        } catch (error) { warnings.push(`Could not request ${reviewer}: ${error.message}`); }
      }
    }
    // Re-read authoritative state. Event payloads and request-review success are not approval evidence.
    const fresh = await snapshot(api, policy, number);
    if (fresh.fingerprint !== state.fingerprint) throw new Error('PR/reviews changed during evaluation; rerun');
    const currentPRs = await api.list('/pulls?state=open');
    if (currentPRs.filter(p => p.head.sha === pr.head.sha).length !== 1) {
      throw new Error('Multiple open PRs share this HEAD; use a distinct commit for each PR');
    }
    const output = summary(fresh, result, warnings);
    await api.request('PATCH', `${api.prefix}/check-runs/${check.id}`, {
      status: 'completed', conclusion: result.pass ? 'success' : 'failure',
      output: {title: result.pass ? 'Module owner approval received' : 'Module approval required', summary: output},
    });
    if (process.env.GITHUB_STEP_SUMMARY) appendFileSync(process.env.GITHUB_STEP_SUMMARY, output + '\n');
    return result;
  } catch (error) {
    if (check) await api.request('PATCH', `${api.prefix}/check-runs/${check.id}`, {
      status: 'completed', conclusion: 'failure',
      output: {title: 'Unable to verify module approval', summary: error.message},
    });
    throw error;
  }
}

export async function run(api, policy, eventName, event, detailsUrl) {
  const openPRs = await api.list('/pulls?state=open');
  let targets;
  if (eventName === 'pull_request_target') targets = [event.pull_request.number];
  else if (eventName === 'workflow_dispatch') {
    const n = Number(event.inputs?.pr_number);
    if (!Number.isSafeInteger(n) || n <= 0) throw new Error('Invalid PR number');
    targets = [n];
  } else if (eventName === 'workflow_run') {
    const run = await api.request('GET', `${api.prefix}/actions/runs/${event.workflow_run.id}`);
    const workflow = await api.request('GET', `${api.prefix}/actions/workflows/${run.workflow_id}`);
    if (workflow.path !== '.github/workflows/module-review-events.yml' || run.event !== 'pull_request_review') {
      throw new Error('Unexpected review-event relay');
    }
    // No artifacts, caches, or code from this unprivileged run are consumed.
    // Fork payloads can omit PR associations; merge refs can differ from the real HEAD.
    // Reconcile all open target PRs when no reliable association is available.
    const associated = new Set((run.pull_requests ?? []).map(p => p.number));
    targets = openPRs.filter(p => associated.has(p.number) || p.head.sha === run.head_sha).map(p => p.number);
    if (!targets.length) targets = openPRs.filter(p => p.base.ref === policy.targetBranch).map(p => p.number);
  } else if (eventName === 'push' || eventName === 'schedule') {
    targets = openPRs.map(p => p.number); // Policy/permission changes and missed webhooks.
  } else throw new Error(`Unsupported event: ${eventName}`);
  const failures = [];
  for (const number of targets) {
    try { await reconcile(api, policy, number, openPRs, detailsUrl); }
    catch (error) { failures.push(`PR #${number}: ${error.message}`); }
  }
  if (failures.length) throw new Error(failures.join('\n'));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const policy = validatePolicy(JSON.parse(readFileSync(new URL('./owners.json', import.meta.url))));
    const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH));
    const api = new GitHub(process.env.GITHUB_TOKEN, process.env.GITHUB_REPOSITORY);
    if (!api.token) throw new Error('GITHUB_TOKEN is required');
    await run(api, policy, process.env.GITHUB_EVENT_NAME, event,
      `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`);
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
