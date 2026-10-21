// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {GitHub, run, CHECK} from './run.mjs';
import {validatePolicy} from './policy.mjs';
import {scopeWithRecommendation} from './recommend.mjs';

const repository = 't-head/pccl';
const number = 19;
const targetBranch = 'test/module-review-base';
const headBranch = 'test/module-review-pr';
const participants = new Set(['honganlxh', 'wensuyang']);

export function assertPilotPR(pr) {
  if (pr.number !== number || pr.state !== 'open' || pr.user.login !== 'honganlxh' ||
      pr.base.ref !== targetBranch || pr.head.ref !== headBranch ||
      pr.base.repo.full_name !== repository || pr.head.repo.full_name !== repository) {
    throw new Error('Pilot is restricted to the existing same-repository PR #19');
  }
}

export function pilotPolicy(source) {
  const policy = validatePolicy(structuredClone(source));
  const module = policy.modules.workflow;
  if (policy.people[module.maintainer].github !== 'honganlxh' ||
      policy.people[module.backup].github !== 'wensuyang') {
    throw new Error('Pilot requires the confirmed workflow Maintainer and Backup');
  }
  // Keep the complete production roster and rules; only the isolated target differs.
  policy.targetBranch = targetBranch;
  return policy;
}

export function assertPilotScope(policy, pr, files, recommendation) {
  if (files.length !== pr.changed_files || files.length === 0) throw new Error('Incomplete pilot file list');
  const scope = scopeWithRecommendation(policy, pr, files, recommendation);
  if (scope.modules.length !== 1 || scope.modules[0] !== 'workflow' || scope.unmatched.length ||
      scope.backups.some(id => id !== 'workflow')) {
    throw new Error('Live pilot covers only workflow / workflow-review');
  }
  return scope;
}

// Defense in depth: even a routing mistake cannot invite a third person or write to another PR.
export class PilotAPI {
  constructor(api, pr) {
    assertPilotPR(pr);
    this.api = api;
    this.prefix = api.prefix;
    this.head = pr.head.sha;
    this.author = pr.user.login.toLowerCase();
    this.checks = new Set();
    if (this.prefix !== `/repos/${repository}`) throw new Error('Wrong pilot repository');
  }
  list(path) { return this.api.list(path); }
  async request(method, path, body) {
    if (method === 'GET') return this.api.request(method, path, body);
    if (method === 'POST' && path === `${this.prefix}/check-runs`) {
      if (body.name !== CHECK || body.head_sha !== this.head ||
          body.external_id !== `pccl-review:${number}:${this.head}`) throw new Error('Unexpected pilot check');
      const check = await this.api.request(method, path, body);
      this.checks.add(check.id);
      return check;
    }
    if (method === 'PATCH' && [...this.checks].some(id => path === `${this.prefix}/check-runs/${id}`)) {
      return this.api.request(method, path, body);
    }
    if (method === 'POST' && path === `${this.prefix}/pulls/${number}/requested_reviewers` &&
        Object.keys(body).length === 1 && Array.isArray(body.reviewers) && body.reviewers.length &&
        body.reviewers.every(login => participants.has(login.toLowerCase()) && login.toLowerCase() !== this.author)) {
      return this.api.request(method, path, body);
    }
    throw new Error('Pilot write rejected: only PR #19 checks and its two participants are allowed');
  }
}

export async function runPilot(api, source, detailsUrl, recommendation) {
  const pr = await api.request('GET', `${api.prefix}/pulls/${number}`);
  assertPilotPR(pr);
  const guarded = new PilotAPI(api, pr);
  const policy = pilotPolicy(source);
  try {
    const files = await api.list(`/pulls/${number}/files`);
    const scope = assertPilotScope(policy, pr, files, recommendation);
    if (!scope.declared.length && !scope.recommendation) {
      throw new Error('Automatic classification is unavailable; add PCCL-Modules to the PR body and retry');
    }
    console.log(`Pilot PR #${number}; modules=${scope.modules}; areas=${scope.areas}; HEAD=${pr.head.sha}`);
  } catch (error) {
    // A malformed declaration must replace an older success, without requesting reviewers.
    await guarded.request('POST', `${api.prefix}/check-runs`, {
      name: CHECK, head_sha: pr.head.sha, external_id: `pccl-review:${number}:${pr.head.sha}`,
      status: 'completed', conclusion: 'failure', details_url: detailsUrl,
      output: {title: 'Pilot scope or declaration is invalid', summary: error.message},
    });
    throw error;
  }
  return run(guarded, policy, 'workflow_dispatch', {inputs: {pr_number: String(number)}}, detailsUrl, recommendation);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    if (process.env.GITHUB_REPOSITORY !== repository ||
        !['pull_request', 'pull_request_review'].includes(process.env.GITHUB_EVENT_NAME)) {
      throw new Error('Unexpected pilot event or repository');
    }
    const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH));
    assertPilotPR(event.pull_request);
    const api = new GitHub(process.env.GITHUB_TOKEN, repository);
    if (!api.token) throw new Error('GITHUB_TOKEN is required');
    const source = JSON.parse(readFileSync(new URL('./owners.json', import.meta.url)));
    const recommendation = process.env.PCCL_RECOMMENDATION ? JSON.parse(process.env.PCCL_RECOMMENDATION) : undefined;
    await runPilot(api, source,
      `${process.env.GITHUB_SERVER_URL}/${repository}/actions/runs/${process.env.GITHUB_RUN_ID}`, recommendation);
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
