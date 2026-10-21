// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {assertPilotPR, assertPilotScope, pilotPolicy, PilotAPI, runPilot} from '../../.github/review/pilot.mjs';
import {CHECK} from '../../.github/review/run.mjs';
import {evaluate, reviewerRequests} from '../../.github/review/policy.mjs';

const source = JSON.parse(readFileSync(new URL('../../.github/review/owners.json', import.meta.url)));
const head = 'a'.repeat(40);
const files = [{filename: 'docs/workflow/module-review.md'}];
const makePR = () => ({number: 19, state: 'open', draft: false, user: {login: 'honganlxh'},
  head: {sha: head, ref: 'test/module-review-pr', repo: {full_name: 't-head/pccl'}},
  base: {sha: 'b'.repeat(40), ref: 'test/module-review-base', repo: {full_name: 't-head/pccl'}},
  body: 'PCCL-Modules: workflow\nPCCL-Areas: workflow-review', labels: [], changed_files: 1,
  requested_reviewers: []});
const permissions = {honganlxh: 'admin', wensuyang: 'write'};
const review = (login, state = 'APPROVED', commit = head) => ({id: 1, user: {login, type: 'User'},
  state, commit_id: commit, submitted_at: '2026-10-19T00:00:00Z'});

test('pilot only changes the target branch of the production policy', () => {
  const policy = pilotPolicy(source);
  assert.equal(source.targetBranch, 'main');
  assert.deepEqual({...policy, targetBranch: 'main'}, source);
});

test('pilot rejects other PRs, authors, repositories, branches and closed PRs', () => {
  for (const change of [p => {p.number = 20;}, p => {p.user.login = 'someone';},
    p => {p.head.repo.full_name = 'someone/pccl';}, p => {p.base.ref = 'main';},
    p => {p.head.ref = 'other';}, p => {p.state = 'closed';}]) {
    const pr = makePR(); change(pr);
    assert.throws(() => assertPilotPR(pr), /restricted/);
  }
});

test('pilot rejects paths, declarations and labels that would involve other modules', () => {
  const policy = pilotPolicy(source);
  assert.deepEqual(assertPilotScope(policy, makePR(), files).modules, ['workflow']);
  for (const setup of [p => {p.body += ', runtime';},
    p => {p.labels = [{name: 'module/device'}];}, p => {p.labels = [{name: 'backup/build'}];}]) {
    const pr = makePR(); setup(pr);
    assert.throws(() => assertPilotScope(policy, pr, files));
  }
  assert.throws(() => assertPilotScope(policy, makePR(), [{filename: 'src/init.cc'}]));
  assert.throws(() => assertPilotScope(policy, makePR(), []), /Incomplete/);
});

test('workflow routes the author to Backup once, including explicit backup labels', () => {
  const policy = pilotPolicy(source);
  const pr = makePR();
  pr.labels = [{name: 'backup/workflow'}];
  const scope = assertPilotScope(policy, pr, files);
  assert.deepEqual(reviewerRequests(policy, pr, scope, [], permissions), ['wensuyang']);
  pr.requested_reviewers = [{login: 'wensuyang'}];
  assert.deepEqual(reviewerRequests(policy, pr, scope, [], permissions), []);
});

test('Copilot access workflows stay within the two-person pilot', () => {
  const policy = pilotPolicy(source), pr = makePR();
  const probeFiles = ['copilot-access.md', 'copilot-access.lock.yml']
    .map(name => ({filename: `.github/workflows/${name}`}));
  pr.changed_files = probeFiles.length;
  const scope = assertPilotScope(policy, pr, probeFiles);
  assert.deepEqual(scope.modules, ['workflow']);
  assert.deepEqual(reviewerRequests(policy, pr, scope, [], permissions), ['wensuyang']);
  pr.requested_reviewers = [{login: 'wensuyang'}];
  assert.deepEqual(reviewerRequests(policy, pr, scope, [], permissions), []);

  pr.changed_files = 1;
  assert.throws(() => assertPilotScope(policy, pr,
    [{filename: '.github/workflows/pr-test-ppu.yml'}]));
});

test('only a current-head Backup approval works on the Maintainer-authored pilot', () => {
  const policy = pilotPolicy(source), pr = makePR(), scope = assertPilotScope(policy, pr, files);
  for (const reviews of [[], [review('honganlxh')], [review('wensuyang', 'COMMENTED')],
    [review('wensuyang', 'DISMISSED')], [review('wensuyang', 'CHANGES_REQUESTED')],
    [review('wensuyang', 'APPROVED', 'c'.repeat(40))]]) {
    assert.equal(evaluate(policy, pr, scope, reviews, permissions).pass, false);
  }
  assert.equal(evaluate(policy, pr, scope, [review('wensuyang')], permissions).pass, true);
});

test('pilot API blocks unrelated invitations, other PR writes, approvals and foreign checks', async () => {
  const calls = [];
  const backend = {prefix: '/repos/t-head/pccl', request: async (...args) => {calls.push(args); return {id: 5};}};
  const api = new PilotAPI(backend, makePR()), p = api.prefix;
  for (const [method, path, body] of [
    ['POST', `${p}/pulls/19/requested_reviewers`, {reviewers: ['ck-milo']}],
    ['POST', `${p}/pulls/19/requested_reviewers`, {reviewers: ['honganlxh']}],
    ['POST', `${p}/pulls/20/requested_reviewers`, {reviewers: ['wensuyang']}],
    ['POST', `${p}/pulls/19/reviews`, {event: 'APPROVE'}],
    ['POST', `${p}/pulls/19/requested_reviewers`, {team_reviewers: ['team']}],
    ['PATCH', `${p}/check-runs/100`, {conclusion: 'success'}],
    ['PUT', `${p}/pulls/19/merge`, {}],
  ]) await assert.rejects(() => api.request(method, path, body), /write rejected/);
  assert.equal(calls.length, 0);
  await api.request('POST', `${p}/pulls/19/requested_reviewers`, {reviewers: ['wensuyang']});
  const check = {name: CHECK, head_sha: head, external_id: `pccl-review:19:${head}`};
  await assert.rejects(() => api.request('POST', `${p}/check-runs`, {...check, head_sha: 'other'}));
  await api.request('POST', `${p}/check-runs`, check);
  await api.request('PATCH', `${p}/check-runs/5`, {status: 'completed', conclusion: 'failure'});
  assert.equal(calls.length, 3);
});

test('an invalid declaration publishes a failure without any reviewer requests', async () => {
  const calls = [], pr = makePR(); pr.body += '\nPCCL-Modules: workflow';
  const backend = {prefix: '/repos/t-head/pccl', list: async () => files,
    request: async (method, path, body) => {
      calls.push({method, path, body});
      return method === 'GET' ? pr : {id: 7};
    }};
  await assert.rejects(() => runPilot(backend, source, 'https://example.test/run'), /exactly one/);
  const writes = calls.filter(c => c.method !== 'GET');
  assert.equal(writes.length, 1);
  assert.equal(writes[0].body.head_sha, head);
  assert.equal(writes[0].body.conclusion, 'failure');
});
