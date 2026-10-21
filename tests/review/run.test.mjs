// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {GitHub, reconcile, run, CHECK} from '../../.github/review/run.mjs';
import {validatePolicy} from '../../.github/review/policy.mjs';

const policy = validatePolicy(JSON.parse(readFileSync(new URL('../../.github/review/owners.json', import.meta.url))));
const head = 'a'.repeat(40);
class FakeAPI {
  prefix = '/repos/t-head/pccl';
  calls = [];
  prReads = 0;
  pr = {number: 42, state: 'open', draft: false, user: {login: 'contributor'}, head: {sha: head},
    base: {ref: 'main', sha: 'b'.repeat(40)}, body: 'PCCL-Modules: kernels', labels: [], changed_files: 1,
    requested_reviewers: []};
  files = [{filename: 'src/collectives/device/all_reduce.cu'}];
  reviews = [{id: 1, state: 'APPROVED', commit_id: head, submitted_at: '2026-10-17T00:00:00Z',
    user: {login: 'ck-milo', type: 'User'}}];
  openPRs = [this.pr];
  relay = {workflow_id: 99, event: 'pull_request_review', head_sha: head, pull_requests: []};
  async request(method, path, body) {
    this.calls.push({method, path, body});
    if (path.endsWith('/pulls/42') && method === 'GET') {
      this.prReads++;
      this.onPRRead?.();
      return structuredClone(this.pr);
    }
    if (path.endsWith('/check-runs') && method === 'POST') return {id: 100};
    if (path.endsWith('/check-runs/100') && method === 'PATCH') return body;
    if (path.endsWith('/permission')) {
      if (this.permissionError) throw this.permissionError;
      return {permission: this.permission ?? 'write'};
    }
    if (path.endsWith('/requested_reviewers')) {
      if (this.requestError) throw new Error('Unavailable specialist');
      this.pr.requested_reviewers.push(...body.reviewers.map(login => ({login})));
      return {};
    }
    if (path.endsWith('/actions/runs/123')) return this.relay;
    if (path.endsWith('/actions/workflows/99')) {
      return {path: this.workflowPath ?? '.github/workflows/module-review-events.yml'};
    }
    throw new Error(`Unexpected request: ${method} ${path}`);
  }
  async list(path) {
    this.calls.push({method: 'LIST', path});
    if (path === '/pulls?state=open') return structuredClone(this.openPRs);
    if (path.endsWith('/files')) return structuredClone(this.files);
    if (path.endsWith('/reviews')) {
      if (this.reviewError) throw new Error('GitHub temporarily unavailable');
      return structuredClone(this.reviews);
    }
    throw new Error(`Unexpected list: ${path}`);
  }
  get completion() { return this.calls.filter(c => c.method === 'PATCH').at(-1)?.body; }
}

test('reconciler publishes the required check on actual PR HEAD after two authoritative reads', async () => {
  const api = new FakeAPI();
  const result = await reconcile(api, policy, 42, api.openPRs, 'https://example.test/run');
  assert.equal(result.pass, true);
  const start = api.calls.find(c => c.method === 'POST' && c.path.endsWith('/check-runs'));
  assert.equal(start.body.name, CHECK);
  assert.equal(start.body.head_sha, head);
  assert.equal(api.prReads, 3);
  assert.equal(api.completion.conclusion, 'success');
});

test('library calls cannot append fixture PRs to an inherited workflow summary', async t => {
  const directory = mkdtempSync(join(tmpdir(), 'pccl-review-summary-'));
  const path = join(directory, 'summary.md');
  const previous = process.env.GITHUB_STEP_SUMMARY;
  t.after(() => {
    if (previous === undefined) delete process.env.GITHUB_STEP_SUMMARY;
    else process.env.GITHUB_STEP_SUMMARY = previous;
    rmSync(directory, {recursive: true, force: true});
  });
  writeFileSync(path, 'Real workflow summary\n');
  process.env.GITHUB_STEP_SUMMARY = path;
  const output = await run(new FakeAPI(), policy, 'workflow_dispatch', {inputs: {pr_number: '42'}});
  assert.match(output, /PR #42/);
  assert.equal(readFileSync(path, 'utf8'), 'Real workflow summary\n');
});

test('successful event handling does not turn missing owner approval into a passing check', async () => {
  const api = new FakeAPI();
  api.reviews = [];
  await run(api, policy, 'pull_request_target', {pull_request: {number: 42}});
  const start = api.calls.find(c => c.method === 'POST' && c.path.endsWith('/check-runs'));
  assert.equal(start.body.name, CHECK);
  assert.equal(start.body.head_sha, head);
  assert.equal(api.completion.conclusion, 'failure');
});

test('one cross-module owner approval passes while other affected modules still receive review requests', async () => {
  const api = new FakeAPI();
  api.pr.body = 'PCCL-Modules: device, ras';
  api.files = [{filename: 'src/include/pccl_device/ptr.h'}, {filename: 'src/misc/comm_dumper.cc'}];
  api.pr.changed_files = api.files.length;
  api.reviews[0].user.login = 'wensuyang';
  const result = await reconcile(api, policy, 42, api.openPRs);
  assert.equal(result.pass, true);
  assert.equal(api.completion.conclusion, 'success');
  assert.deepEqual(result.modules.find(mod => mod.id === 'ras').approvedBy, []);
  assert.equal(api.pr.requested_reviewers.some(user => user.login === 'ck-milo'), true);
  assert.match(api.completion.output.summary, /Owner approval received from: wensuyang/);
  assert.doesNotMatch(api.completion.output.summary, /awaiting|BLOCKED/i);
});

test('administrator status alone cannot satisfy module owner approval', async () => {
  const api = new FakeAPI();
  api.permission = 'admin';
  api.reviews[0].user.login = 'non-owner-admin';
  const result = await reconcile(api, policy, 42, api.openPRs);
  assert.equal(result.pass, false);
  assert.equal(api.completion.conclusion, 'failure');
});

test('workflow entry point authenticates with GITHUB_TOKEN and rejects a missing token', t => {
  const directory = mkdtempSync(join(tmpdir(), 'pccl-review-token-'));
  t.after(() => rmSync(directory, {recursive: true, force: true}));
  const eventPath = join(directory, 'event.json');
  writeFileSync(eventPath, '{}');
  const runner = fileURLToPath(new URL('../../.github/review/run.mjs', import.meta.url));
  const script = `
    import assert from 'node:assert/strict';
    import {pathToFileURL} from 'node:url';
    let requests = 0;
    globalThis.fetch = async (url, options) => {
      requests++;
      assert.equal(options.headers.Authorization, 'Bearer workflow-test-token');
      assert.equal(options.method, 'GET');
      assert.equal(url, 'https://api.github.com/repos/t-head/pccl/pulls?state=open&per_page=100&page=1');
      return {ok: true, status: 200, json: async () => []};
    };
    process.argv[1] = ${JSON.stringify(runner)};
    await import(pathToFileURL(process.argv[1]).href);
    assert.equal(requests, process.env.GITHUB_TOKEN ? 1 : 0);
  `;
  const env = {GITHUB_REPOSITORY: 't-head/pccl', GITHUB_EVENT_NAME: 'schedule', GITHUB_EVENT_PATH: eventPath};
  const withToken = spawnSync(process.execPath, ['--input-type=module', '--eval', script], {
    encoding: 'utf8', env: {...env, GITHUB_TOKEN: 'workflow-test-token'},
  });
  assert.equal(withToken.status, 0, withToken.stderr);
  const withoutToken = spawnSync(process.execPath, ['--input-type=module', '--eval', script], {
    encoding: 'utf8', env,
  });
  assert.equal(withoutToken.status, 1);
  assert.match(withoutToken.stderr, /GITHUB_TOKEN is required/);
});

test('new push during evaluation never approves old or new HEAD', async () => {
  const api = new FakeAPI();
  api.onPRRead = () => { if (api.prReads === 3) api.pr.head.sha = 'c'.repeat(40); };
  await assert.rejects(() => reconcile(api, policy, 42, api.openPRs), /changed during/);
  assert.equal(api.completion.conclusion, 'failure');
});

test('dismissal, module edit, or draft transition during evaluation invalidates result', async () => {
  const mutations = [a => {a.reviews[0].state = 'DISMISSED';},
    a => {a.pr.body += ', network';}, a => {a.pr.draft = true;}];
  for (const mutate of mutations) {
    const api = new FakeAPI();
    api.onPRRead = () => { if (api.prReads === 3) mutate(api); };
    await assert.rejects(() => reconcile(api, policy, 42, api.openPRs), /changed during/);
    assert.equal(api.completion.conclusion, 'failure');
  }
});

test('incomplete/truncated file list and GitHub errors fail closed', async () => {
  for (const setup of [a => {a.pr.changed_files = 3001;}, a => {a.files = [];}, a => {a.reviewError = true;},
    a => {a.permissionError = Object.assign(new Error('Rate limited'), {status: 403});}]) {
    const api = new FakeAPI(); setup(api);
    await assert.rejects(() => reconcile(api, policy, 42, api.openPRs));
    assert.equal(api.completion.conclusion, 'failure');
  }
});

test('non-collaborator permissions do not confer approval', async () => {
  const api = new FakeAPI();
  api.permissionError = Object.assign(new Error('Not found'), {status: 404});
  const result = await reconcile(api, policy, 42, api.openPRs);
  assert.equal(result.pass, false);
  assert.equal(api.completion.conclusion, 'failure');
});

test('duplicate open PRs sharing SHA cannot reuse a passing module result', async () => {
  const api = new FakeAPI();
  api.openPRs.push({...api.pr, number: 43});
  await assert.rejects(() => reconcile(api, policy, 42, api.openPRs), /Multiple open PRs/);
  assert.equal(api.completion.conclusion, 'failure');
});

test('request-reviewer API failure cannot invent approval, and is visible in output', async () => {
  const api = new FakeAPI(); api.requestError = true; api.reviews = [];
  const result = await reconcile(api, policy, 42, api.openPRs);
  assert.equal(result.pass, false);
  assert.match(api.completion.output.summary, /Could not request/);
});

test('closed PR and unsupported target cause no writes', async () => {
  for (const setup of [a => {a.pr.state = 'closed';}, a => {a.pr.base.ref = 'ppu1.7';}]) {
    const api = new FakeAPI(); setup(api);
    await reconcile(api, policy, 42, api.openPRs);
    assert.equal(api.calls.some(c => ['POST', 'PATCH'].includes(c.method)), false);
  }
});

test('review relay uses API facts even if fork associations are empty or SHA is a merge ref', async () => {
  const api = new FakeAPI(); api.relay.head_sha = 'merge-ref-sha';
  await run(api, policy, 'workflow_run', {workflow_run: {id: 123, pull_requests: [{number: 999}]}});
  assert.equal(api.completion.conclusion, 'success');
  assert.equal(api.calls.some(c => /artifact|cache/.test(c.path)), false);
});

test('unexpected relay workflow or event cannot invoke the trusted evaluator', async () => {
  for (const setup of [a => {a.workflowPath = 'attacker.yml';}, a => {a.relay.event = 'push';}]) {
    const api = new FakeAPI(); setup(api);
    await assert.rejects(() => run(api, policy, 'workflow_run', {workflow_run: {id: 123}}),
      /Unexpected review-event/);
    assert.equal(api.calls.some(c => c.method === 'POST'), false);
  }
});

test('manual refresh validates input; push and periodic recovery refresh open PRs', async () => {
  const invalid = {inputs: {pr_number: '../1'}};
  await assert.rejects(() => run(new FakeAPI(), policy, 'workflow_dispatch', invalid), /Invalid PR/);
  for (const event of ['push', 'schedule']) {
    const api = new FakeAPI(); await run(api, policy, event, {});
    assert.equal(api.completion.conclusion, 'success');
  }
});

test('REST pagination fetches all review pages and rejects non-array responses', async () => {
  const api = new GitHub('test', 't-head/pccl');
  const pages = [];
  api.request = async (method, path) => {
    pages.push(path);
    return path.endsWith('page=1') ? Array.from({length: 100}, (_, id) => ({id})) : [{id: 100}];
  };
  const results = await api.list('/pulls/42/reviews');
  assert.equal(results.length, 101);
  assert.equal(pages.length, 2);
  api.request = async () => ({message: 'not an array'});
  await assert.rejects(() => api.list('/pulls/42/reviews'), /Expected a paginated array/);
});
