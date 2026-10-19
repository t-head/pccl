// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validatePolicy, classify, evaluate, reviewerRequests, declaration} from '../../.github/review/policy.mjs';

const config = new URL('../../.github/review/owners.json', import.meta.url);
const policy = validatePolicy(JSON.parse(readFileSync(config)));
const head = 'a'.repeat(40);
const pr = {number: 42, state: 'open', draft: false, body: 'PCCL-Modules: kernels',
  user: {login: 'contributor'}, head: {sha: head}, base: {ref: 'main', sha: 'b'.repeat(40)}, requested_reviewers: []};
const files = [{filename: 'src/collectives/device/all_reduce.cu'}];
const scope = classify(policy, files, pr.body);
const crossFiles = [{filename: 'src/include/pccl_device/ptr.h'}, {filename: 'src/misc/comm_dumper.cc'}];
const crossScope = classify(policy, crossFiles, 'PCCL-Modules: device, ras');
const permissions = Object.fromEntries(Object.values(policy.people)
  .filter(p => p.github).map(p => [p.github.toLowerCase(), 'write']));
const review = (login, state = 'APPROVED', commit = head, id = 1) => ({
  id, user: {login, type: 'User'}, state, commit_id: commit, submitted_at: '2026-10-17T00:00:00Z',
});

test('roster has ten modules, 23 areas, and confirmed backup assignments', () => {
  assert.equal(Object.keys(policy.modules).length, 10);
  assert.equal(Object.values(policy.modules).flatMap(m => m.areas).length, 23);
  assert.equal(policy.modules.workflow.maintainer, 'xianghan');
  assert.equal(policy.modules.workflow.backup, 'suyang');
  assert.equal(policy.modules.build.maintainer, 'weilai');
  assert.equal(policy.modules.build.backup, 'chenkai');
  assert.equal(policy.modules.device.backup, 'zhouwang');
  assert.equal(policy.modules.kernels.backup, 'yuanchao');
  assert.equal(policy.people.yinxin.name, '寅鑫');
  assert.equal(policy.people.zian.github, null);
});

test('bad approver identity or path/area mismatch is rejected', () => {
  const p = structuredClone(policy);
  p.people[p.modules.kernels.maintainer].github = null;
  assert.throws(() => validatePolicy(p), /Missing kernels maintainer|Missing .* backup/);
  const q = structuredClone(policy);
  q.pathRules[0].areas = ['network-rdma'];
  assert.throws(() => validatePolicy(q), /outside rule modules/);
});

test('declarations can supplement but cannot remove the path-derived module', () => {
  const s = classify(policy, files, 'PCCL-Modules: build', ['module/network']);
  assert.deepEqual(s.modules, ['build', 'kernels', 'network']);
});

test('renames retain ownership from both original and new path', () => {
  const renamed = [{filename: 'docs/new.md', previous_filename: files[0].filename}];
  assert.deepEqual(classify(policy, renamed, 'PCCL-Modules: build').modules, ['build', 'kernels']);
});

test('specific graph tuning rule precedes topology fallback', () => {
  assert.deepEqual(classify(policy, [{filename: 'src/graph/tuning.cc'}]).modules, ['tuning']);
  assert.deepEqual(classify(policy, [{filename: 'src/graph/search.cc'}]).modules, ['topology']);
});

test('contribution and review automation route to workflow independently of build CI', () => {
  const paths = ['AGENTS.md', 'CONTRIBUTING.md', '.github/pull_request_template.md',
    '.github/copilot-instructions.md', '.github/ISSUE_TEMPLATE/BUG.yaml',
    '.github/review/owners.json', '.github/review/run.mjs',
    '.github/workflows/module-review.yml', '.github/workflows/module-review-events.yml',
    '.github/workflows/module-review-tests.yml', '.agents/review_rules.yaml',
    '.agents/skills/pccl-contribution/SKILL.md', 'tests/review/policy.test.mjs',
    'docs/workflow/module-review.zh-CN.md'];
  for (const filename of paths) {
    const result = classify(policy, [{filename}]);
    assert.deepEqual(result.modules, ['workflow'], filename);
    assert.deepEqual(result.areas, ['workflow-review'], filename);
  }
});

test('build scripts and hardware CI keep their own routing', () => {
  for (const filename of ['.github/workflows/pr-test-ppu.yml', '.github/workflows/scripts/build.sh',
    '.github/workflows/scripts/pr-ut-runner.py', '.aoneci/test.yml']) {
    const result = classify(policy, [{filename}]);
    assert.deepEqual(result.modules, ['build'], filename);
    assert.deepEqual(result.areas, ['build-2'], filename);
  }
  const result = classify(policy, [{filename: 'Makefile'}]);
  assert.deepEqual(result.modules, ['build']);
  assert.deepEqual(result.areas, ['build-1']);
});

test('workflow approval and backup routing use its dedicated owners', () => {
  const s = classify(policy, [{filename: 'CONTRIBUTING.md'}], 'PCCL-Modules: workflow');
  for (const login of ['honganlxh', 'wensuyang']) {
    assert.equal(evaluate(policy, pr, s, [review(login)], permissions).pass, true, login);
  }
  for (const login of ['levilee-lwl', 'ck-milo']) {
    assert.equal(evaluate(policy, pr, s, [review(login)], permissions).pass, false, login);
  }
  assert.deepEqual(reviewerRequests(policy, pr, s, [], permissions), ['honganlxh']);
  assert.deepEqual(reviewerRequests(policy, pr, {...s, backups: ['workflow']}, [], permissions), ['wensuyang']);
  const own = {...pr, user: {login: 'honganlxh'}};
  assert.deepEqual(reviewerRequests(policy, own, s, [], permissions), ['wensuyang']);
  assert.equal(evaluate(policy, own, s, [review('honganlxh')], permissions).pass, false);
  assert.equal(evaluate(policy, own, s, [review('wensuyang')], permissions).pass, true);
});

test('GIN backend is classified under both device and network; Device API stays within device', () => {
  assert.deepEqual(classify(policy, [{filename: 'src/transport/net_ib/gin.cc'}]).modules, ['device', 'network']);
  assert.deepEqual(classify(policy, [{filename: 'src/include/pccl_device/ptr.h'}]).modules, ['device']);
});

test('unknown paths require triage ownership and remain visible for manual scope review', () => {
  const s = classify(policy, [{filename: 'new-platform/kernel.cu'}], 'PCCL-Modules: kernels');
  assert.deepEqual(s.modules, ['build', 'kernels']);
  assert.deepEqual(s.unmatched, ['new-platform/kernel.cu']);
});

test('area selection adds its module; SNIC can be declared even without a dedicated directory', () => {
  const s = classify(policy, files, 'PCCL-Modules: kernels\nPCCL-Areas: network-snic');
  assert.deepEqual(s.modules, ['kernels', 'network']);
});

test('unknown module/area/backup labels fail, duplicate declarations fail, commented examples do not count', () => {
  for (const label of ['module/typo', 'area/typo', 'backup/typo']) {
    assert.throws(() => classify(policy, files, '', [label]), /Unknown/);
  }
  assert.throws(() => declaration('PCCL-Modules: build\nPCCL-Modules: kernels', 'PCCL-Modules'), /exactly one/);
  assert.deepEqual(declaration('<!-- PCCL-Modules: build -->', 'PCCL-Modules'), []);
});

test('both Maintainer and Backup can approve without a backup label', () => {
  assert.equal(evaluate(policy, pr, scope, [review('ck-milo')], permissions).pass, true);
  assert.equal(evaluate(policy, pr, scope, [review('XIANGZHIXYC')], permissions).pass, true);
});

test('a specialist approval does not authorize its entire module', () => {
  assert.equal(evaluate(policy, pr, scope, [review('wensuyang'), review('qinghanke')], permissions).pass, false);
});

test('author cannot approve own PR; Backup can approve a Maintainer-authored PR', () => {
  const own = {...pr, user: {login: 'ck-milo'}};
  assert.equal(evaluate(policy, own, scope, [review('ck-milo')], permissions).pass, false);
  assert.equal(evaluate(policy, own, scope, [review('xiangzhixyc')], permissions).pass, true);
});

test('approval is bound to current HEAD and current collaborator write access', () => {
  assert.equal(evaluate(policy, pr, scope, [review('ck-milo', 'APPROVED', 'old')], permissions).pass, false);
  assert.equal(evaluate(policy, pr, scope, [review('ck-milo')], {...permissions, 'ck-milo': 'read'}).pass, false);
});

test('any one affected-module Maintainer or Backup can approve a cross-module PR', () => {
  assert.deepEqual(crossScope.modules, ['device', 'ras']);
  assert.equal(evaluate(policy, pr, crossScope, [], permissions).pass, false);
  for (const owner of ['wensuyang', 'snowdrift-zw', 'ck-milo', 'QingxunLy']) {
    const result = evaluate(policy, pr, crossScope, [review(owner)], permissions);
    assert.equal(result.pass, true, owner);
    assert.deepEqual(result.approvedBy, [owner.toLowerCase()]);
    assert.equal(result.modules.filter(mod => mod.approvedBy.length).length, 1);
  }
});

test('owners of unaffected modules cannot approve a cross-module PR', () => {
  const reviews = ['levilee-lwl', 'whale494255', 'honganlxh'].map(login => review(login));
  assert.equal(evaluate(policy, pr, crossScope, reviews, permissions).pass, false);
});

test('scope changes retain an approval only while the approver owns an affected module', () => {
  const approved = [review('wensuyang')];
  const device = classify(policy, [crossFiles[0]], 'PCCL-Modules: device');
  const ras = classify(policy, [crossFiles[1]], 'PCCL-Modules: ras');
  assert.equal(evaluate(policy, pr, device, approved, permissions).pass, true);
  assert.equal(evaluate(policy, pr, crossScope, approved, permissions).pass, true);
  assert.equal(evaluate(policy, pr, ras, approved, permissions).pass, false);
});

test('cross-module approval survives a dismissal only when another eligible approval remains', () => {
  const dismissed = [review('wensuyang'), review('wensuyang', 'DISMISSED', head, 2)];
  assert.equal(evaluate(policy, pr, crossScope, dismissed, permissions).pass, false);
  assert.equal(evaluate(policy, pr, crossScope,
    [...dismissed, review('ck-milo', 'APPROVED', 'old', 3)], permissions).pass, false);
  assert.equal(evaluate(policy, pr, crossScope,
    [...dismissed, review('ck-milo', 'APPROVED', head, 3)], permissions).pass, true);
  assert.equal(evaluate(policy, pr, crossScope,
    [review('wensuyang', 'CHANGES_REQUESTED'), review('ck-milo', 'APPROVED', head, 2)], permissions).pass, false);
});

test('comments do not revoke approval; dismissal and change requests do', () => {
  const approved = review('ck-milo');
  const commented = [approved, review('ck-milo', 'COMMENTED', head, 2)];
  assert.equal(evaluate(policy, pr, scope, commented, permissions).pass, true);
  for (const state of ['DISMISSED', 'CHANGES_REQUESTED']) {
    assert.equal(evaluate(policy, pr, scope, [approved, review('ck-milo', state, head, 2)], permissions).pass, false);
  }
});

test('writer change requests block even when a different owner approved', () => {
  const writer = [review('ck-milo'), review('qinghanke', 'CHANGES_REQUESTED', 'old', 2)];
  const outsider = [review('ck-milo'), review('outsider', 'CHANGES_REQUESTED', head, 2)];
  assert.equal(evaluate(policy, pr, scope, writer, permissions).pass, false);
  assert.equal(evaluate(policy, pr, scope, outsider, permissions).pass, true);
});

test('bots and unsubmitted reviews cannot approve', () => {
  const bot = {...review('ck-milo'), user: {login: 'ck-milo', type: 'Bot'}};
  for (const r of [bot, {...review('ck-milo'), submitted_at: null}]) {
    assert.equal(evaluate(policy, pr, scope, [r], permissions).pass, false);
  }
});

test('draft, wrong target, missing declaration, or no files fails closed', () => {
  assert.equal(evaluate(policy, {...pr, draft: true}, scope, [review('ck-milo')], permissions).pass, false);
  assert.equal(evaluate(policy, {...pr, base: {ref: 'ppu1.7'}}, scope, [review('ck-milo')], permissions).pass, false);
  assert.equal(evaluate(policy, pr, classify(policy, files), [review('ck-milo')], permissions).pass, false);
  assert.equal(evaluate(policy, pr, classify(policy, [], pr.body), [review('ck-milo')], permissions).pass, false);
});

test('routing requests primary, or backup for author/absence, and deduplicates specialists', () => {
  const normal = reviewerRequests(policy, pr, scope, [], permissions);
  assert.equal(normal.includes('ck-milo'), true);
  assert.equal(normal.includes('xiangzhixyc'), false);
  assert.equal(normal.includes('qinghanke'), true);
  const away = reviewerRequests(policy, pr, {...scope, backups: ['kernels']}, [], permissions);
  assert.equal(away.includes('xiangzhixyc'), true);
  assert.equal(away.includes('ck-milo'), false);
  const own = reviewerRequests(policy, {...pr, user: {login: 'ck-milo'}}, scope, [], permissions);
  assert.equal(own.includes('xiangzhixyc'), true);
  assert.equal(own.includes('ck-milo'), false);
});

test('approved module does not request another owner; pending requests remain idempotent', () => {
  const requests = reviewerRequests(policy, {...pr, requested_reviewers: [{login: 'qinghanke'}, {login: 'honganlxh'}]},
    scope, [review('xiangzhixyc')], permissions);
  assert.deepEqual(requests, []);
});

test('unresolved EIC specialist is skipped; reviewers never inherit Maintainer authority', () => {
  const s = classify(policy, [{filename: 'src/include/u2mm.h'}], 'PCCL-Modules: network');
  assert.deepEqual(reviewerRequests(policy, pr, s, [], permissions).sort(), ['ctt284373', 'xiangzhixyc']);
});
