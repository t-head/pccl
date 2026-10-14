// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {selectedModules, route} = require('./module-review.cjs');
const body = '## Affected Modules\n- [x] Env Check\n- [x] Dispatch Strategy\n';

test('parse modules only in the real section', () => {
  assert.deepEqual(selectedModules(body), ['Env Check', 'Dispatch Strategy']);
  assert.deepEqual(selectedModules('<!--\n'+body+'-->\n```markdown\n'+body+'```\n'+body), ['Env Check', 'Dispatch Strategy']);
  assert.throws(() => selectedModules('## Affected Modules\n- [ ] Env Check\n## Tests\n- [x] Env Check'), /at least one/);
  assert.throws(() => selectedModules(body+body), /exactly one/);
  assert.throws(() => selectedModules('## Affected Modules\n- [x] Typo'), /Unknown/);
  assert.throws(() => selectedModules('```\n'+body+'```'), /exactly one/);
  assert.throws(() => selectedModules('<!--\n'+body), /exactly one/);
});

function fixture(overrides = {}) {
  const statuses = [], requests = [], failures = [];
  const pr = {state:'open',body,head:{sha:'head'},user:{login:'honganlxh'},draft:false,...overrides};
  const pulls = {
    get: async () => ({data:pr}),
    listRequestedReviewers: async () => ({data:{users:[]}}),
    listReviews: () => {},
    requestReviewers: async args => requests.push(args.reviewers),
  };
  const github = {rest:{pulls,repos:{createCommitStatus:async args => statuses.push(args.state)}},paginate:async () => []};
  const summary = {addHeading(){return this;},addTable(){return this;},addRaw(){return this;},async write(){}};
  return {args:{github,context:{repo:{owner:'t-head',repo:'pccl'},payload:{pull_request:{number:1}},serverUrl:'https://github.com',runId:1},core:{info(){},summary,setFailed:message=>failures.push(message)}},statuses,requests,failures};
}

test('invite mapped reviewer and skip PR author', async () => {
  const f=fixture();await route(f.args);
  assert.deepEqual(f.requests,[['wensuyang']]);assert.deepEqual(f.statuses,['pending','success']);
});
test('both owners can be invited for another author', async () => {
  const f=fixture({user:{login:'someone-else'}});await route(f.args);
  assert.deepEqual(f.requests,[['wensuyang','honganlxh']]);
});
test('do not notify pending or completed reviewers again', async () => {
  const f=fixture({user:{login:'someone-else'}});
  f.args.github.rest.pulls.listRequestedReviewers=async()=>({data:{users:[{login:'Wensuyang'}]}});
  f.args.github.paginate=async()=>[{state:'APPROVED',user:{login:'honganlxh'}}];
  await route(f.args);assert.deepEqual(f.requests,[]);assert.equal(f.statuses.at(-1),'success');
});
test('drafts defer invitations', async () => {
  const f=fixture({draft:true});await route(f.args);assert.deepEqual(f.requests,[]);assert.equal(f.statuses.at(-1),'success');
});
test('missing selections fail without inviting', async () => {
  const f=fixture({body:'## Affected Modules\n- [ ] Env Check'});await route(f.args);
  assert.deepEqual(f.requests,[]);assert.equal(f.statuses.at(-1),'failure');assert.equal(f.failures.length,1);
});
test('API failure fails the status', async () => {
  const f=fixture();f.args.github.rest.pulls.requestReviewers=async()=>{throw new Error('permission denied');};
  await route(f.args);assert.equal(f.statuses.at(-1),'failure');assert.deepEqual(f.failures,['permission denied']);
});
