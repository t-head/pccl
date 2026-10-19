// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
import {validatePolicy, classify} from './policy.mjs';

// Local read-only helper. BASE must be a fetched, verified target ref.
const base = process.argv[2];
if (!base || base.startsWith('-')) throw new Error('Usage: node .github/review/classify.mjs BASE_REF [BODY_FILE]');
execFileSync('git', ['rev-parse', '--verify', `${base}^{commit}`]);
const raw = execFileSync('git', ['diff', '--name-status', '-z', '-M', `${base}...HEAD`, '--'], {encoding: 'utf8'});
const fields = raw.split('\0');
const files = [];
for (let i = 0; fields[i];) {
  const status = fields[i++], first = fields[i++];
  if (/^[RC]/.test(status)) files.push({filename: fields[i++], previous_filename: first});
  else files.push({filename: first});
}
const policy = validatePolicy(JSON.parse(readFileSync(new URL('./owners.json', import.meta.url))));
const body = process.argv[3] ? readFileSync(process.argv[3], 'utf8') : '';
const scope = classify(policy, files, body);
console.log(JSON.stringify(scope, null, 2));
console.log(`PCCL-Modules: ${scope.modules.join(', ')}`);
console.log(`PCCL-Areas: ${scope.areas.join(', ')}`);
