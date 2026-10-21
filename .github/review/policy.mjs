// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// Pure policy functions. No network, checkout, or PR-supplied code execution.
export function validatePolicy(policy) {
  if (policy.version !== 1 || !policy.targetBranch || !Array.isArray(policy.pathRules)) {
    throw new Error('Unsupported or incomplete review policy');
  }
  const people = policy.people ?? {};
  const logins = new Set();
  for (const person of Object.values(people)) {
    if (person.github === null) continue; // Unconfirmed specialist; never an approver.
    if (!/^[a-z\d](?:[a-z\d-]{0,37}[a-z\d])?$/i.test(person.github ?? '')) {
      throw new Error('Invalid GitHub login');
    }
    const login = person.github.toLowerCase();
    if (logins.has(login)) throw new Error(`Duplicate identity: ${login}`);
    logins.add(login);
  }
  const areas = new Set();
  const areaModules = new Map();
  for (const [id, mod] of Object.entries(policy.modules ?? {})) {
    if (!/^[a-z][a-z0-9-]*$/.test(id)) throw new Error(`Invalid module: ${id}`);
    for (const role of ['maintainer', 'backup']) {
      if (!people[mod[role]]?.github) throw new Error(`Missing ${id} ${role} account`);
    }
    if (mod.maintainer === mod.backup) throw new Error(`Identical primary/backup: ${id}`);
    for (const area of mod.areas) {
      if (!/^[a-z][a-z0-9-]*$/.test(area.id)) throw new Error(`Invalid area: ${area.id}`);
      if (areas.has(area.id)) throw new Error(`Duplicate area: ${area.id}`);
      areas.add(area.id);
      areaModules.set(area.id, id);
      for (const person of area.reviewers) {
        if (!people[person]) throw new Error(`Unknown reviewer: ${person}`);
      }
    }
  }
  if (!Object.keys(policy.modules ?? {}).length) throw new Error('No modules');
  if (!policy.unmappedModules?.length || policy.unmappedModules.some(id => !policy.modules[id])) {
    throw new Error('Unmapped paths need a valid triage module');
  }
  for (const rule of policy.pathRules) {
    if (!(rule.files?.length || rule.prefixes?.length)) throw new Error('Empty path rule');
    if (!rule.modules?.length) throw new Error('Path rule has no modules');
    for (const mod of rule.modules) {
      if (!policy.modules[mod]) throw new Error(`Unknown module in path rule: ${mod}`);
    }
    for (const area of rule.areas ?? []) {
      if (!areas.has(area)) throw new Error(`Unknown area in path rule: ${area}`);
      if (!rule.modules.includes(areaModules.get(area))) throw new Error(`Area outside rule modules: ${area}`);
    }
  }
  return policy;
}

export function declaration(body, key) {
  const visible = (body ?? '').replace(/<!--[\s\S]*?-->/g, '');
  const matches = [...visible.matchAll(new RegExp(`^${key}:[ \\t]*(.*)$`, 'gm'))];
  if (matches.length > 1) throw new Error(`Use exactly one ${key} line`);
  return matches.length ? matches[0][1].split(',').map(s => s.trim()).filter(Boolean) : [];
}

export function classify(policy, files, body = '', labels = []) {
  const modules = new Set(declaration(body, 'PCCL-Modules'));
  const areas = new Set(declaration(body, 'PCCL-Areas'));
  const backups = new Set();
  const areaModules = new Map(Object.entries(policy.modules)
    .flatMap(([id, mod]) => mod.areas.map(area => [area.id, id])));
  for (const label of labels) {
    if (label.startsWith('module/')) modules.add(label.slice(7));
    if (label.startsWith('area/')) areas.add(label.slice(5));
    if (label.startsWith('backup/')) backups.add(label.slice(7));
  }
  const declared = [...modules];
  for (const id of [...modules, ...backups]) {
    if (!policy.modules[id]) throw new Error(`Unknown module: ${id}`);
  }
  for (const id of areas) {
    if (!areaModules.has(id)) throw new Error(`Unknown area: ${id}`);
    modules.add(areaModules.get(id));
  }
  const unmatched = [];
  const paths = new Set(files.flatMap(file => [file.filename, file.previous_filename].filter(Boolean)));
  for (const path of paths) {
    const rule = policy.pathRules.find(r => r.files?.includes(path) || r.prefixes?.some(p => path.startsWith(p)));
    if (!rule) {
      unmatched.push(path);
      policy.unmappedModules.forEach(id => modules.add(id));
    }
    else {
      rule.modules.forEach(id => modules.add(id));
      (rule.areas ?? []).forEach(id => areas.add(id));
    }
  }
  return {modules: [...modules].sort(), areas: [...areas].sort(), backups: [...backups].sort(),
    declared, unmatched, fileCount: files.length};
}

export function latestDecisions(reviews) {
  const latest = new Map();
  for (const review of [...reviews].sort((a, b) => a.id - b.id)) {
    if (review.user?.type !== 'User' || !review.submitted_at) continue;
    // A comment does not revoke an earlier approval or request for changes.
    if (['APPROVED', 'CHANGES_REQUESTED', 'DISMISSED'].includes(review.state)) {
      latest.set(review.user.login.toLowerCase(), review);
    }
  }
  return latest;
}

export function ownerLogins(policy, module) {
  const mod = policy.modules[module];
  return [mod.maintainer, mod.backup].map(id => policy.people[id].github.toLowerCase());
}

export function evaluate(policy, pr, scope, reviews, permissions = {}) {
  const errors = [];
  const author = pr.user.login.toLowerCase();
  const decisions = latestDecisions(reviews);
  const canApprove = login => ['write', 'maintain', 'admin'].includes(permissions[login]);
  if (pr.state !== 'open' || pr.draft) errors.push('PR must be open and ready for review');
  if (pr.base.ref !== policy.targetBranch) errors.push(`Only ${policy.targetBranch} is supported`);
  if (!scope.declared.length && !scope.recommendation) {
    errors.push('Automatic classification is unavailable; declare PCCL-Modules in the PR body or apply module/ labels');
  }
  if (!scope.modules.length || !scope.fileCount) errors.push('No affected modules/files');
  const modules = scope.modules.map(id => {
    const owners = ownerLogins(policy, id);
    const approvedBy = owners.filter(login => {
      const r = decisions.get(login);
      return login !== author && canApprove(login) && r?.state === 'APPROVED' && r.commit_id === pr.head.sha;
    });
    return {id, owners, approvedBy};
  });
  const approvedBy = [...new Set(modules.flatMap(mod => mod.approvedBy))];
  if (modules.length && !approvedBy.length) {
    const owners = [...new Set(modules.flatMap(mod => mod.owners))];
    errors.push(`Needs one current-head approval from an affected module's Maintainer or Backup: ${owners.join(', ')}`);
  }
  // Native review protection still applies, including changes requested by other writers.
  for (const [login, review] of decisions) {
    if (review.state === 'CHANGES_REQUESTED' && canApprove(login) && login !== author) {
      errors.push(`Changes requested by ${login}; resolve or dismiss the review`);
    }
  }
  return {pass: !errors.length, errors, modules, approvedBy};
}

export function reviewerRequests(policy, pr, scope, reviews, permissions = {}) {
  const author = pr.user.login.toLowerCase();
  const pending = new Set((pr.requested_reviewers ?? []).map(r => r.login.toLowerCase()));
  const decisions = latestDecisions(reviews);
  const requested = new Set();
  for (const id of scope.modules) {
    const mod = policy.modules[id];
    const primary = policy.people[mod.maintainer].github.toLowerCase();
    const backup = policy.people[mod.backup].github.toLowerCase();
    const covered = [primary, backup].some(login => login !== author &&
      ['write', 'maintain', 'admin'].includes(permissions[login]) &&
      decisions.get(login)?.state === 'APPROVED' && decisions.get(login)?.commit_id === pr.head.sha);
    if (!covered) requested.add(primary === author || scope.backups.includes(id) ? backup : primary);
    for (const area of mod.areas.filter(a => scope.areas.includes(a.id))) {
      for (const person of area.reviewers) {
        const login = policy.people[person].github;
        // Module owners are requested by the primary/backup rule, not again as specialists.
        if (login && ![primary, backup].includes(login.toLowerCase())) requested.add(login.toLowerCase());
      }
    }
  }
  return [...requested].filter(login => login !== author && !pending.has(login) &&
    !(decisions.get(login)?.state === 'APPROVED' && decisions.get(login)?.commit_id === pr.head.sha));
}
