#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apply reviewed hardware profiles to shared public test-family bundles."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

import run_cases


ROOT = Path(__file__).resolve().parent
PLATFORM_ROOT = ROOT / "platforms"
DEFAULT_PLATFORM = "m890p"


def load_platform(name, root=PLATFORM_ROOT):
    path = Path(root) / f"{name}.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "name", "node_selector", "cards_per_node",
                "allowed_nodes", "plan_prefixes", "families"}
    if (profile.get("schema_version") != 1 or not required <= set(profile)
            or set(profile) - required - {"rename"}
            or profile["name"] != name or not run_cases.SAFE_NAME.fullmatch(name)
            or type(profile["cards_per_node"]) is not int or profile["cards_per_node"] <= 0
            or not profile["allowed_nodes"]
            or any(type(value) is not int or value <= 0 for value in profile["allowed_nodes"])):
        raise ValueError(f"Invalid hardware platform profile: {name}")
    return profile


def load_platforms(root=PLATFORM_ROOT):
    return {path.stem: load_platform(path.stem, root)
            for path in sorted(Path(root).glob("*.json"))}


def case_platform(case, platforms=None):
    platforms = platforms or load_platforms()
    plan = case["plan"] if isinstance(case, dict) else case[0]
    matches = [name for name, profile in platforms.items()
               if any(plan.startswith(prefix) for prefix in profile["plan_prefixes"])]
    if not matches:
        return DEFAULT_PLATFORM
    if len(matches) != 1:
        raise ValueError(f"Case matches multiple hardware platforms: {plan}")
    return matches[0]


def _rename(value, rules):
    for old, new in rules:
        value = value.replace(old, new)
    return value


def _number(field, value, numeric):
    values = numeric.get(field, {})
    return values.get(str(value), value)


def _command_numbers(command, numeric):
    patterns = {
        "ranks": r"(--ranks\s+)([0-9]+)",
        "nproc": r"(--nproc\s+)([0-9]+)",
        "nprocs": r"(--nprocs\s+)([0-9]+)",
        "npernode": r"(-npernode\s+)([0-9]+)",
        "np": r"(-np\s+)([0-9]+)",
    }
    for field, pattern in patterns.items():
        command = re.sub(pattern, lambda match: match.group(1) + str(
            _number(field, match.group(2), numeric)), command)
    return command


def _name_numbers(name, numeric):
    patterns = {
        "ranks": r"(ranks_)([0-9]+)",
        "nproc": r"(nproc_)([0-9]+)",
        "nprocs": r"(process_)([0-9]+)",
        "np": r"(np)([0-9]+)",
    }
    for field, pattern in patterns.items():
        name = re.sub(pattern, lambda match: match.group(1) + str(
            _number(field, match.group(2), numeric)), name)
    return name


def apply_platform(bundle, profile, family=None):
    """Return a new bundle; the shared family input is never mutated."""
    if profile["name"] == DEFAULT_PLATFORM:
        return deepcopy(bundle)
    generation = bundle["generation"]
    family = family or generation["family"]
    rule = profile["families"].get(family)
    if rule is None:
        raise ValueError(f"Platform {profile['name']} does not support family {family}")
    renames = list(profile.get("rename", [])) + list(rule.get("rename", []))
    numeric = rule.get("numeric", {})
    suite_map = {}
    suites = []
    for source in bundle["suites"]:
        if source["nodes"] not in rule.get("nodes", profile["allowed_nodes"]):
            continue
        row = deepcopy(source)
        row["plan"] = _rename(row["plan"], renames)
        row["suite"] = _rename(row["suite"], renames)
        row["cards_per_node"] = profile["cards_per_node"]
        suite_map[(source["plan"], source["suite"])] = (row["plan"], row["suite"])
        suites.append(row)
    rows = []
    for source in bundle["rows"]:
        key = (source["plan"], source["suite"])
        if key not in suite_map:
            continue
        row = deepcopy(source)
        row["plan"], row["suite"] = suite_map[key]
        row["name"] = _name_numbers(_rename(row["name"], renames), numeric)
        row["cmd"] = _command_numbers(row["cmd"], numeric)
        rows.append(row)
    if not rows:
        raise ValueError(f"Platform {profile['name']} selected no Cases for {family}")
    keys = [(row["plan"], row["suite"], row["name"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Platform conversion produced duplicate Cases for {family}")
    ordered_rows = sorted(rows, key=lambda row: (row["plan"], row["suite"], row["name"]))
    case_keys = "".join(f"{row['plan']}/{row['suite']}/{row['name']}\n" for row in ordered_rows)
    generation = {**generation, "family": family, "case_count": len(rows),
                  "platform": profile["name"], "node_selector": profile["node_selector"],
                  "case_keys_sha256": sha256(case_keys.encode()).hexdigest()}
    if "base_family" in rule:
        generation["base_family"] = rule["base_family"]
    return {"metadata": deepcopy(bundle["metadata"]), "suites": suites,
            "rows": ordered_rows,
            "generation": generation, "environments": sorted({row["env"] for row in suites})}


def _filter_variant(bundle, selection_profile):
    if selection_profile == "full":
        return bundle
    rows = [row for row in bundle["rows"] if "all_reduce_perf" in row["name"]]
    keys = {(row["plan"], row["suite"]) for row in rows}
    suites = [row for row in bundle["suites"] if (row["plan"], row["suite"]) in keys]
    case_keys = "".join(f"{row['plan']}/{row['suite']}/{row['name']}\n" for row in rows)
    return {"metadata": bundle["metadata"], "suites": suites, "rows": rows,
            "generation": {**bundle["generation"], "profile": selection_profile,
                           "case_count": len(rows),
                           "case_keys_sha256": sha256(case_keys.encode()).hexdigest()},
            "environments": sorted({row["env"] for row in suites})}


def apply_platforms(bundles, profile, variant_bundles=None):
    converted = []
    for bundle in bundles:
        family = bundle["generation"]["family"]
        converted.append(apply_platform(bundle, profile, family))
        if profile["name"] != DEFAULT_PLATFORM:
            for target, rule in profile["families"].items():
                if rule.get("base_family") == family:
                    source = bundle
                    if variant_bundles is not None:
                        source = next(candidate for candidate in variant_bundles
                                      if candidate["generation"]["family"] == family)
                    converted.append(_filter_variant(
                        apply_platform(source, profile, target), bundle["generation"]["profile"]))
    return converted
