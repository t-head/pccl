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


def allocation_cards_per_node(case, platform=None, platforms=None):
    """Derive the scheduler allocation without changing the Case definition."""
    spec = case["suite_spec"]
    platform = platform or case_platform(case, platforms)
    if platform != "oam810e_eic":
        return spec["cards_per_node"]
    command = case["cmd"]
    nodes = spec["nodes"]

    def option(flag):
        match = re.search(re.escape(flag) + r"(?:=|\s+)([0-9]+)", command)
        return int(match.group(1)) if match else None

    def per_node(total):
        if total <= 0 or total % nodes:
            raise ValueError(f"Ranks do not divide evenly across nodes: {case['plan']}/{case['suite']}/{case['name']}")
        return total // nodes

    def positive(value):
        if value <= 0:
            raise ValueError(f"Invalid 810E allocation ranks: {case['plan']}/{case['suite']}/{case['name']}")
        return value

    def power_of_two(value):
        if value <= 0:
            raise ValueError(f"Invalid 810E allocation ranks: {case['plan']}/{case['suite']}/{case['name']}")
        return 1 << (value - 1).bit_length()

    rank_cards = None
    if (value := option("-np")) is not None:
        if "pcclCommAbortTest" in command:
            value *= (option("-t") or 1) * (option("-g") or 1)
        rank_cards = per_node(value)
    elif (value := option("--ranks")) is not None:
        rank_cards = per_node(value)
    elif (threads := option("--num_threads")) is not None:
        rank_cards = per_node(threads * (option("--num_gpus") or 1))
    elif (value := option("--num_gpus")) is not None:
        rank_cards = per_node(value)
    elif (threads := option("-t")) is not None and (gpus := option("-g")) is not None:
        rank_cards = per_node(threads * gpus)
    elif (value := option("--nprocs")) is not None:
        rank_cards = positive(value)
    elif (value := option("--nproc")) is not None:
        rank_cards = positive(value)
    else:
        match = re.search(r"(?:^|_)ranks_([0-9]+)(?:_|$)", case["name"])
        if match:
            rank_cards = per_node(int(match.group(1)))

    device_spans = []
    for pattern in run_cases.DEVICE_SELECTION_PATTERNS:
        for match in re.finditer(pattern, command):
            device_spans.append(max(int(item) for item in match.group(3).split(",")) + 1)
    if device_spans:
        allocation = max(rank_cards or 0, max(device_spans))
    elif rank_cards is not None:
        allocation = rank_cards
    elif "graphSearchTest" in command:
        # graphSearchTest builds a graph from the topology XML given by --root and names PPUs in
        # its gtest filter, never in a rank switch, so the Suite's declared allocation is the
        # only card count it has.
        return power_of_two(spec["cards_per_node"])
    else:
        raise ValueError(f"Cannot derive 810E allocation ranks: {case['plan']}/{case['suite']}/{case['name']}")
    return power_of_two(allocation)


def _rename(value, rules):
    for old, new in rules:
        value = value.replace(old, new)
    return value


def _without_fp8(rows):
    return [row for row in rows if "f8e4m3" not in row["cmd"] and "f8e5m2" not in row["cmd"]]


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
        "np": r"((?:^|_)np)([0-9]+)",
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
    cards_per_node = rule.get("cards_per_node", profile["cards_per_node"])
    if type(cards_per_node) is not int or cards_per_node <= 0:
        raise ValueError(f"Platform {profile['name']} has an invalid card count for {family}")
    suite_map = {}
    suites = []
    for source in bundle["suites"]:
        if source["nodes"] not in rule.get("nodes", profile["allowed_nodes"]):
            continue
        row = deepcopy(source)
        row["plan"] = _rename(row["plan"], renames)
        row["suite"] = _rename(row["suite"], renames)
        row["cards_per_node"] = cards_per_node
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
    rows = _without_fp8(rows)
    if family == "device_api_shm":
        rows = [row for row in rows if "GIN_TYPE=" not in row["cmd"]]
    if not rows:
        raise ValueError(f"Platform {profile['name']} selected no Cases for {family}")
    selected_suites = {(row["plan"], row["suite"]) for row in rows}
    suites = [row for row in suites if (row["plan"], row["suite"]) in selected_suites]
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
                        source = next((candidate for candidate in variant_bundles
                                       if candidate["generation"]["family"] == family), None)
                        if source is None:
                            raise ValueError(f"Missing variant bundle for family {family}")
                    converted.append(_filter_variant(
                        apply_platform(source, profile, target), bundle["generation"]["profile"]))
    return converted
