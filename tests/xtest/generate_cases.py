#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a temporary public Case CSV bundle from reviewed test-family JSON."""

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import shutil
import tempfile

import run_cases


RUNTIME_FIELDS = ("python_env", "mpi_interface", "path", "ld_library_path", "hostfile",
                  "binary_dir", "test_root")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_definition(path):
    definition = _json(path)
    required = {"schema_version", "family", "defaults", "axes", "layouts", "bindings", "groups", "profiles"}
    if (not required <= set(definition) or set(definition) - required - {"explicit_cases"}
            or definition["schema_version"] != 2):
        raise ValueError("Unsupported public test-list schema")
    if not run_cases.SAFE_NAME.fullmatch(definition["family"]):
        raise ValueError("Invalid test-family name")
    if (not {"check", "case_name"} <= set(definition["defaults"])
            or set(definition["defaults"]) - {"check", "case_name", "command"}
            or definition["defaults"]["check"] not in run_cases.CHECKS):
        raise ValueError("Family defaults require a supported check and case_name")
    for name, values in definition["axes"].items():
        if (not run_cases.SAFE_NAME.fullmatch(name) or not isinstance(values, list) or not values
                or any(isinstance(value, (dict, list)) for value in values)):
            raise ValueError(f"Invalid parameter axis: {name}")
    for nodes, binding in definition["bindings"].items():
        if (not nodes.isdigit() or int(nodes) <= 0 or set(binding) != {"plan", "suite"}
                or any(not run_cases.SAFE_NAME.fullmatch(value) for value in binding.values())):
            raise ValueError("Invalid node/Suite binding")
    for set_name, layouts in definition["layouts"].items():
        if not run_cases.SAFE_NAME.fullmatch(set_name) or not isinstance(layouts, list) or not layouts:
            raise ValueError(f"Invalid layout set: {set_name}")
        names = set()
        for layout in layouts:
            if (set(layout) - {"name", "values", "nodes"} or not {"name", "values"} <= set(layout)
                    or not run_cases.SAFE_NAME.fullmatch(layout["name"])
                    or layout["name"] in names or not isinstance(layout["values"], dict)):
                raise ValueError(f"Invalid or duplicate layout in {set_name}")
            if "nodes" in layout and (type(layout["nodes"]) is not int
                                      or str(layout["nodes"]) not in definition["bindings"]):
                raise ValueError(f"Invalid layout node binding in {set_name}")
            names.add(layout["name"])
    for group in definition["groups"]:
        required_group = {"name", "nodes", "axes", "layout_set"}
        if (not required_group <= set(group)
                or set(group) - required_group - {"select", "case_name", "layouts", "command",
                                                  "command_by_nodes", "check"}):
            raise ValueError("Invalid test-family group")
        if (not run_cases.SAFE_NAME.fullmatch(group["name"])
                or not isinstance(group["nodes"], list) or not group["nodes"]
                or any(type(nodes) is not int or str(nodes) not in definition["bindings"]
                       for nodes in group["nodes"])
                or group["layout_set"] not in definition["layouts"]
                or not group["axes"] or any(axis not in definition["axes"] for axis in group["axes"])
                or group.get("check", definition["defaults"]["check"]) not in run_cases.CHECKS
                or (not group.get("command", definition["defaults"].get("command", "")).strip()
                    and any(str(nodes) not in group.get("command_by_nodes", {}) for nodes in group["nodes"]))
                or any(char in group.get("command", definition["defaults"].get("command", ""))
                       for char in "\0\r\n")):
            raise ValueError(f"Invalid group: {group.get('name')}")
        for axis, values in group.get("select", {}).items():
            if (axis not in definition["axes"] or not isinstance(values, list) or not values
                    or any(value not in definition["axes"][axis] for value in values)):
                raise ValueError(f"Invalid group selection: {group['name']}")
        if ("layouts" in group and
                (not isinstance(group["layouts"], list) or not group["layouts"]
                 or any(name not in {layout["name"] for layout in definition["layouts"][group["layout_set"]]}
                        for name in group["layouts"]))):
            raise ValueError(f"Invalid layout selection: {group['name']}")
        if ("command_by_nodes" in group and
                (not isinstance(group["command_by_nodes"], dict)
                 or any(str(nodes) not in {str(value) for value in group["nodes"]}
                        or not isinstance(command, str) or not command.strip()
                        or any(char in command for char in "\0\r\n")
                        for nodes, command in group["command_by_nodes"].items()))):
            raise ValueError(f"Invalid node command override: {group['name']}")
    for explicit in definition.get("explicit_cases", []):
        if (set(explicit) != {"nodes", "name", "check", "cmd"}
                or type(explicit["nodes"]) is not int
                or str(explicit["nodes"]) not in definition["bindings"]
                or not run_cases.SAFE_NAME.fullmatch(explicit["name"])
                or explicit["check"] not in run_cases.CHECKS
                or not explicit["cmd"].strip() or any(char in explicit["cmd"] for char in "\0\r\n")):
            raise ValueError("Invalid explicit Case")
    if "full" not in definition["profiles"]:
        raise ValueError("Test-family profiles must include full")
    return definition


def _profile(definition, profile, nodes):
    if profile not in definition["profiles"]:
        raise ValueError(f"Unknown generation profile: {profile}")
    selected = definition["profiles"][profile]
    if (not isinstance(selected, dict) or set(selected) - {"nodes", "filters", "include_explicit"}
            or type(selected.get("include_explicit", False)) is not bool):
        raise ValueError("Unsupported generation profile")
    selected_nodes = {int(value) for value in selected.get("nodes", definition["bindings"])}
    if nodes is not None:
        selected_nodes &= {nodes}
    if not selected_nodes:
        raise ValueError("Generation profile selected no node counts")
    filters = selected.get("filters", {})
    if not isinstance(filters, dict) or any(not isinstance(values, list) or not values
                                            for values in filters.values()):
        raise ValueError("Generation profile filters must be nonempty lists")
    layout_keys = {key for layouts in definition["layouts"].values()
                   for layout in layouts for key in layout["values"]}
    unknown = set(filters) - set(definition["axes"]) - layout_keys - {"layout"}
    if unknown:
        raise ValueError(f"Unknown generation filters: {sorted(unknown)}")
    return selected_nodes, filters


def _layouts(group, definition, filters, nodes):
    selected = []
    for layout in definition["layouts"][group["layout_set"]]:
        if "layouts" in group and layout["name"] not in group["layouts"]:
            continue
        if layout.get("nodes", nodes) != nodes:
            continue
        values = layout["values"]
        if "layout" in filters and layout["name"] not in filters["layout"]:
            continue
        if any(key in filters and values.get(key) not in filters[key] for key in values):
            continue
        selected.append(layout)
    return selected


def _options(values):
    if "devices" in values:
        return f"--devices {values['devices']}"
    if "nproc" in values:
        options = f"--nproc {values['nproc']}"
        if "threads" in values:
            options += f" --threads {values['threads']}"
        return options
    return ""


def generate_rows(definition, profile="full", nodes=None):
    selected_nodes, filters = _profile(definition, profile, nodes)
    rows = []
    runtime = {name: "{" + name + "}" for name in RUNTIME_FIELDS}
    for group in definition["groups"]:
        axes = []
        for axis in group["axes"]:
            values = group.get("select", {}).get(axis, definition["axes"][axis])
            if axis in filters:
                values = [value for value in values if value in filters[axis]]
            if not values:
                break
            axes.append(values)
        else:
            combinations = list(product(*axes))
            for node_count in sorted(set(group["nodes"]) & selected_nodes):
                binding = definition["bindings"][str(node_count)]
                layouts = _layouts(group, definition, filters, node_count)
                for layout in layouts:
                    for combination in combinations:
                        values = dict(zip(group["axes"], combination))
                        values.update(layout["values"])
                        values.update(runtime)
                        values.update({
                            "nodes": node_count, "layout": layout["name"],
                            "options": _options(layout["values"]),
                            "operation_name": str(values.get("operation", "")).removesuffix("_perf"),
                            "size_name": str(values.get("size", "")).replace(".", "point"),
                            "size_name_underscore": str(values.get("size", "")).replace(".", "_"),
                            "config_name": str(values.get("config", "")).replace(".json", "_json").replace(".", "_"),
                        })
                        command = group.get("command", definition["defaults"].get("command"))
                        command = group.get("command_by_nodes", {}).get(str(node_count), command)
                        rows.append({
                            "plan": binding["plan"], "suite": binding["suite"],
                            "name": group.get("case_name", definition["defaults"]["case_name"]).format(**values),
                            "check": group.get("check", definition["defaults"]["check"]),
                            "cmd": command.format(**values),
                        })
    if definition["profiles"][profile].get("include_explicit", False):
        for explicit in definition.get("explicit_cases", []):
            if explicit["nodes"] not in selected_nodes:
                continue
            binding = definition["bindings"][str(explicit["nodes"])]
            rows.append({"plan": binding["plan"], "suite": binding["suite"],
                         "name": explicit["name"], "check": explicit["check"], "cmd": explicit["cmd"]})
    if not rows:
        raise ValueError("Generation profile selected no Cases")
    keys = [(row["plan"], row["suite"], row["name"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Generated Case keys are not unique")
    if any(not run_cases.SAFE_NAME.fullmatch(row["name"]) for row in rows):
        raise ValueError("Generated Case name is not portable")
    return sorted(rows, key=lambda row: (row["plan"], row["suite"], row["name"]))


def generate_bundle(definition_path, suites_path, source_revision, runner_revision,
                    profile="full", nodes=None):
    definition_path = Path(definition_path).resolve()
    suites_path = Path(suites_path).resolve()
    if not run_cases.HEX40.fullmatch(source_revision) or not run_cases.HEX40.fullmatch(runner_revision):
        raise ValueError("Source and runner revisions must be full lowercase Git SHAs")
    definition = load_definition(definition_path)
    suite_document = _json(suites_path)
    if (set(suite_document) != {"schema_version", "suites"} or suite_document["schema_version"] != 1
            or not isinstance(suite_document["suites"], list)):
        raise ValueError("Unsupported public Suite definitions")
    suite_rows = {}
    for row in suite_document["suites"]:
        if set(row) != set(run_cases.SUITE_FIELDS):
            raise ValueError("Invalid public Suite definition")
        key = (row["plan"], row["suite"])
        if key in suite_rows:
            raise ValueError("Duplicate public Suite definition")
        suite_rows[key] = row
    rows = generate_rows(definition, profile, nodes)
    selected_suites = {}
    for row in rows:
        key = (row["plan"], row["suite"])
        if key not in suite_rows:
            raise ValueError(f"Generated Case references an undefined Suite: {key}")
        selected_suites[key] = suite_rows[key]
    case_keys = "".join(f"{row['plan']}/{row['suite']}/{row['name']}\n" for row in rows)
    generation = {
        "schema_version": 1, "family": definition["family"], "profile": profile,
        "nodes": nodes, "case_count": len(rows),
        "definition_sha256": sha256(definition_path.read_bytes()).hexdigest(),
        "suite_definitions_sha256": sha256(suites_path.read_bytes()).hexdigest(),
        "case_keys_sha256": sha256(case_keys.encode()).hexdigest(),
    }
    return {
        "metadata": {"format_version": run_cases.FORMAT_VERSION,
                     "source_revision": source_revision, "runner_revision": runner_revision,
                     "xtest_version": "not-used"},
        "suites": [selected_suites[key] for key in sorted(selected_suites)],
        "rows": rows, "generation": generation,
        "environments": sorted({row["env"] for row in selected_suites.values()}),
    }


def combine_bundles(bundles):
    if not bundles:
        raise ValueError("No generated bundles to combine")
    metadata = bundles[0]["metadata"]
    profile = bundles[0]["generation"]["profile"]
    nodes = bundles[0]["generation"]["nodes"]
    rows, suites, environments = {}, {}, set()
    for bundle in bundles:
        if bundle["metadata"] != metadata or bundle["generation"]["profile"] != profile \
                or bundle["generation"]["nodes"] != nodes:
            raise ValueError("Cannot combine bundles with different metadata or selection")
        for row in bundle["rows"]:
            key = (row["plan"], row["suite"], row["name"])
            if key in rows and rows[key] != row:
                raise ValueError("Combined Case key has conflicting commands")
            rows[key] = row
        for row in bundle["suites"]:
            key = (row["plan"], row["suite"])
            if key in suites and suites[key] != row:
                raise ValueError("Combined Suite key has conflicting resources")
            suites[key] = row
        environments.update(bundle["environments"])
    ordered_rows = [rows[key] for key in sorted(rows)]
    case_keys = "".join(f"{row['plan']}/{row['suite']}/{row['name']}\n" for row in ordered_rows)
    generation = {
        "schema_version": 1, "profile": profile, "nodes": nodes, "case_count": len(ordered_rows),
        "families": [bundle["generation"] for bundle in bundles],
        "case_keys_sha256": sha256(case_keys.encode()).hexdigest(),
    }
    return {"metadata": metadata, "suites": [suites[key] for key in sorted(suites)],
            "rows": ordered_rows, "generation": generation, "environments": sorted(environments)}


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_bundle(bundle, output, env_root):
    output = Path(output).resolve()
    env_root = Path(env_root).resolve()
    if output.exists():
        raise ValueError("Output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (temporary / "envs").mkdir()
        (temporary / "export.json").write_text(
            json.dumps(bundle["metadata"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "generation.json").write_text(
            json.dumps(bundle["generation"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_csv(temporary / "suites.csv", run_cases.SUITE_FIELDS, bundle["suites"])
        _write_csv(temporary / "standard.csv", run_cases.CASE_FIELDS, bundle["rows"])
        _write_csv(temporary / "notsm.csv", run_cases.CASE_FIELDS, [])
        for name in bundle["environments"]:
            if not run_cases.SAFE_NAME.fullmatch(name) or not name.endswith(".env"):
                raise ValueError("Invalid environment filename")
            source = env_root / name
            if not source.is_file():
                raise ValueError(f"Missing environment file: {name}")
            shutil.copyfile(source, temporary / "envs" / name)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    selected = run_cases.load_inputs(output, "standard")
    if len(selected["cases"]) != bundle["generation"]["case_count"]:
        raise ValueError("Generated output failed exact consumer coverage validation")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", type=Path, action="append", required=True)
    parser.add_argument("--suites", type=Path, required=True)
    parser.add_argument("--env-root", type=Path, required=True)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runner-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        bundles = [generate_bundle(definition, args.suites, args.source_revision,
                                   args.runner_revision, args.profile, args.nodes)
                   for definition in args.definition]
        bundle = bundles[0] if len(bundles) == 1 else combine_bundles(bundles)
        write_bundle(bundle, args.output, args.env_root)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(bundle["generation"], sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
