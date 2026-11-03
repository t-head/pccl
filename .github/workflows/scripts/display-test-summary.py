#!/usr/bin/env python3
"""Parse JUnit XML test results and generate a Markdown summary.

Usage:
    python3 display-test-summary.py <path-to-test-results.xml>... [--env-info <path>]
    python3 display-test-summary.py --results-dir <dir>

Each XML file renders one summary section with a collapsible env-details
block (from a sibling env_info.json, or from --env-info) and a collapsible
test-details table.  --results-dir walks <dir> recursively for XML files
and labels each section by its path relative to <dir>, so the aggregate
summary job can render every machine/script result in one page.

If the environment variable GITHUB_STEP_SUMMARY is set, the Markdown
summary is appended to that file (for GitHub Actions).  Otherwise it is
printed to stdout, which is useful for local debugging.
"""

import argparse
import glob
import html
import json
import os
import sys
import xml.etree.ElementTree as ET


def parse_test_results(xml_path):
    """Parse a JUnit XML file and return a dict with summary statistics
    plus a list of per-testcase dicts."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    total = int(root.get("tests", 0))
    failures = int(root.get("failures", 0))
    errors = int(root.get("errors", 0))
    skipped = int(root.get("skipped", 0))
    passed = total - failures - errors - skipped
    time = float(root.get("time", 0))

    testcases = []
    for tc in root.findall(".//testcase"):
        name = tc.get("name", "unknown")
        t = float(tc.get("time", 0))
        is_skipped = tc.find("skipped") is not None
        is_failure = tc.find("failure") is not None
        is_error = tc.find("error") is not None
        testcases.append({
            "name": name,
            "time": t,
            "passed": not is_failure and not is_error and not is_skipped,
            "skipped": is_skipped,
        })

    return {
        "total": total,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "passed": passed,
        "time": time,
        "testcases": testcases,
    }


def parse_env_info(json_path):
    """Load runtime environment information from JSON."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def merge_stats(stats_list):
    """Merge multiple parsed result dicts into a single summary."""
    merged = {
        "total": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "passed": 0,
        "time": 0.0,
        "testcases": [],
    }
    for stats in stats_list:
        merged["total"] += stats["total"]
        merged["failures"] += stats["failures"]
        merged["errors"] += stats["errors"]
        merged["skipped"] += stats["skipped"]
        merged["passed"] += stats["passed"]
        merged["time"] += stats["time"]
        merged["testcases"].extend(stats["testcases"])
    return merged


def render_overview_table(stats):
    """Render the summary statistics table (pure HTML table rows)."""
    return [
        '<h3>概览</h3>',
        '<table>',
        '<tr>'
        '<th>Total Cases</th>'
        '<th>✅ Passed</th>'
        '<th>❌ Failed</th>'
        '<th>⏭ Skipped</th>'
        '<th>⏱ Run Time</th>'
        '</tr>',
        '<tr>'
        f"<td>{stats['total']}</td>"
        f"<td>{stats['passed']}</td>"
        f"<td>{stats['failures'] + stats['errors']}</td>"
        f"<td>{stats['skipped']}</td>"
        f"<td>{stats['time']:.1f}s</td>"
        '</tr>',
        '</table>',
    ]


def render_env_details(env_info):
    """Render the collapsible environment details block."""
    if not env_info:
        return []
    lines = ['<details>']
    lines.append(f"<summary><b>Env Details ({len(env_info)})</b></summary>")
    lines.append('<table>')
    lines.append('<tr><th>Item</th><th>Value</th></tr>')
    for key, value in env_info.items():
        safe_key = html.escape(str(key))
        safe_value = html.escape(str(value))
        lines.append(
            f"<tr><td><code>{safe_key}</code></td><td><pre>{safe_value}</pre></td></tr>")
    lines.append('</table>')
    lines.append('</details>')
    return lines


def render_test_details(testcases):
    """Render the collapsible per-testcase details block."""
    lines = ['<details>']
    lines.append(f"<summary><b>Test Details ({len(testcases)})</b></summary>")
    lines.append('<table>')
    lines.append('<tr><th>Test Case</th><th>Time</th><th>Status</th></tr>')
    for tc in testcases:
        if tc.get("skipped"):
            status = "⏭ Skipped"
        elif tc["passed"]:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        lines.append(
            f"<tr><td>{html.escape(tc['name'])}</td>"
            f"<td>{tc['time']:.1f}s</td><td>{status}</td></tr>")
    lines.append('</table>')
    lines.append('</details>')
    return lines


def generate_markdown_summary(stats, env_info=None, section_title=None):
    """Render one section: optional title, overview, env and test details."""
    lines = []
    if section_title:
        lines.append(f"### {html.escape(section_title)}\n")
    lines.extend(render_overview_table(stats))
    lines.extend(render_env_details(env_info))
    lines.extend(render_test_details(stats["testcases"]))
    return "\n".join(lines) + "\n"


def build_title():
    """Build the page title: <Repo> Test Results #<run-number> @ <branch>."""
    run_number = os.environ.get("GITHUB_RUN_NUMBER")
    repo = os.environ.get("GITHUB_REPOSITORY", "").split("/")[-1]
    branch = os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")

    title_parts = []
    if run_number:
        title_parts.append(f"#{run_number}")
    if branch:
        title_parts.append(f"@ {branch}")

    suffix = f" {' '.join(title_parts)}" if title_parts else ""
    repo_prefix = f"{repo} " if repo else ""
    return f"## {repo_prefix}Test Results{suffix}\n"


def load_env_info_safe(path):
    """Load env_info.json, reporting errors instead of failing the summary."""
    try:
        return parse_env_info(path)
    except (json.JSONDecodeError, OSError) as e:
        print(f"❌ Failed to parse environment info {path}: {e}", file=sys.stderr)
        return None


def collect_sections(xml_files, results_dir=None, env_info=None):
    """Parse every XML file and return (merged_stats, markdown_sections)."""
    stats_list = []
    sections = []
    single = len(xml_files) == 1
    for xml_file in xml_files:
        try:
            stats = parse_test_results(xml_file)
        except ET.ParseError as e:
            print(f"❌ Failed to parse XML {xml_file}: {e}", file=sys.stderr)
            continue
        stats_list.append(stats)

        # --env-info applies to the single-XML invocation; otherwise pick up
        # the env_info.json written next to each XML file automatically.
        section_env = env_info if (env_info is not None and single) else None
        if section_env is None:
            sibling = os.path.join(os.path.dirname(xml_file) or ".", "env_info.json")
            if os.path.exists(sibling):
                section_env = load_env_info_safe(sibling)

        if results_dir:
            label = os.path.relpath(os.path.dirname(xml_file), results_dir)
            sections.append(generate_markdown_summary(stats, section_env, label))
        else:
            sections.append(generate_markdown_summary(stats, section_env))
    return (merge_stats(stats_list) if stats_list else None), sections


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary from JUnit XML test results.")
    parser.add_argument(
        "xml_files",
        nargs="*",
        help="Path(s) to the test-results.xml file(s)",
    )
    parser.add_argument(
        "--env-info",
        help="Optional path to env_info.json (single-XML mode)",
    )
    parser.add_argument(
        "--results-dir",
        help="Aggregate mode: recursively summarize XML files under this directory",
    )
    args = parser.parse_args()

    if args.results_dir:
        if not os.path.isdir(args.results_dir):
            print(f"No results directory found at {args.results_dir}", file=sys.stderr)
            return
        xml_files = sorted(
            glob.glob(os.path.join(args.results_dir, "**", "*.xml"), recursive=True))
        if not xml_files:
            print(f"No test-results.xml found under {args.results_dir}", file=sys.stderr)
            return
        merged, sections = collect_sections(xml_files, results_dir=args.results_dir)
        if merged is None:
            print("❌ No valid XML files to summarize.", file=sys.stderr)
            sys.exit(1)
        # One merged overview at the top, then one full section per result.
        markdown = build_title() + "\n" + "\n".join(render_overview_table(merged)) \
            + "\n\n" + "\n".join(sections)
    else:
        if not args.xml_files:
            parser.error("provide XML files or --results-dir")
        env_info = None
        if args.env_info:
            if not os.path.exists(args.env_info):
                print(f"❌ File not found: {args.env_info}", file=sys.stderr)
                sys.exit(1)
            env_info = load_env_info_safe(args.env_info)
            if env_info is None:
                sys.exit(1)
        xml_files = []
        for xml_file in args.xml_files:
            if not os.path.exists(xml_file):
                print(f"❌ File not found: {xml_file}", file=sys.stderr)
                continue
            xml_files.append(xml_file)
        merged, sections = collect_sections(xml_files, env_info=env_info)
        if merged is None:
            print("❌ No valid XML files to summarize.", file=sys.stderr)
            sys.exit(1)
        markdown = build_title() + "\n" + "\n".join(sections)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(markdown)
        print(f"✅ Summary written to {summary_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
