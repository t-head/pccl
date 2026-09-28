#!/usr/bin/env python3
"""Parse JUnit XML test results and generate a Markdown summary.

Usage:
    python3 display_test_summary.py <path-to-test-results.xml>

If the environment variable GITHUB_STEP_SUMMARY is set, the Markdown
summary is appended to that file (for GitHub Actions).  Otherwise it is
printed to stdout, which is useful for local debugging.
"""

import argparse
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


def generate_markdown_summary(stats):
    """Render the parsed statistics as a Markdown string.

    Uses an HTML table layout for each section.  All inner content is pure
    HTML because GitHub Flavored Markdown does not parse Markdown syntax
    (e.g. pipe tables) inside an HTML block.
    """
    lines = []
    run_number = os.environ.get("GITHUB_RUN_NUMBER")
    repo = os.environ.get("GITHUB_REPOSITORY", "").split("/")[-1]
    branch = os.environ.get("GITHUB_REF_NAME")

    # Build title: <Repo> Test Results #<run-number> @ <branch>
    title_parts = []
    if run_number:
        title_parts.append(f"#{run_number}")
    if branch:
        title_parts.append(f"@ {branch}")

    suffix = f" {' '.join(title_parts)}" if title_parts else ""
    repo_prefix = f"{repo} " if repo else ""
    lines.append(f"## {repo_prefix}Test Results{suffix}\n")
    # ---- Summary overview ----
    lines.append('<h3>概览</h3>')
    lines.append('<table>')
    lines.append('<tr>'
                 '<th>Total Cases</th>'
                 '<th>✅ Passed</th>'
                 '<th>❌ Failed</th>'
                 '<th>⏭ Skipped</th>'
                 '<th>⏱ Run Time</th>'
                 '</tr>')
    lines.append('<tr>'
                 f"<td>{stats['total']}</td>"
                 f"<td>{stats['passed']}</td>"
                 f"<td>{stats['failures'] + stats['errors']}</td>"
                 f"<td>{stats['skipped']}</td>"
                 f"<td>{stats['time']:.1f}s</td>"
                 '</tr>')
    lines.append('</table>')
    # ---- Test-case details (below overview) ----
    lines.append('<h3>Test Details</h3>')
    lines.append('<table>')
    lines.append('<tr><th>Test Case</th><th>Time</th><th>Status</th></tr>')
    for tc in stats["testcases"]:
        if tc.get("skipped"):
            status = "⏭ Skipped"
        elif tc["passed"]:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        lines.append(f"<tr><td>{tc['name']}</td><td>{tc['time']:.1f}s</td><td>{status}</td></tr>")
    lines.append('</table>')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary from JUnit XML test results."
    )
    parser.add_argument(
        "xml_files",
        nargs="+",
        help="Path(s) to the test-results.xml file(s)",
    )
    args = parser.parse_args()

    stats_list = []
    for xml_file in args.xml_files:
        if not os.path.exists(xml_file):
            print(f"❌ File not found: {xml_file}", file=sys.stderr)
            continue
        try:
            stats_list.append(parse_test_results(xml_file))
        except ET.ParseError as e:
            print(f"❌ Failed to parse XML {xml_file}: {e}", file=sys.stderr)

    if not stats_list:
        print("❌ No valid XML files to summarize.", file=sys.stderr)
        sys.exit(1)

    stats = merge_stats(stats_list)

    markdown = generate_markdown_summary(stats)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(markdown)
        print(f"✅ Summary written to {summary_path}")
    else:
        print(markdown)


if __name__ == "__main__":
    main()
