#!/usr/bin/env python3
"""Run PR test cases from a caselist and emit a JUnit XML report.

Each non-empty, non-comment line in the caselist is executed as a shell
command (relative paths resolve against the repository root).
"""

import subprocess
import os
import sys
import time
import argparse
import re
import concurrent.futures
from junit_xml import TestCase, TestSuite

# Configuration: output file name
OUTPUT_XML = "test-results.xml"


def derive_test_name(case_command):
    """Build a test name from the full shell command.

    Keep the whole command (interpreter included) so each case stays
    distinguishable, and collapse any run of whitespace into a single
    underscore so the name is a single token in the JUnit report.
    """
    return re.sub(r"\s+", "_", case_command.strip())


def run_one_case(i, case_command, total_count):
    """
    Run a single test case.
    case_command: full shell command string with arguments.

    Returns a tuple (junit_case, is_success, output_text).
    """
    test_name = derive_test_name(case_command)

    junit_case = TestCase(name=test_name, classname="shTest")

    header = f"\n[{i}/{total_count}] Running: {test_name}\nCommand: {case_command}\n{'-' * 50}\n"
    print(header, end='')

    start_time = time.time()
    stdout_output = []

    try:
        process = subprocess.Popen(
            case_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                stdout_output.append(line)

        process.wait(timeout=1200)
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration

        output_text = "".join(stdout_output)
        if process.returncode == 0:
            result_msg = f">>> Result: [PASS]"
            is_success = True
        else:
            msg = f"Exit code: {process.returncode}"
            result_msg = f">>> Result: [FAIL] ({msg})"
            junit_case.add_failure_info(message="Test failed", output=output_text)
            is_success = False

        return junit_case, is_success, output_text + result_msg + "\n"

    except subprocess.TimeoutExpired:
        process.kill()
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration
        output_text = "".join(stdout_output) + "Timeout\n"
        junit_case.add_failure_info(message="Timeout", output="Test timed out")
        return junit_case, False, output_text + ">>> Result: [FAIL] (Timeout)\n"

    except Exception as e:
        duration = time.time() - start_time
        junit_case.elapsed_sec = duration
        output_text = "".join(stdout_output) + f"Exception: {e}\n"
        junit_case.add_error_info(message="Exception occurred", output=str(e))
        return junit_case, False, output_text + f">>> Result: [FAIL] (Exception: {e})\n"


def parse_commands(filepath):
    """
    Parse test commands from a file.
    Accept all non-empty, non-comment lines as shell commands.
    """
    test_commands = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            test_commands.append(line)
    return test_commands


def resolve_build_dir(build_dir):
    """Resolve the test working directory; default to the repository root."""
    if build_dir:
        return os.path.abspath(build_dir)

    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def run_cases(caselist_path, build_dir, suite_name, output_xml=OUTPUT_XML, workers=1):
    if not os.path.exists(caselist_path):
        print(f"Error: test list file not found: {caselist_path}")
        sys.exit(1)

    test_commands = parse_commands(caselist_path)

    if not test_commands:
        print("No valid test commands found in list")
        return

    # Switch to build directory
    try:
        os.chdir(build_dir)
        print(f"Working directory changed to: {os.getcwd()}")
    except Exception as e:
        print(f"Failed to change directory: {e}")
        sys.exit(1)

    total_count = len(test_commands)
    indexed_commands = list(enumerate(test_commands, 1))

    all_test_objects = []
    global_passed_count = 0

    if workers > 1:
        print(f"Running {total_count} test case(s) in parallel with {workers} workers...\n")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # executor.map preserves the order of inputs.
            results = executor.map(
                lambda args: run_one_case(args[0], args[1], total_count),
                indexed_commands
            )
            for junit_case_obj, is_success, output_text in results:
                print(output_text)
                all_test_objects.append(junit_case_obj)
                if is_success:
                    global_passed_count += 1
    else:
        print(f"Running {total_count} test case(s) sequentially...\n")
        for i, cmd in indexed_commands:
            try:
                junit_case_obj, is_success, output_text = run_one_case(i, cmd, total_count)
                print(output_text)
                all_test_objects.append(junit_case_obj)
                if is_success:
                    global_passed_count += 1
            except Exception as e:
                print(f"Critical error: {e}")

    # Resolve after changing directory so relative paths use the test working directory.
    output_xml_path = os.path.abspath(output_xml)
    output_xml_dir = os.path.dirname(output_xml_path)
    os.makedirs(output_xml_dir, exist_ok=True)

    # Generate JUnit XML report
    suite = TestSuite(suite_name, all_test_objects)
    with open(output_xml_path, 'w', encoding='utf-8') as f:
        TestSuite.to_file(f, [suite], prettyprint=True)

    print(f"\nTests completed. Total: {total_count}, Passed: {global_passed_count}")
    print(f"Report generated: {output_xml_path}")

    if global_passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PR tests from a caselist.")
    parser.add_argument("--caselist", required=True, help="Path to the caselist file")
    parser.add_argument(
        "--suite-name",
        default="PR Tests",
        help="Name used for the JUnit testsuite (default: PR Tests)",
    )
    parser.add_argument(
        "--build-dir",
        help="Directory in which test commands run. Defaults to the repository root.",
    )
    parser.add_argument(
        "--output-xml",
        default=OUTPUT_XML,
        help=(
            "JUnit XML output path. Absolute paths are used directly; relative "
            f"paths are resolved from the test working directory (default: {OUTPUT_XML})."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of test cases to run in parallel (default: 1).",
    )
    args = parser.parse_args()

    caselist_abs_path = os.path.abspath(args.caselist)
    build_dir = resolve_build_dir(args.build_dir)

    run_cases(caselist_abs_path, build_dir, args.suite_name, args.output_xml, args.workers)
