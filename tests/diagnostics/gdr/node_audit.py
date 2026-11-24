#!/usr/bin/env python3
"""Read kernel module state on failed machines using owned, CPU-only pods."""
import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import time

FAILED_HOSTS = tuple(f"na131g-m890-{number:02d}" for number in (*range(1, 9), *range(17, 25)))
PAIR = ("na131g-m890-03", "na131g-m890-04")
HISTORICAL_MISSING = ("01", "02", "05", "07", "08", "18", "19", "20", "21", "23")
IMAGE = "download.t-head.cn/docker_release/ppu:sdk2.2.0-hggcrt3-ubuntu24.04-py312-20261005"


def classify(modules, sysfs_loaded):
    if sysfs_loaded or (modules is not None and "alixpu_peermem" in modules):
        return "loaded"
    # An empty/hidden module view is not evidence that a host module is absent.
    if modules and any(name.startswith(("alixpu", "mlx5", "ib_core")) for name in modules):
        return "not_loaded"
    return "unknown"


PROBE = inspect.getsource(classify) + '''
import json, os, platform, time
from pathlib import Path
errors=[]
try:
    modules={line.split()[0] for line in Path('/proc/modules').read_text().splitlines() if line.split()}
except OSError as error:
    modules=None
    errors.append(str(error))
loaded=Path('/sys/module/alixpu_peermem').is_dir()
versions={}
for name in sorted(modules or []):
    if name.startswith(('alixpu', 'mlx5', 'ib_core')):
        path=Path('/sys/module')/name/'version'
        try: versions[name]=path.read_text().strip() if path.exists() else None
        except OSError as error: versions[name]=None; errors.append(str(error))
print(json.dumps({'marker':'pccl-gdr-node-audit-v1', 'node_name':os.environ['NODE_NAME'],
    'checked_at_unix':time.time(), 'kernel_release':platform.release(),
    'peermem':classify(modules,loaded), 'sysfs_peermem_present':loaded,
    'module_count':len(modules) if modules is not None else None,
    'driver_modules':versions, 'errors':errors},sort_keys=True))
'''


def kube(*args, payload=None):
    # Match ci-infra's adapter: this runner loses in-cluster credentials when
    # kubectl receives a nonzero --request-timeout. Bound the process externally.
    process = subprocess.run(["kubectl", *args],
        input=json.dumps(payload) if payload is not None else None,
        text=True, capture_output=True, timeout=35)
    if process.returncode:
        raise RuntimeError(process.stderr.strip()[:2000])
    return process.stdout


def pod_manifest(name, namespace, run, node_label, image):
    return {"apiVersion": "v1", "kind": "Pod", "metadata": {
        "name": name, "namespace": namespace, "labels": {"pccl-gdr-audit": run}},
        "spec": {"schedulerName": "default-scheduler", "restartPolicy": "Never",
            "automountServiceAccountToken": False, "enableServiceLinks": False,
            "activeDeadlineSeconds": 120,
            "nodeSelector": {"kubernetes.io/hostname": node_label},
            "tolerations": [{"key": "alibabacloud.com/ppu", "operator": "Exists", "effect": "NoSchedule"}],
            "containers": [{"name": "audit", "image": image, "imagePullPolicy": "IfNotPresent",
                "command": ["python3", "-c", PROBE],
                "env": [{"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
                        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"},
                        {"name": "CUDA_VISIBLE_DEVICES", "value": ""},
                        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"}],
                "resources": {"requests": {"cpu": "25m", "memory": "96Mi"},
                              "limits": {"cpu": "100m", "memory": "192Mi"}},
                "securityContext": {"runAsNonRoot": True, "runAsUser": 65534,
                    "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                    "capabilities": {"drop": ["ALL"]}, "seccompProfile": {"type": "RuntimeDefault"}}}]}}


def has_gpu_request(pod):
    for container in pod["spec"]["containers"]:
        for kind in ("requests", "limits"):
            if container.get("resources", {}).get(kind, {}).get("alibabacloud.com/ppu", "0") not in ("0", 0):
                return True
    return False


def audit(namespace, run, image, output, timeout):
    output.mkdir(parents=True, exist_ok=False)
    nodes = json.loads(kube("get", "nodes", "-o", "json"))["items"]
    mapping = {node["metadata"]["name"].split(".")[0]: node for node in nodes}
    rows = {}
    created = []
    pending = {}
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for host in FAILED_HOSTS:
            node = mapping.get(host)
            if node is None:
                rows[host] = {"peermem": "unknown", "error": "Node was not found in the authorized cluster"}
                continue
            label = node["metadata"].get("labels", {}).get("kubernetes.io/hostname")
            if not label:
                rows[host] = {"peermem": "unknown", "error": "Node has no hostname label"}
                continue
            name = f"pccl-gdr-audit-{run}-{host}"
            manifest = pod_manifest(name, namespace, run, label, image)
            try:
                review = json.loads(kube("create", "--dry-run=server", "-f", "-", "-o", "json", payload=manifest))
                if has_gpu_request(review):
                    raise RuntimeError("Admission would add a PPU to a CPU-only probe; probe was not created")
                pod = json.loads(kube("create", "-f", "-", "-o", "json", payload=manifest))
                created.append(name)
                if has_gpu_request(pod):
                    kube("delete", "pod", "-n", namespace, name, "--ignore-not-found", "--wait=false")
                    raise RuntimeError("Admission added a PPU resource to a CPU-only probe; aborting")
                pending[name] = host
            except RuntimeError as error:
                rows[host] = {"peermem": "unknown", "error": str(error)}
        deadline = time.monotonic() + timeout
        while pending and time.monotonic() < deadline:
            pods = json.loads(kube("get", "pods", "-n", namespace, "-l", f"pccl-gdr-audit={run}", "-o", "json"))["items"]
            for pod in pods:
                name = pod["metadata"]["name"]
                if name not in pending:
                    continue
                host = pending[name]
                phase = pod.get("status", {}).get("phase")
                if phase == "Succeeded":
                    text = kube("logs", "-n", namespace, name, "-c", "audit")
                    (output / f"{host}.log").write_text(text)
                    try:
                        result = json.loads(text.strip().splitlines()[-1])
                        assert result["marker"] == "pccl-gdr-node-audit-v1"
                        assert result["node_name"] == mapping[host]["metadata"]["name"]
                        containers = pod.get("status", {}).get("containerStatuses", [])
                        result["image_id"] = next((c.get("imageID") for c in containers if c["name"] == "audit"), None)
                        rows[host] = result
                    except (ValueError, KeyError, AssertionError, IndexError) as error:
                        rows[host] = {"peermem": "unknown", "error": f"Invalid probe evidence: {error}"}
                    del pending[name]
                elif phase == "Failed":
                    try:
                        text = kube("logs", "-n", namespace, name, "-c", "audit")
                    except RuntimeError as error:
                        text = str(error)
                    (output / f"{host}.log").write_text(text)
                    states = pod.get("status", {}).get("containerStatuses", [])
                    rows[host] = {"peermem": "unknown", "error": pod.get("status", {}).get("message", "CPU probe failed"),
                                  "container_states": states, "probe_log": text[-2000:]}
                    del pending[name]
            if pending:
                time.sleep(5)
        for name, host in pending.items():
            rows[host] = {"peermem": "unknown", "error": "CPU probe did not complete within the diagnostic deadline"}
    except (RuntimeError, KeyboardInterrupt) as error:
        for host in FAILED_HOSTS:
            rows.setdefault(host, {"peermem": "unknown", "error": f"Diagnostic interrupted: {error}"})
    finally:
        signal.signal(signal.SIGTERM, previous)
        for name in created:
            try:
                kube("delete", "pod", "-n", namespace, name, "--ignore-not-found", "--wait=false")
            except RuntimeError as error:
                print(f"Own probe cleanup failed for {name}: {error}", flush=True)
    complete = len(rows) == len(FAILED_HOSTS) and all(row["peermem"] in ("loaded", "not_loaded") for row in rows.values())
    selected = []
    for host in PAIR:
        node = mapping.get(host)
        if node:
            selected.append({"host": host, "node_name": node["metadata"]["name"],
                             "hostname_label": node["metadata"].get("labels", {}).get("kubernetes.io/hostname"),
                             "peermem": rows.get(host, {}).get("peermem", "unknown"),
                             "image_id": rows.get(host, {}).get("image_id")})
    summary = {"source_run": 45191076117, "checked_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete, "machine_count": len(FAILED_HOSTS), "machines": rows,
        "loaded": sum(row["peermem"] == "loaded" for row in rows.values()),
        "not_loaded": sum(row["peermem"] == "not_loaded" for row in rows.values()),
        "unknown": sum(row["peermem"] == "unknown" for row in rows.values()),
        "selected_pair": selected, "historical_missing_suffixes": HISTORICAL_MISSING,
        "historical_log_window": "2026-11-23 22:22-22:26 Asia/Shanghai",
        "scope": "Current module state; not a reconstruction of each failure instant",
        "ppu_requested": 0, "host_configuration_modified": False}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="ppu-sched")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not args.run_id.isdecimal() or args.timeout <= 0:
        parser.error("run-id must be numeric and timeout positive")
    return audit(args.namespace, args.run_id, args.image, args.output, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
