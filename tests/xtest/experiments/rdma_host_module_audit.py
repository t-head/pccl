#!/usr/bin/env python3
"""Read-only host driver/module inventory; never execute host programs or load modules."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess


MODULES = {'alixpu', 'alipci', 'ib_core', 'ib_uverbs', 'mlx5_core', 'mlx5_ib', 'mlx_compat'}


def host_path(root, path):
    """Resolve host absolute symlinks within the mounted host, not the container."""
    parts = list(PurePosixPath(str(path)).parts)
    resolved = []
    links = 0
    while parts:
        part = parts.pop(0)
        if part in ('/', '.'):
            continue
        if part == '..':
            if resolved:
                resolved.pop()
            continue
        candidate = root.joinpath(*resolved, part)
        if candidate.is_symlink():
            links += 1
            if links > 40:
                raise ValueError('Too many host symlinks')
            target = PurePosixPath(os.readlink(candidate))
            if target.is_absolute():
                resolved = []
            parts = list(target.parts) + parts
        else:
            resolved.append(part)
    return root.joinpath(*resolved)


def readonly_mount(path, mountinfo):
    path = str(path)
    matches = []
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        point = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), fields[4])
        if path == point or path.startswith(point.rstrip('/') + '/'):
            matches.append((len(point), 'ro' in fields[5].split(',')))
    return bool(matches) and max(matches)[1]


def selected_module(name):
    return name in MODULES or ('alixpu' in name and 'peer' in name)


def module_name(path):
    return re.sub(r'\.ko(?:\.(?:xz|gz|zst))?$', '', path.name).replace('-', '_')


def metadata(path):
    result = subprocess.run(['modinfo', str(path)], capture_output=True, text=True, timeout=10)
    if result.returncode:
        return {'error': result.stderr.strip(), 'return_code': result.returncode}
    fields = {}
    for line in result.stdout.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key == 'parm':
                fields.setdefault('parameters', []).append(value.strip())
            if key in ('filename', 'name', 'version', 'srcversion', 'vermagic', 'depends',
                       'description', 'license', 'signer'):
                fields[key] = value.strip()
    return fields


def audit(root, kernel, mountinfo):
    if not readonly_mount(root, mountinfo):
        raise ValueError('Host root must be mounted read-only before inspection')
    report = {'hostname': os.uname().nodename, 'kernel': kernel, 'read_only': True,
              'loaded_modules': {}, 'module_files': [], 'configuration': [],
              'package_notes': [], 'errors': []}
    for line in Path('/proc/modules').read_text().splitlines():
        fields = line.split()
        if fields and selected_module(fields[0]):
            entry = {'proc_modules': line}
            for key in ('version', 'srcversion', 'taint'):
                path = Path('/sys/module') / fields[0] / key
                if path.is_file():
                    entry[key] = path.read_text().strip()
            parameters = Path('/sys/module') / fields[0] / 'parameters'
            if parameters.is_dir():
                entry['parameters'] = {}
                for path in parameters.iterdir():
                    try:
                        if path.is_file():
                            entry['parameters'][path.name] = path.read_text().strip()
                    except OSError as error:
                        entry['parameters'][path.name] = {'read_error': str(error)}
            report['loaded_modules'][fields[0]] = entry
    for path in ('/sys/module/alixpu_peermem', '/sys/module/alixpu_peer_mem',
                 '/sys/kernel/mm/memory_peers/alixpu_peer_mem'):
        actual = Path(path)
        entry = {'exists': actual.is_dir()}
        if (actual / 'version').is_file():
            entry['version'] = (actual / 'version').read_text().strip()
        report.setdefault('peer_memory_interfaces', {})[path] = entry
    version = Path('/proc/driver/alixpu/version')
    if version.is_file():
        report['loaded_driver_version'] = version.read_text().strip()
    modules = host_path(root, '/lib/modules')
    if not modules.is_dir() or not readonly_mount(modules, mountinfo):
        report['errors'].append('Host module tree unavailable or not read-only')
        return report
    report['installed_kernel_trees'] = sorted(p.name for p in modules.iterdir() if p.is_dir())
    for tree in modules.iterdir():
        if not tree.is_dir() or not readonly_mount(tree, mountinfo):
            continue
        for directory, children, files in os.walk(tree, followlinks=False):
            children[:] = [name for name in children if name not in ('build', 'source')]
            if not readonly_mount(directory, mountinfo):
                children[:] = []
                continue
            for filename in files:
                path = Path(directory) / filename
                if not re.search(r'\.ko(?:\.(?:xz|gz|zst))?$', filename) or not selected_module(module_name(path)):
                    continue
                target = host_path(root, '/' + str(path.relative_to(root)))
                if not readonly_mount(target, mountinfo) or not target.is_file():
                    report['errors'].append(f'Unavailable module target: {path.relative_to(root)}')
                    continue
                if not re.search(r'\.ko(?:\.(?:xz|gz|zst))?$', target.name):
                    report['errors'].append(f'Unexpected module target: {target.relative_to(root)}')
                    continue
                record = {'path': str(path.relative_to(root)), 'resolved_path': str(target.relative_to(root)),
                          'kernel_tree': tree.name, 'for_running_kernel': tree.name == kernel,
                          'bytes': target.stat().st_size, 'metadata': metadata(target)}
                info = record['metadata']
                record['vermagic_matches_running_kernel'] = info.get('vermagic', '').split(' ')[0] == kernel
                loaded = report['loaded_modules'].get(info.get('name', module_name(path)), {})
                if loaded.get('srcversion') and info.get('srcversion'):
                    record['matches_loaded_srcversion'] = loaded['srcversion'] == info['srcversion']
                report['module_files'].append(record)
    os_release = host_path(root, '/etc/os-release')
    if os_release.is_file() and readonly_mount(os_release, mountinfo):
        report['host_os_release'] = os_release.read_text()
    pattern = re.compile(r'alixpu|peermem|peer_mem|mlx5|OFED|MLNX|GDR|PACKAGE_VERSION|kernel', re.I)
    for base in ('/etc/modprobe.d', '/etc/modules-load.d'):
        directory = host_path(root, base)
        if not directory.is_dir() or not readonly_mount(directory, mountinfo):
            continue
        for path in directory.glob('*.conf'):
            target = host_path(root, '/' + str(path.relative_to(root)))
            if not target.is_file() or not readonly_mount(target, mountinfo):
                continue
            lines = [line for line in target.read_text(errors='replace').splitlines() if pattern.search(line)]
            if lines:
                report['configuration'].append({'path': str(path.relative_to(root)), 'relevant_lines': lines[:80]})
    for base in ('/usr/src', '/var/lib/dkms'):
        directory = host_path(root, base)
        if not directory.is_dir() or not readonly_mount(directory, mountinfo):
            continue
        for package in directory.iterdir():
            if not package.is_dir() or not re.search(r'alixpu|peermem|mlnx|ofed', package.name, re.I):
                continue
            report['package_notes'].append({'package_path': str(package.relative_to(root))})
            found = 0
            for parent, children, files in os.walk(package, followlinks=False):
                children[:] = [name for name in children if name not in ('build', '.git', 'kernel', 'drivers')]
                if Path(parent).relative_to(package).parts and len(Path(parent).relative_to(package).parts) > 3:
                    children[:] = []
                for name in files:
                    if found >= 12 or not (name.lower().startswith(('readme', 'release', 'requirements')) or name == 'dkms.conf'):
                        continue
                    target = host_path(root, '/' + str((Path(parent) / name).relative_to(root)))
                    if not target.is_file() or not readonly_mount(target, mountinfo) or target.stat().st_size > 131072:
                        continue
                    lines = [line for line in target.read_text(errors='replace').splitlines() if pattern.search(line)]
                    report['package_notes'].append({'path': str(target.relative_to(root)), 'relevant_lines': lines[:100]})
                    found += 1
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.host_root.absolute(), os.uname().release, Path('/proc/self/mountinfo').read_text())
    args.output.write_text(json.dumps(result, indent=2) + '\n')
