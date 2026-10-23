#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

#!/usr/bin/env python3

"""Compare bench_oki layouts and matched assembly timing windows (Python 3.6+)."""
import argparse
import csv
import datetime
import hashlib
import html
import json
import math
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

GROUPS = [
    ('pipeline_16x1', 'Pipeline: 16×1 / interleaved', 'bandwidth-with-assembly', 16, 1, 'interleaved'),
    ('pipeline_2x8', 'Pipeline: 2×8 / interleaved', 'bandwidth-with-assembly', 2, 8, 'interleaved'),
    ('pipeline_per_die', 'Pipeline: 2×8 / per-die', 'bandwidth-with-assembly', 2, 8, 'per-die'),
    ('batch_with', 'Single-batch: with assembly', 'bandwidth-with-assembly', 2, 8, 'per-die'),
    ('batch_without', 'Single-batch: without assembly', 'bandwidth-without-assembly', 2, 8, 'per-die'),
]
LABELS = {g[0]: g[1] for g in GROUPS}


def positive(text):
    if not re.fullmatch(r'[0-9]+', text) or not 0 < int(text) <= 2147483647:
        raise argparse.ArgumentTypeError('must be a positive integer <= INT_MAX')
    return int(text)


def sizes_arg(text):
    result = []
    for part in text.split(','):
        m = re.fullmatch(r'([0-9]+)(K|k|KiB|M|m|MiB)?', part.strip())
        if not m:
            raise argparse.ArgumentTypeError('sizes: comma-separated bytes or K/KiB/M/MiB')
        n = int(m[1]) * (1024 if (m[2] or '').lower().startswith('k') else
                         1048576 if m[2] else 1)
        if n < 1 or n > 16 * 1048576:
            raise argparse.ArgumentTypeError('each command must fit a 16-QP slice (1..16MiB)')
        if n in result:
            raise argparse.ArgumentTypeError('duplicate command size')
        result.append(n)
    return sorted(result)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def command_text(argv):
    return ' '.join(shlex.quote(str(x)) for x in argv)


def write_json(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def plan_runs(binary, config):
    cases = []
    for trial in range(1, config['trials'] + 1):
        # Reverse order on alternate trials to reduce systematic order bias.
        groups = GROUPS if trial % 2 else list(reversed(GROUPS))
        for size in config['sizes']:
            for key, label, mode, blocks, warps, mapping in groups:
                batch = key.startswith('batch_')
                copies = config['batch_copies'] if batch else config['pipeline_copies']
                rounds = 1 if batch else config['pipeline_rounds']
                args = [str(binary), '--mode', mode, '--cmd-bytes', str(size),
                        '--copies-per-round', str(copies), '--rounds', str(rounds),
                        '--reps', str(config['reps']), '--inflight-rounds', '1' if batch else 'auto',
                        '--blocks', str(blocks), '--warps-per-block', str(warps), '--mapping', mapping]
                cases.append(dict(id='t{:02d}_s{}_{}'.format(trial, size, key), trial=trial,
                                  group=key, size=size, copies=copies, rounds=rounds,
                                  reps=config['reps'], mode=mode, blocks=blocks, warps=warps,
                                  args=args, command=command_text(args)))
    return cases


def fields(line):
    return dict(re.findall(r'([\w/]+)=([^\s]+)', line))


def validate_output(text, case):
    if re.search(r'VERIFY FAIL|VERIFICATION FAILED|CQ mismatch|verified=NO|ABOVE LINK LIMIT', text):
        raise ValueError('verification failure or above-link-limit measurement')
    if 'benchmark done' not in text:
        raise ValueError('missing completion marker')
    reps = [fields(l) for l in text.splitlines() if l.startswith('BANDWIDTH_REP ')]
    summaries = [fields(l) for l in text.splitlines() if l.startswith('BANDWIDTH ')]
    if len(reps) != case['reps'] or len(summaries) != 1:
        raise ValueError('missing/extra repetition or summary rows')
    seen = set()
    values = []
    times = []
    data_bytes = 16 * case['copies'] * case['rounds'] * case['size']
    for r in reps:
        if (r['mode'] != case['mode'] or int(r['QPs']) != 16 or
                int(r['blocks']) != case['blocks'] or r['verified'] != 'yes' or
                int(r['bytes']) != data_bytes):
            raise ValueError('unexpected mode, layout, payload or verification state')
        rep, gbps, ms = int(r['rep']), float(r['GB/s']), float(r['event_ms'])
        if rep in seen or not 0 <= rep < case['reps']:
            raise ValueError('duplicate/invalid repetition index')
        if not math.isfinite(gbps) or not math.isfinite(ms) or not 0 < gbps <= 400 or ms <= 0:
            raise ValueError('invalid bandwidth/time')
        if not math.isclose(gbps, data_bytes/ms/1e6, rel_tol=0.001, abs_tol=0.001):
            raise ValueError('bandwidth does not match bytes / wall time')
        seen.add(rep)
        values.append(gbps)
        times.append(ms)
    summary = summaries[0]
    if (summary['mode'] != case['mode'] or int(summary['QPs']) != 16 or
            int(summary['blocks']) != case['blocks'] or int(summary['warps_per_block']) != case['warps'] or
            int(summary['reps']) != case['reps'] or summary['verified'] != 'yes'):
        raise ValueError('unexpected summary configuration')
    for suffix, numbers in [('GB/s', values), ('ms', times)]:
        for name, want in [('min', min(numbers)), ('median', statistics.median(numbers)), ('max', max(numbers))]:
            if not math.isclose(float(summary[name+'_'+suffix]), want, rel_tol=1e-6, abs_tol=2e-6):
                raise ValueError('summary statistic mismatch: ' + name + '_' + suffix)
    return dict(repetitions=reps, median_gbps=statistics.median(values),
                min_gbps=min(values), max_gbps=max(values), median_ms=statistics.median(times))


def aggregate(results, config):
    rows = []
    for size in config['sizes']:
        for group in GROUPS:
            runs = [r for r in results if r['status'] == 'PASS' and
                    r['size'] == size and r['group'] == group[0]]
            if not runs:
                continue
            medians = [r['median_gbps'] for r in runs]
            values = [float(p['GB/s']) for r in runs for p in r['repetitions']]
            rows.append(dict(size_bytes=size, group=group[0], label=group[1],
                             trials=len(runs), expected_trials=config['trials'],
                             complete=len(runs) == config['trials'],
                             trial_median_min_gbps=min(medians),
                             median_gbps=statistics.median(medians),
                             trial_median_max_gbps=max(medians),
                             all_reps_min_gbps=min(values), all_reps_median_gbps=statistics.median(values),
                             all_reps_max_gbps=max(values)))
    return rows


def write_csv(path, rows, keys):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def render_report(out, manifest, results, state):
    config = manifest['config']
    summaries = aggregate(results, config)
    write_json(out/'results.json', dict(manifest=manifest, state=state, runs=results, summary=summaries))
    write_csv(out/'summary.csv', summaries, ['size_bytes', 'group', 'label', 'trials', 'expected_trials',
              'complete', 'median_gbps', 'trial_median_min_gbps', 'trial_median_max_gbps',
              'all_reps_min_gbps', 'all_reps_median_gbps', 'all_reps_max_gbps'])
    rep_rows = []
    for r in results:
        if r['status'] == 'PASS':
            for p in r['repetitions']:
                rep_rows.append(dict(case_id=r['id'], trial=r['trial'], group=r['group'],
                                     size_bytes=r['size'], rep=p['rep'], gbps=p['GB/s'], event_ms=p['event_ms']))
    write_csv(out/'repetitions.csv', rep_rows, ['case_id','trial','group','size_bytes','rep','gbps','event_ms'])
    write_csv(out/'runs.csv', results, ['id','trial','group','size','status','returncode','seconds','error','log','command'])
    e = html.escape
    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def get(size, group):
        return next((r for r in summaries if r['size_bytes']==size and r['group']==group), None)
    def value(size, group):
        r = get(size,group)
        return '{:.2f}{}'.format(r['median_gbps'], '' if r['complete'] else ' *') if r else '—'
    def ratio(size, a, b, percent=False):
        ra,rb=get(size,a),get(size,b)
        if not ra or not rb or not ra['complete'] or not rb['complete']:return '—'
        v=ra['median_gbps']/rb['median_gbps']
        return '{:+.1f}%'.format((v-1)*100) if percent else '{:.2f}×'.format(v)
    pipeline = table(['cmd','16×1 interleaved','2×8 interleaved','2×8 per-die','Layout change','Mapping change'],
        [[str(s//1024)+'K' if s%1024==0 else str(s)+'B']+
         [value(s,g) for g in ['pipeline_16x1','pipeline_2x8','pipeline_per_die']]+
         [ratio(s,'pipeline_2x8','pipeline_16x1',True),ratio(s,'pipeline_per_die','pipeline_2x8',True)] for s in config['sizes']])
    matched = table(['cmd','With assembly GB/s','Without assembly GB/s','Timing window bandwidth ratio'],
        [[str(s//1024)+'K' if s%1024==0 else str(s)+'B',value(s,'batch_with'),value(s,'batch_without'),ratio(s,'batch_without','batch_with')] for s in config['sizes']])
    spread = table(['cmd bytes','Group','Completed trials','Trial median min','Trial median','Trial median max'],
        [[r['size_bytes'],r['label'],'{}/{}'.format(r['trials'],r['expected_trials']),
          '{:.3f}'.format(r['trial_median_min_gbps']),'{:.3f}'.format(r['median_gbps']),'{:.3f}'.format(r['trial_median_max_gbps'])] for r in summaries])
    details=[]
    for r in results:
        text=(out/r['log']).read_text(encoding='utf-8',errors='replace')
        details.append('<details><summary>{} · {} · {} B · trial {} · {}</summary><pre>{}</pre><p>{}</p><pre>{}</pre></details>'.format(
            e(r['status']),e(LABELS[r['group']]),r['size'],r['trial'],e(r['id']),e(r['command']),e(r.get('error','')),e(text)))
    passed=sum(r['status']=='PASS' for r in results)
    title='OKI Bandwidth Comparison · '+manifest['created_at']
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>body{margin:0;background:#f2f5f8;color:#193149;font:15px/1.7 system-ui,sans-serif}header{background:#132d47;color:white;padding:32px max(24px,calc((100vw - 1200px)/2))}main{max-width:1200px;margin:auto;padding:24px}section{background:white;padding:24px;border:1px solid #dce5ee;border-radius:10px;margin:0 0 20px}h1{margin:0;font-size:29px}h2{margin:0 0 10px;font-size:21px}p{margin:10px 0}.muted{color:#61768a}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{padding:10px;text-align:right;border-bottom:1px solid #dce5ee}th:first-child,td:first-child{text-align:left}th{background:#edf3f8}pre{background:#edf3f8;padding:15px;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:420px;overflow:auto;font-size:12px}details{border-bottom:1px solid #ddd;padding:10px 0}summary{cursor:pointer}a{color:#1265be}.notice{padding:12px;background:#fff4d9;border-left:4px solid #d99324}input{font:inherit;padding:8px;width:90%;margin:10px 0}@media(max-width:650px){main{padding:12px}section{padding:14px}}</style>
<header><h1>__TITLE__</h1><p>Same binary · __TRIALS__ processes × __REPS__ repetitions per group · Status: __STATE__</p></header><main>
<section><h2>Collection Status</h2><p>Passed __PASS__ / __TOTAL__ configuration processes.</p><p class="notice">__REASON__</p><p>Primary statistic: median of repetition bandwidths per process, then median across trials. * indicates incomplete collection; incomplete groups excluded from gain calculations.</p><p><a href="summary.csv">Summary CSV</a> · <a href="repetitions.csv">Repetitions CSV</a> · <a href="runs.csv">Commands & Status CSV</a> · <a href="results.json">Full JSON</a></p></section>
<section><h2>Continuous Pipeline: Layout and Mapping</h2><p>16 QPs, __PC__ copies × __ROUNDS__ rounds per QP; inflight=auto, includes WQE assembly. Unit: GB/s.</p>__PIPELINE__<p class="muted">Layout change compares 2×8 interleaved / 16×1 interleaved; mapping change compares 2×8 per-die / 2×8 interleaved. All use current binary and cannot substitute for historical B0 tests on original code, nor prove uniform DMA engine distribution.</p></section>
<section><h2>Single Batch: With / Without WQE Assembly</h2><p>16 QPs, 2×8 per-die, __BC__ copies × 1 round per QP, inflight=1. Two groups differ only in mode.</p>__MATCHED__<p class="muted">The without-assembly group still executes prepare; timing uses hggcEvent on the execution stream and excludes prepare and host sync overhead. Cache and launch states differ between sides, so the time difference shouldn't be treated as isolated assembly instruction cost. The bandwidth ratio here doesn't equal end-to-end application speedup including prepare. Continuous pipeline and single-batch groups don't directly calculate optimization gains.</p></section>
<section><h2>Cross-Process Variation</h2>__SPREAD__<p class="muted">Unit: GB/s. Raw repetitions preserved in repetitions.csv; even-count reps/trials average the middle two values. Commands alternate forward/reverse order by trial to reduce systematic order bias.</p></section>
<section><h2>Reproduction Environment</h2><pre>__ENV__</pre><p class="muted">Device processes checked before each run; this doesn't constitute cross-user exclusive locking. Loaded shared libraries and binary checksums are recorded; resume requires matching configuration, binary, dependencies, and script.</p></section>
<section><h2>Raw Commands and Logs</h2><input id="filter" placeholder="Filter by name, status, or command" aria-label="Filter logs"><div id="logs">__DETAILS__</div></section>
</main><script>document.getElementById('filter').oninput=function(){let q=this.value.toLowerCase();document.querySelectorAll('#logs details').forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(q))}</script></html>'''
    replacements=dict(TITLE=e(title),TRIALS=str(config['trials']),REPS=str(config['reps']),STATE=e(state['status']),PASS=str(passed),TOTAL=str(len(manifest['plan'])),REASON=e(state['reason']),PC=str(config['pipeline_copies']),ROUNDS=str(config['pipeline_rounds']),BC=str(config['batch_copies']),PIPELINE=pipeline,MATCHED=matched,SPREAD=spread,ENV=e(json.dumps(manifest['environment'],ensure_ascii=False,indent=2)),DETAILS=''.join(details))
    for key,value in replacements.items():body=body.replace('__'+key+'__',value)
    (out/'report.html').write_text(body,encoding='utf-8')


def capture(argv):
    p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=20)
    if p.returncode:raise RuntimeError('command failed: '+command_text(argv)+'\n'+p.stdout)
    return p.stdout


def idle_snapshot(smi):
    text=capture([str(smi)])
    if re.search(r'^\|\s*\d+\s+\S+\s+\S+\s+\d+\s+',text,re.M):
        raise RuntimeError('PPU has running processes; no benchmark launched')
    if 'No running processes found' not in text:
        raise RuntimeError('cannot confirm idle PPU state from ppu-smi output')
    return text


def main(argv=None):
    directory=Path(__file__).resolve().parents[1]/'tests/unit_test'
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--binary',type=Path,default=directory/'build/bench_oki')
    parser.add_argument('--sizes',type=sizes_arg,default=sizes_arg('4K,8K,16K,32K,64K,128K'),help='comma-separated command sizes')
    parser.add_argument('--trials',type=positive,default=3,help='independent processes per group/size')
    parser.add_argument('--reps',type=positive,default=3,help='timed repetitions per process')
    parser.add_argument('--pipeline-copies',type=positive,default=64)
    parser.add_argument('--pipeline-rounds',type=positive,default=1024)
    parser.add_argument('--batch-copies',type=positive,default=2048)
    parser.add_argument('--timeout',type=positive,default=120,help='seconds per benchmark process')
    parser.add_argument('--output',type=Path,help='new output directory (required with --resume)')
    parser.add_argument('--resume',action='store_true',help='resume PASS-checkpointed work, retain failed attempts')
    parser.add_argument('--smi',type=Path,help='path to ppu-smi; otherwise discover it')
    parser.add_argument('--dry-run',action='store_true',help='print exact commands without running or writing files')
    args=parser.parse_args(argv)
    if args.batch_copies>2048:parser.error('--batch-copies must be <=2048')
    if args.resume and not args.output:parser.error('--resume requires --output')
    config={k:getattr(args,k) for k in ['sizes','trials','reps','pipeline_copies','pipeline_rounds','batch_copies','timeout']}
    binary=args.binary.resolve()
    plan=plan_runs(binary,config)
    if args.dry_run:
        print('\n'.join(c['command'] for c in plan));return 0
    out=(args.output or directory/'build'/('bench_oki_compare_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if not args.resume and out.exists() and any(out.iterdir()):
        parser.error('output directory is not empty; use a new directory or --resume')
    smi=args.smi or shutil.which('ppu-smi') or directory.parents[1].parent/'PPU_SDK/ppu-smi/bin/ppu-smi'
    smi=Path(smi).resolve()
    if not binary.is_file() or not smi.is_file():parser.error('binary or ppu-smi not found; set --binary/--smi')
    help_text=capture([str(binary),'--help'])
    if not all(x in help_text for x in ['bandwidth-with-assembly','bandwidth-without-assembly','--mapping','--copies-per-round']):
        parser.error('binary does not support the named benchmark interface')
    ldd=capture(['ldd',str(binary)])
    if 'not found' in ldd:parser.error('unresolved shared libraries; load the SDK environment')
    library_paths=set(re.findall(r'=> (/\S+)',ldd))
    dependencies={p:sha(p) for p in sorted(library_paths) if Path(p).is_file()}
    fingerprint=dict(binary_sha256=sha(binary),dependencies=dependencies,runner_sha256=sha(__file__))
    if args.resume:
        manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if manifest['config']!=config or manifest['plan']!=plan or manifest['fingerprint']!=fingerprint:
            parser.error('resume configuration/binary/dependencies/script differ from original run')
        results=json.loads((out/'results.json').read_text(encoding='utf-8'))['runs'] if (out/'results.json').exists() else []
    else:
        out.mkdir(parents=True,exist_ok=True)
        (out/'logs').mkdir(exist_ok=True)
        environment=dict(host=platform.node(),platform=platform.platform(),python=sys.version,
                         binary=str(binary),smi=str(smi),fingerprint=fingerprint,ldd=ldd)
        source=directory/'src/bench_oki.cu'
        if source.is_file():environment['source_sha256']=sha(source)
        manifest=dict(schema=1,created_at=datetime.datetime.now().astimezone().isoformat(),config=config,
                      plan=plan,fingerprint=fingerprint,environment=environment)
        write_json(out/'manifest.json',manifest)
        shutil.copyfile(str(Path(__file__).resolve()),str(out/'runner.py'))
        results=[]
    done={r['id'] for r in results if r['status']=='PASS'}
    state=dict(status='RUNNING',reason='Collection in progress; completed data saved incrementally.')
    render_report(out,manifest,results,state)
    print('Output: '+str(out),flush=True)
    try:
        for c in plan:
            if c['id'] in done:continue
            if sha(binary)!=fingerprint['binary_sha256']:raise RuntimeError('binary changed during comparison')
            snapshot=idle_snapshot(smi)
            attempt=1+sum(r['id']==c['id'] for r in results)
            log='logs/{}_a{}.log'.format(c['id'],attempt)
            (out/'logs'/('{}_a{}.smi.txt'.format(c['id'],attempt))).write_text(snapshot,encoding='utf-8')
            r=dict(c,log=log,status='FAIL',error='',returncode=None)
            start=time.time()
            try:
                with (out/log).open('w',encoding='utf-8') as f:
                    p=subprocess.run(c['args'],stdout=f,stderr=subprocess.STDOUT,timeout=args.timeout)
                r['returncode']=p.returncode
                if p.returncode:raise ValueError('benchmark exit code '+str(p.returncode))
                text=(out/log).read_text(encoding='utf-8',errors='replace')
                r.update(validate_output(text,c));r['status']='PASS'
            except subprocess.TimeoutExpired:
                r['status']='TIMEOUT';r['error']='benchmark timed out; inspect device state before retrying'
            except KeyboardInterrupt:
                r['status']='INTERRUPTED';r['error']='interrupted by user'
            except (ValueError,KeyError,OSError) as exc:r['error']=str(exc)
            r['seconds']=round(time.time()-start,3)
            results.append(r)
            if r['status']=='PASS':done.add(c['id'])
            else:state=dict(status='STOPPED',reason=r['id']+': '+r['error'])
            render_report(out,manifest,results,state)
            print('{}/{} {} {}'.format(len(done),len(plan),r['status'],c['id']),flush=True)
            if r['status']!='PASS':return 2
        state=dict(status='COMPLETE',reason='All configurations passed data/CQ verification and output consistency checks.')
        render_report(out,manifest,results,state)
        print('Report: '+str(out/'report.html'),flush=True)
        return 0
    except (RuntimeError,OSError,subprocess.SubprocessError,KeyboardInterrupt) as exc:
        state=dict(status='STOPPED',reason=str(exc) or 'interrupted by user')
        render_report(out,manifest,results,state)
        print('Stopped: '+state['reason'],file=sys.stderr)
        return 2


if __name__=='__main__':
    try:sys.exit(main())
    except (RuntimeError,OSError,ValueError,subprocess.SubprocessError) as exc:
        print('Error: '+str(exc),file=sys.stderr);sys.exit(2)
