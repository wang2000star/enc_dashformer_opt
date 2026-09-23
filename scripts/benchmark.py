#!/usr/bin/env python3
"""Paired local engineering runs. No input rewriting or accuracy claims."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=Path('build/enc-dashformer'))
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--examples',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if a.repeats<1:p.error('--repeats must be positive')
    binary,data,examples,out=[x.resolve() for x in [a.binary,a.data_dir,a.examples,a.output_dir]]
    if not binary.is_file() or not examples.is_file():p.error('binary and examples must exist')
    model_files=[data/'dashformer_tokenizer.json',*sorted((data/'dashformer_model_parameters').glob('*.txt'))]
    if len(model_files)<2 or not model_files[0].is_file():p.error('tokenizer and model files are required')
    hashes={str(f.relative_to(data)):sha(f) for f in model_files}
    out.mkdir(parents=True,exist_ok=False)
    common=[str(binary),'--data-dir',str(data),'--examples',str(examples),'--log-p','31,31']
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
                  binary_sha256=sha(binary),input_sha256=sha(examples),model_sha256=hashes,
                  notes='Fresh encryption keys per run. Optimized path changes approximation; timings alone do not establish accuracy preservation.',runs=[])
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    subprocess.run(common+['--check-inputs'],check=True,stdout=subprocess.DEVNULL)
    modes={'baseline':['--baseline'], 'optimized':['--value-basis','--hoisted-rotations']}
    for repetition in range(a.repeats):
        order=list(modes) if repetition%2==0 else list(reversed(modes))
        for mode in order:
            dest=out/f'{repetition+1:02d}-{mode}';dest.mkdir()
            command=common+modes[mode]+['--output',str(dest)]
            start=time.monotonic()
            with (dest/'run.log').open('w') as log:
                proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            record=dict(repetition=repetition+1,mode=mode,command=command,wall_seconds=time.monotonic()-start,exit_code=proc.returncode)
            result=dest/'output.txt'
            if result.exists():record['output_sha256']=sha(result)
            manifest['runs'].append(record);save()
            if proc.returncode or not result.exists():raise SystemExit(f'Failed {mode}; inspect {dest}/run.log')
            print(f'{mode}: {record["wall_seconds"]:.3f}s',flush=True)
    manifest['median_wall_seconds']={mode:statistics.median(r['wall_seconds'] for r in manifest['runs'] if r['mode']==mode) for mode in modes}
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat();save()


if __name__=='__main__':main()
