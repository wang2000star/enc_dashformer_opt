#!/usr/bin/env python3
"""Render the recorded historical comparison; requires matplotlib."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'dashformer-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'benchmarks/recorded/comparison.json'


def main():
    data=json.loads(DATA.read_text());historical=data['historical_163'];a,b=historical['records']
    for row in historical['records']:
        for name,digest in row['sources'].items():
            assert hashlib.sha256((DATA.parent/name).read_bytes()).hexdigest()==digest,name
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    specs=[('wall_seconds','End-to-end runtime','Seconds',1),('peak_rss_kib','Peak resident memory','GiB',1024**2)]
    for ax,(field,title,unit,divisor) in zip(axes,specs):
        vals=[row[field]/divisor for row in [a,b]]
        bars=ax.bar(['Full-rank\nreference','Optimised\ncandidate'],vals,color=['#64748b','#087e8b'],width=.6)
        ax.set_title(title+' — lower is better',fontsize=12)
        ax.set_ylabel(unit);ax.set_ylim(0,max(vals)*1.23)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=5)
        ax.yaxis.grid(True,alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Historical internal comparison · 163 development examples',fontsize=14,y=.97)
    fig.text(.5,.08,'8 September 2026 · one run per mode · same experimental executable\nNot a direct upstream-release benchmark; no statistical significance established.',ha='center',fontsize=10,color='#475569')
    fig.tight_layout(rect=(0,.16,1,.94))
    dest=ROOT/'docs/images';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'historical-comparison.png',dpi=180)
    fig.savefig(dest/'historical-comparison.svg')
    print(f'Runtime ratio: {a["wall_seconds"]/b["wall_seconds"]:.3f}x; memory reduction: {100*(1-b["peak_rss_kib"]/a["peak_rss_kib"]):.2f}%')


if __name__=='__main__':main()
