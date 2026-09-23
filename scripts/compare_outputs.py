#!/usr/bin/env python3
"""Compare decrypted score files; agreement is not predictive accuracy."""
import argparse
import json
import math
from pathlib import Path


def read_scores(path):
    rows=[]
    for number,line in enumerate(Path(path).read_text().splitlines(),1):
        row=[float(v) for v in line.split()]
        if not row or any(not math.isfinite(v) for v in row):
            raise ValueError(f'{path}: row {number} is empty or non-finite')
        if rows and len(row)!=len(rows[0]):
            raise ValueError(f'{path}: inconsistent score dimensions at row {number}')
        rows.append(row)
    if not rows:raise ValueError(f'{path}: empty score file')
    return rows


def compare(baseline, candidate):
    a,b=read_scores(baseline),read_scores(candidate)
    if len(a)!=len(b) or len(a[0])!=len(b[0]):
        raise ValueError('output shapes differ')
    errors=[abs(x-y) for ar,br in zip(a,b) for x,y in zip(ar,br)]
    # hypot avoids intermediate squared-error overflow on large finite scores.
    rmse=math.hypot(*errors)/math.sqrt(len(errors))
    maximum=max(errors)
    if not math.isfinite(maximum) or not math.isfinite(rmse):
        raise ValueError('score differences exceed finite floating-point range')
    agreement=sum(max(range(len(ar)),key=ar.__getitem__)==max(range(len(br)),key=br.__getitem__) for ar,br in zip(a,b))/len(a)
    return dict(rows=len(a),classes=len(a[0]),max_absolute_difference=maximum,
                root_mean_square_difference=rmse,top1_agreement=agreement,
                interpretation='Pairwise score differences and agreement only; no labels or predictive accuracy evaluated.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args()
    try:result=compare(a.baseline,a.candidate)
    except (ValueError,OSError) as error:p.error(str(error))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
