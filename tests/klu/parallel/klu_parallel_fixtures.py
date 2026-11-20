"""Deterministic nonsymmetric and permuted BTF fixtures for refactor tests."""
from pathlib import Path
import random
import sys

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
for name,sizes in [('grid',[28]),('btf',[22,18,12])]:
    for scaled in [False,True]:
        n=sum(side*side for side in sizes);entries=[];offset=0
        for side in sizes:
            for r in range(side*side):
                row=offset+r;scale=10**(row%9-4) if scaled else 1.
                entries.append((row,row,8*scale))
                for d in [-side,side,-1,1]:
                    c=r+d
                    if c<0 or c>=side*side or (abs(d)==1 and r//side!=c//side):continue
                    entries.append((row,offset+c,(-1-.05*(row%3))*scale))
            if offset:
                for j in range(side*side):entries.append((j%offset,offset+j,.01))
            offset+=side*side
        permutation=list(range(n));random.Random(144).shuffle(permutation)
        entries=sorted(((permutation[r],permutation[c],v) for r,c,v in entries),key=lambda x:(x[1],x[0]))
        with (root/(name+('-scaled' if scaled else '-plain')+'.mtx')).open('w') as file:
            file.write('%%MatrixMarket matrix coordinate real general\n'+f'{n} {n} {len(entries)}\n')
            for r,c,value in entries:file.write(f'{r+1} {c+1} {value:.17g}\n')
