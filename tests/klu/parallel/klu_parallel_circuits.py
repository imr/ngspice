"""Compare complete DC/transient/AC runs with the optional KLU fast path."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from circuit_fixtures import cases,compare

def run(binary,folder,body,vectors,analysis,threads,enabled):
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'test.cir').write_text('KLU optimized circuit regression\n'+body+
        '\n.option klu reltol=1e-5 abstol=1e-12 vntol=1e-8\n.save '+vectors+
        '\n.control\n'+('set klu_parallel\n' if enabled else 'unset klu_parallel\n')+
        f'set num_threads={threads}\nset numdgt=17\nset wr_vecnames\nset wr_singlescale\n'+analysis+
        '\nwrdata wave.dat '+vectors+'\nrusage all\nquit\n.endc\n.end\n')
    with (folder/'run.log').open('w') as log:
        subprocess.run([str(binary),'-n','-b','test.cir'],cwd=folder,
            env=dict(os.environ,OMP_WAIT_POLICY='passive'),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    log=(folder/'run.log').read_text()
    assert re.search(r'ngspice-\S+ done',log) and not re.search(r'Error:|doAnalyses:|analysis aborted',log,re.I)
    return {key:float(re.search(re.escape(key)+r'\s*=\s*([0-9.eE+-]+)',log)[1]) for key in ['Accepted timepoints','Rejected timepoints']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);a=p.parse_args()
    fixtures=cases(); side=40
    mesh=['Vdd in 0 dc .5 ac 1 pulse(.5 1 1n 100p 100p 4n 10n)','Rin in n0 10']
    for row in range(side):
        for col in range(side):
            n=row*side+col;mesh += [f'C{n} n{n} 0 20f',f'Rg{n} n{n} 0 10meg']
            if col+1<side:mesh.append(f'Rh{n} n{n} n{n+1} 100')
            if row+1<side:mesh.append(f'Rv{n} n{n} n{n+side} 100')
    fixtures['switch-large']=('\n'.join(mesh),'v(in) v(n0) v(n1599) i(vdd)',
        'op\ntran 50p 2n 0 50p\nwrdata before.dat v(n1599)\nac dec 5 1meg 10gig\nwrdata ac.dat v(n1599)\ntran 50p 2n 0 50p')
    results=[]
    for name,(body,vectors,analysis) in fixtures.items():
        base=a.output/name/'serial'; expected=run((a.baseline or a.binary).resolve(),base,body,vectors,analysis,1,False)
        for threads in [1,8,16,32]:
            out=a.output/name/f'optimized-{threads}';actual=run(a.binary.resolve(),out,body,vectors,analysis,threads,True)
            assert actual==expected,(name,threads,actual,expected)
            row={'case':name,'threads':threads,'timepoints':actual,'wave':compare(base/'wave.dat',out/'wave.dat')}
            assert row['wave']['byte_identical'],(name,threads)
            for file in ['before.dat','ac.dat']:
                if (base/file).exists():
                    row[file]=compare(base/file,out/file)
                    assert row[file]['byte_identical'],(name,threads,file)
            if name=='switch-large':assert 'KLU optimized summary:' in (out/'run.log').read_text()
            results.append(row)
    (a.output/'result.json').write_text(json.dumps({'passed':True,'runs':results},indent=2)+'\n')
    print(f'PASS {len(results)} configurations: byte-identical saved waveforms, matching timepoint decisions')

if __name__=='__main__':main()
