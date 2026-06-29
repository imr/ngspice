import os
import sys


testnum = 1


def writeit(cd, cmd, outd):
    global testnum
    pwd = os.getcwd()
    outfname = outd + '/testfile' + str(testnum) + '.sh'
    outf = open(outfname, 'w')
    testnum = testnum + 1
    outf.write('#!/bin/bash\n')
    outf.write('NGSPICE="ngspice -i "\n')
    p1 = 'VALGRIND="valgrind --leak-check=full --suppressions='
    p2 = p1 + pwd + '/ignore_shared_libs.supp"\n'
    outf.write(p2)
    outf.write(cd)
    if cmd.endswith('&\n'):
        outf.write(cmd[:-2] + '\n')
    else:
        outf.write(cmd)
    os.chmod(outfname, 0o777)
    outf.close()
    return 0


def main():
    infile = sys.argv[1]
    outdir = sys.argv[2]
    os.mkdir(outdir)
    inp = open(infile, 'r')
    for line in inp:
        if line.startswith('cd '):
            cdname = line
        elif line.startswith('$VALGRIND'):
            writeit(cdname, line, outdir)
    inp.close()
    return 0


if __name__ == '__main__':
    main()
