#!/bin/bash

ADMS=$(readlink -f ../../adms/adms-2.3.0-1500/admsXml)

top_srcdir=$(readlink -f $(dirname $0))
top_builddir=$(readlink -f $(dirname $0)/../build-wip-adms3--)


mkdir -p $top_builddir

cd $top_srcdir  &&  ./autogen.sh
cd $top_builddir &&  $top_srcdir/configure --enable-adms3
cd $top_builddir && make

function compile_dev() {
    cd $top_srcdir && make -C src/spicelib/devices/adms/$1/adms3va \
        top_srcdir="$top_srcdir" \
        top_builddir="$top_builddir" \
        ADMS="$ADMS" \
        $2
}


for dev in hicum0 ekv myvares ; do
    compile_dev "$dev" clean
    compile_dev "$dev"
done


echo "expect: Model issue on line 0 : .model x1:hic0_full hicum0 is=1.3525e-18 vef=8.0 iqf=3.0 ..."
echo "expect: unrecognized parameter (hicum0) - ignored"
echo "expect: 0	2.000000e-01	2.996258e-12	-3.00000e+00"
echo "expect: 1	7.000000e-01	-8.84510e-07	-3.00000e+00"	


cd $top_srcdir && \
  LD_LIBRARY_PATH=src/spicelib/devices/adms/hicum0/adms3va:src/spicelib/devices/adms/myvares/adms3va \
    $top_builddir/src/ngspice -p < hic0+.cir


# (compile "bash wip-adms3.sh")
