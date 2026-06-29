#!/bin/bash
NGSPICE="ngspice -i "
VALGRIND="valgrind --leak-check=full --suppressions=/home/holger/Software/paranoia_parallel/ignore_shared_libs.supp"
cd examples/control_structs
$VALGRIND --log-file=../../s-param.vlog $NGSPICE -o ../../s-param.log s-param.cir
