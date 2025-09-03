#!/bin/sh
set -xv

ngspice -b cd4007n_200402idvg.net
ngspice -b cdxy.net
ngspice -b Tbicmpd1.cir
ngspice -b Tbicmpd1xy.cir
ngspice -b Tbicmpu1.cir
ngspice -b Tbicmpu1xy.cir
