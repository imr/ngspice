#!/bin/bash

ngspice -b 7490a.cir
ngspice -b 74f524.cir
ngspice -b behav-283-1.cir
ngspice -b behav-283.cir
ngspice -b behav-568.cir
ngspice -b behav-tristate-pulse.cir
ngspice -b behav-tristate.cir
ngspice -b counter.cir
ngspice -b decoder.cir
ngspice -b encoder.cir
ngspice -b ex283.cir
ngspice -b ex4.cir
ngspice -b ex5.cir
ngspice -b xspice_c4.cir
