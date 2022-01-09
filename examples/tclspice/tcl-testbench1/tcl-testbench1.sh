#!/bin/bash
# -*- mode: tcl -*- \
        exec wish -f "$0" ${1+"$@"}

# old name:  analyse-20070504-0.tcl
package require BLT
load ../../../src/.libs/libspice.so

# Test of virtual capacitore circuit
# Vary the control voltage and log the resulting capacitance
spice::source "testCapa.cir" 

set n 30
set dv 0.2
set vmax [expr $dv/2]
set vmin [expr -1 * $dv/2]
set pas [expr $dv/ $n]

blt::vector create Ctmp 
blt::vector create Cim 
blt::vector create check 

blt::vector create Vcmd 
blt::graph .cimvd -title "Cim = f(Vd)"
blt::graph .checkvd -title "Rim = f(Vd)"

blt::vector create Iex
blt::vector create freq
blt::graph .freqanal -title "Analyse frequentielle"

set v [expr {$vmin + $n * $pas / 4}]
spice::alter vd = $v
spice::op
spice::ac dec 10 100 100k
spice::vectoblt {Vex#branch} Iex
spice::vectoblt {frequency} freq
pack .freqanal
.freqanal element create line1 -xdata freq -ydata Iex

for {set i 0} {[expr $n - $i]} {incr i } {
set v [expr {$vmin + $i * $pas}]
spice::alter vd = $v
spice::op
spice::ac dec 10 100 100k


spice::let Cim = real(mean(Vex#branch/(2*Pi*i*frequency*(V(5)-V(6)))))
spice::vectoblt Cim Ctmp
Cim append $Ctmp(0:end)
spice::let err = real(mean(sqrt((Vex#branch-(2*Pi*i*frequency*Cim*V(5)-V(6)))^2)))
spice::vectoblt err Ctmp
check append $Ctmp(0:end)
Vcmd append $v

}

pack .cimvd
.cimvd element create line1 -xdata Vcmd -ydata Cim
pack .checkvd
.checkvd element create line1 -xdata Vcmd -ydata check


