#!/bin/sh
# -*- mode: tcl -*- \
	exec wish -f "$0" ${1+"$@"}

package require BLT
load ../../../src/.libs/libspice.so
namespace import blt::*

wm title . "Vector Test script"
wm geometry . 800x600+40+40
pack propagate . false

stripchart .chart
pack .chart -side top -fill both -expand true
.chart axis configure x -title "Time"


# Create a vector (and call it $vector)
#vector create v1
spice::source example.cir
spice::bg run

after 1000

vector create a0
vector create b0
vector create a1
vector create b1
vector create stime
proc bltupdate {} {
    #puts [spice::spice_data]
    spice::spicetoblt a0 a0 
    spice::spicetoblt b0 b0
    spice::spicetoblt a1 a1
    spice::spicetoblt b1 b1
    spice::spicetoblt time stime
    #puts $spice::lastitercount
    after 100 bltupdate
}
bltupdate



.chart element create a0 -color red -xdata stime -ydata a0
.chart element create b0 -color blue -xdata stime -ydata b0
.chart element create a1 -color yellow -xdata stime -ydata a1
.chart element create b1 -color black -xdata stime -ydata b1

