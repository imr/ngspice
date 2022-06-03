# tcl script for gtkwave: show vcd file data created by ngspice
# used by counter.cir
set nfacs [ gtkwave::getNumFacs ]

for {set i 0} {$i < $nfacs } {incr i} {
    set facname [ gtkwave::getFacName $i ]
    set num_added [ gtkwave::addSignalsFromList $facname ]
}

gtkwave::/Edit/UnHighlight_All
gtkwave::/Time/Zoom/Zoom_Full
