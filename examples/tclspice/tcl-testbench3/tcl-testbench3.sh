#!/bin/bash
# -*- mode: tcl -*- \
        exec wish -f "$0" ${1+"$@"}

package require BLT
load ../../../src/.libs/libspice.so
source differentiate.tcl
spice::codemodel ../../../src/xspice/icm/spice2poly/spice2poly.cm 
proc temperatures_calc {temp_inf temp_sup points} {
  set tstep [ expr " ( $temp_sup - $temp_inf ) / $points " ]
  set t $temp_inf
  set temperatures ""
  for { set i 0 } { $i < $points } { incr i } {
    set t [ expr { $t + $tstep } ]
    set temperatures "$temperatures $t"
  }
  return $temperatures
}

proc thermistance_calc { res B points } {
  set tzero 273.15
  set tref 25
  set thermistance ""
  foreach t $points {
    set res_temp [expr " $res * exp ( $B * ( 1 / ($tzero + $t) - 1 / ( $tzero + $tref ) ) ) " ]
    set thermistance "$thermistance $res_temp"
  }
  return $thermistance
}

proc tref_calc { points } {
  set tref ""
  foreach t $points {
    set tref " $tref [ expr " 6 * (2.275-0.005*($t - 20) ) - 9 " ] "
  }
  return $tref
}

proc iteration { t } {
  set tzero 273.15
  spice::alter r11=[ thermistance_calc 10000 3900 $t ]
  #spice::set temp = [ expr " $tzero + $t " ]
  spice::op
  spice::vectoblt vref_temp tref_tmp
  spice::destroy all
  return [ tref_tmp range 0 0 ]
}

proc cost_square { r10 r12 } {
  tref_blt length 0

  spice::alter r10=$r10
  spice::alter r12=$r12
  
  foreach point [ temperatures_blt range 0 [ expr " [temperatures_blt length ] - 1" ] ] {
    tref_blt append [ iteration $point ]
  }
  
  set result [ blt::vector expr " sum(( tref_blt - expected_blt )^2 )" ]
  puts "result square : r10 = $r10 r12 = $r12  gives  $result"

  return $result 
}

proc cost_sup { r10 r12 } {
  tref_blt length 0

  spice::alter r10=$r10
  spice::alter r12=$r12
  
  foreach point [ temperatures_blt range 0 [ expr " [temperatures_blt length ] - 1" ] ] {
    tref_blt append [ iteration $point ]
  }
  
  set result [ blt::vector expr " max(sqrt(( tref_blt - expected_blt )^2 ))" ]
  puts "result sup : $result"
  puts "result sup : r10 = $r10 r12 = $r12  gives  $result"

  return $result 
}

proc disp_curve { r10 r12 } {
.g configure -title "Valeurs optimales: R10 = $r10 R12 = $r12"
}

#
# Optimisation
#

blt::vector create tref_tmp
blt::vector create tref_blt
blt::vector create expected_blt
blt::vector create temperatures_blt
temperatures_blt append [ temperatures_calc -25 75 30 ] 
expected_blt append [ tref_calc [temperatures_blt range 0 [ expr " [ temperatures_blt length ] - 1" ] ] ]
blt::graph .g 
pack .g -side top -fill both -expand true
.g  element create real -pixels 4 -xdata temperatures_blt -ydata tref_blt
.g  element create expected -fill red -pixels 0 -dashes dot -xdata temperatures_blt -ydata expected_blt

spice::source FB14.cir
# point1 max iteration is the last argument
set r10r12 [ ::math::optimize::minimumSteepestDescent cost_square { 10000 10000 } 1.9 20 ]
puts "$r10r12 "
regexp {([0-9.]*) ([0-9.]*)} $r10r12 r10r12 r10 r12
puts "result square with : r10 = $r10 r12 = $r12 "
set r10r12 [ ::math::optimize::minimumSteepestDescent cost_sup " $r10 $r12 " 0.05 20 ]
puts "$r10r12 "
regexp {([0-9.]*) ([0-9.]*)} $r10r12 r10r12 r10 r12
puts "result sup with : r10 = $r10 r12 = $r12 "


#
# Results
#


spice::alter r10=$r10
spice::alter r12=$r12
foreach point [ temperatures_blt range 0 [ expr " [temperatures_blt length ] - 1" ] ] {
  tref_blt append [ iteration $point ]
}
disp_curve $r10 $r12 


