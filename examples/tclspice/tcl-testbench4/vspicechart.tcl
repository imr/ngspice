############ spice chart program                      ###########
############ programmer: stephan thiel                ###########
############             thiel@mikro.ee.tu-berlin.de  ###########   
############             (c) 2008 Berlin, Germany      ###########
############             Don't trust any version      ###########
############             before 1.0                   ###########  


package require BLT
load "../../../src/.libs/libspice.so"

source selectfromlist.tcl
source bltGraph.tcl

namespace import blt::*

wm title . "vspicechart 0.01"
wm geometry . 800x450+40+40
pack propagate . false

set globals(colors) { red green blue orange yellow white gray lightblue pink darkblue \
              lightred lightgray darkgray darkblue darkgreen darkred violet salmon \
              gray100 gold SeaGreen RoyalBlue RosyBrown orchid MintCream magenta LimeGreen \
              gray33 DeepSkyBlue DarkGoldenrod chocolate gray77 aquamarine brown coral \
              DarkOliveGreen DarkOrange DarkSlateGray gray99 HotPink IndianRed LemonChiffon \
              LightSteelBlue PaleGreen peru sienna seashell SpringGreen tomato wheat WhiteSmoke} 


proc replacechar { str pat pat1} {
  set erg "" 
  for { set i 0 } { $i < [string length $str] } {incr i 1 } {
   if { [ string index $str  $i ] == $pat } {
     append erg $pat1
   }  else {
     append erg [string index $str $i ]
   }
  }
  return $erg
}


proc realtostr { r } {
  set b [ expr abs($r) ]  
  set mul 1e-18
  set prefix a
   if { $b > 9.9999999999e-16 } { 
        set mul 1e15 
        set prefix f
 }
  if { $b > 9.9999999999e-13 } { 
        set mul 1e12 
        set prefix p
 }
  if { $b > 9.9999999999e-10 } { 
        set mul 1e9 
        set prefix n
 }
  if { $b > 9.9999999999e-7 } { 
        set mul 1e6 
        set prefix u
 }
  if { $b > 9.9999999999e-4 } { 
        set mul 1e3 
        set prefix m
 }

  if { $b > 0.999999999999999 } { 
        set mul 1 
        set prefix ""
 }

  if { $b > 999 } { 
        set mul 1e-3 
        set prefix K
 }

  if { $b > 9.999999999e5 } { 
        set mul 1e-6 
        set prefix MEG
 }
  if { $b > 9.9999999999e8 } { 
        set mul 1e-9 
        set prefix G
 }
  if { $b > 9.99999999999e11 } { 
        set mul 1e-12 
        set prefix T
 }
  set str [ format "%1.8g$prefix" [expr $r*$mul] ]
 if { $str=="0a" } { set str "0" }
  return $str
}

proc realtostr1 { elem r } {
  scan $r "%f" erg 
 return [ realtostr $erg ]
}

set globals(signals) {};

proc readconfigfile { } {
    global globals
    global const
    if { [file exists $globals(CONFIGFILE)] } {
        set fid [open $globals(CONFIGFILE) r]
	 while { ![eof $fid] } {
           gets $fid tempstring
             if { [string first "PACK-PATH=" $tempstring]==0 } {
               scan $tempstring "PACK-PATH=%s" globals(PACK-PATH)
	     }
             if { [string first "SIMULATOR=" $tempstring]==0 } {
               scan $tempstring "SIMULATOR=%s" globals(SIMULATOR)
	     }
         } 
        close $fid
     } else {
       set globals(PACK-PATH) ""
       set globals(SIMULATOR) "INTERNAL"
     }
}

proc select_vector { } {
   global globals
   set thissignals [spice::spice_data]
   set signals {}
   foreach sig $thissignals {
       if { [lindex $sig 0] != "time" } {
         lappend signals [lindex $sig 0]
       }
   }  
   set selectedsignal [selectionwindow::selectfromlist .select "Select Signal" $signals ]
   if { ( [string trim $selectedsignal] != "") && ([lsearch -exact $globals(signals) $selectedsignal] == -1) } {
      eval "$globals(LSELECTEDSIGNALS) insert end $selectedsignal"
      vector create [replacechar $selectedsignal "\#" "_"]
   }
}


proc start_new_sel { } {
  global globals

    set elemlist [ eval "$globals(chart0) element show" ]  
    for { set j 0 } {$j < [llength $elemlist] } {incr j 1} {
       $globals(chart0)   element delete [lindex $elemlist $j ]   
    }

  set i 0
  foreach sig $globals(signals) {
    set nsig [replacechar $sig "\#" "_"]
    vector create $nsig
    spice::spicetoblt $sig $nsig
  
    $globals(chart0) element create $sig -color  [lindex $globals(colors) $i]  -xdata stime -ydata $nsig -symbol none
    incr i 1
  } 
}


proc delete_selected { } {
   global globals
   set elem [$globals(LSELECTEDSIGNALS) curselection]
    if { $elem != "" } {
       $globals(LSELECTEDSIGNALS) delete  $elem
    }
}



set filename [ lindex $argv 0] 

if { [file exists $filename ] } {
   spice::source $filename
   spice::bg run

after 1000



frame .f1
pack .f1 -side left -expand true -fill both

listbox .f1.blistbox -listvariable globals(signals)
pack .f1.blistbox -side top -fill both -expand true

set globals(LSELECTEDSIGNALS) .f1.blistbox


button .f1.baddvec -text "Select Vector" -command "select_vector"
pack .f1.baddvec -side top -fill x -expand true

button .f1.bdelvec -text "Delete Vector"  -command  "delete_selected"
pack .f1.bdelvec -side top -fill x -expand true

button .f1.bstartsel -text "Start with new selection" -command start_new_sel
pack .f1.bstartsel -side top -fill x -expand true


button .f1.simstop -text "Simulation Stop" -command "spice::stop"
pack .f1.simstop -side top -fill x -expand true

button .f1.simresume -text "Simulation Resume" -command "spice::bg resume"
pack .f1.simresume -side top -fill x -expand true


button .f1.bexit -text "Exit" -command "exit"
pack .f1.bexit -side top -fill x -expand true


frame .f2
pack .f2 -side left -expand true -fill both

stripchart .f2.chart
pack .f2.chart -side top -fill both -expand true 
.f2.chart axis configure x -title "Time in s"


.f2.chart grid configure -hide no 
Blt_ZoomStack .f2.chart
Blt_Crosshairs .f2.chart
Blt_ClosestPoint .f2.chart
Blt_PrintKey .f2.chart
Blt_ActiveLegend .f2.chart
.f2.chart crosshairs configure -color lightblue 

.f2.chart axis configure x -command realtostr1
.f2.chart axis configure y -command realtostr1

set globals(chart0) .f2.chart



  vector create stime


  proc bltupdate {} {
      global globals
   
      spice::spicetoblt time stime
      foreach sig $globals(signals) {
         set nsig [replacechar $sig "\#" "_"]
         spice::spicetoblt $sig $nsig
      } 
    
      after 100 bltupdate
  }
  bltupdate

} else {
    exit;
}
