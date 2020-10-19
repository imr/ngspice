namespace eval selectionwindow {
  variable selectionvalue
  variable selectionwindow
}

proc selectionwindow::selectfromlist { window title selectionlist args } {
 variable selectionvalue
 variable selectionwindow
 if { [winfo exists $window] } { 
   raise $window; 
   return [lindex $selectionlist 0] 
 }
 set selectionwindow $window
 toplevel $selectionwindow
 wm geometry $selectionwindow +200+200
 focus -force $selectionwindow
 wm title $selectionwindow $title
 set maxstrlength [expr [string length $title]+12]

 if { [llength $selectionlist]==0 } { destroy $selectionwindow; return {}  }

 foreach elem $selectionlist {
  if { [string length $elem]>$maxstrlength } {
   set  maxstrlength [string length $elem]
  } 
 }
  
 scrollbar $selectionwindow.scroll -command "$selectionwindow.listbox yview"
 eval "listbox $selectionwindow.listbox -yscroll \"$selectionwindow.scroll set\" \
	 -width $maxstrlength -height 10 -setgrid 1 $args"
 pack $selectionwindow.listbox $selectionwindow.scroll -side left -fill y -expand 1
 foreach elem $selectionlist {
   $selectionwindow.listbox insert end $elem
 }
 bind $selectionwindow.listbox <Double-1> {
  namespace eval selectionwindow {
    set selectionvalue [selection get]
    destroy $selectionwindow
  }
 }
 tkwait window $selectionwindow

 if { [info exists selectionvalue] } {
  return $selectionvalue 
 } else {
   if { [llength $selectionlist] != 0 } {
     return [lindex $selectionlist 0]
   } else {
    return ""
  }
 }
}

# puts [selectionwindow::selectfromlist .demo "Wähle Frucht" { Apfel Birne Zitrone dsfsdfdsfdsfdsfsdfds}]

