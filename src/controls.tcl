
namespace eval controls {

	variable wName ".controls"	

	proc create { } {
		variable wName
		
		if {[winfo exists $wName]} {
			wm deiconify $wName
			raise $wName
			return
		}

		set w [toplevel $wName]

		pack [button $w.bRun -text "Run" -command "spicewish::run" -width 8] -fill x
		pack [button $w.bHalt -text "Halt" -command "spicewish::halt"] -fill x
		pack [button $w.bPlot -text "Plot" -command "spicewish::plot"] -fill x
		pack [button $w.bExit -text "Exit" -command "[namespace current]::Exit"] -fill x
		pack [label $w.lFake -height 1] -fill x
	}

	proc Exit { } {
		set widget_list [winfo children .]

		# test if tk console ( .tkcon ) is being used 
		if [info exists ::CAD_HOME] {

			# kill only the spicewish based windows
			foreach winname $widget_list  {
				# viewer windows
				if {[regexp {.tclspice[0-9]+} $winname]} {
					wm withdraw $winname
		    		} 
			    	if { $winname == ".controls" } { 
					wm withdraw ".controls" 
				}  

			 }
    		} else { exit }
	}
}
