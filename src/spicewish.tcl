#package require spice

namespace eval spicewish {

	set dir [file join $::spice_library spice]
	variable stepCallBack_list ""

	source [file join $dir viewer.tcl ]
	source [file join $dir controls.tcl ]
	source [file join $dir engUnits.tcl ]
	source [file join $dir nodeDialog.tcl ]
        source [file join $dir readline.tcl ]
	source [file join $dir measure.tcl ]
	source [file join $dir gnuplot.tcl]

        proc plot {args} {
		
		# no plots passed load selection dialog
		if {$args == ""} {
			spicewish::nodeDialog::create
			return
		}
		eval spicewish::viewer::plot $args
        }

	proc gui { } {
		spicewish::controls::create
	}

        proc iplot {args} {
		eval spicewish::viewer::iplot $args
        }

	proc source { fileName } {
		if {![file exists $fileName]} {
			puts "$fileName: No such file"
			return
		}

		if {[string tolower [file extension $fileName]] == ".tcl" } {
			uplevel 1 ::tclSource $fileName			
			return
		}
		 
		spice::source $fileName
	}
	
	proc load { fileName } {
		if {[string tolower [file extension $fileName]] == ".raw" } {
                        spice::load $fileName
                        return
                }

		uplevel 1 ::tclLoad $fileName
	}

        proc run { } {
		if {$::spice::steps_completed == 0} {
			spice::bg run
		} else {
			spice::bg resume
		}		
        }

        proc halt { } {
		spice::halt
		spicewish::viewer::Update
        }

	proc measDialog { node } {
		spicewish::meas_dialog::add $node
	}
	
	proc write2gnuplot { args } {
		eval spicewish::gnuplot::plot $args
	}

	proc stepCallBack { } {
		variable stepCallBack_list
		
		foreach callBack $stepCallBack_list {
			eval $callBack
		}
	}

	spice::registerStepCallback spicewish::stepCallBack 1 1

	namespace export plot gui iplot source run halt load measDialog write2gnuplot
}
