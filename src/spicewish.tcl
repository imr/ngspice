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

		set l ""
		foreach p $args {
			set  l "$l\\\"[spicewish::viewer::evaluate $p]\\\" "
		}
		#eval spice::bltplot $l
		eval spice::bltplot $args

		return "${::spicewish::viewer::wNames}[expr $::spicewish::viewer::viewCnt -1]"

        }

	proc gui { } {
		spicewish::controls::create
	}

#        proc iplot {args} {
#		eval spicewish::viewer::iplot $args
#        }

	proc oplot {args} { 
		eval spicewish::viewer::oplot $args
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
			spicewish::nodeDialog::Update
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

	namespace export plot gui oplot source run halt load measDialog write2gnuplot
}


set ::spicewish::viewer::bltplotCnt -1

proc spice_gr_Plot { args } { 

	# time time s a0 voltage V 1
	set xName  [lindex $args 0]
	set xType  [lindex $args 1]
	set xUnits [lindex $args 2]
	set yName  [lindex $args 3]
	set yType  [lindex $args 4]
	set yUnits [lindex $args 5]
	set viewerCnt [lindex $args 6]	
		
	set plotNum [spicewish::vectors::get_plotNum [spice::getplot]]

	if {$spicewish::viewer::vecUpdate_flag} {
		# update exsisting plots
		
	} else {
		# add new plot
		if {$::spicewish::viewer::bltplotCnt == $viewerCnt } {
			eval spicewish::viewer::addplot [expr $::spicewish::viewer::viewCnt -1] $plotNum $args
		} else {
			eval spicewish::viewer::newplot $plotNum $args
			set ::spicewish::viewer::bltplotCnt $viewerCnt
		}
	}

}
