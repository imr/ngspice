
namespace eval readline {

	proc completer { word start end line } {

		if {[info procs ::tclreadline::TryFromList] == ""} {
			::tclreadline::ScriptCompleter  $word $start $end $line
		}

		set command [lindex $line 0]
		
		switch $command {
			"spicewish\:\:source" - "spice\:\:source" - "source" {
				return [::tclreadline::TryFromList $word [glob *]]
			}
		        "spicewish\:\:load" - "spice\:\:load" - "load" {
                                return [::tclreadline::TryFromList $word [glob *]]
                        }
			"spicewish\:\:plot" - "plot" {
				return [::tclreadline::TryFromList $word [spice::plot_variables 0]]
			}
			"spicewish\:\:iplot" - "iplot" {
				return [::tclreadline::TryFromList $word [spice::plot_variables 0]]
			}
			"spicewish\:\:oplot" - "oplot" {
	
				if {[lsearch "[spicewish::vectors::avaliable_plotNames]" [lindex $line 1]] != -1} {
					set plotNum [spicewish::vectors::get_plotNum [lindex $line 1]]
                                        return [::tclreadline::TryFromList $word [spice::plot_variables $plotNum]]
				} else {
					return [::tclreadline::TryFromList $word [spicewish::vectors::avaliable_plotNames]]
				}

                        }
			"spicewish\:\:measDialog" - "measDialog" {
				return [::tclreadline::TryFromList $word [spice::plot_variables 0]]
			}
                        "spicewish\:\:write2gnuplot" - "write2gnuplot" {
                                return [::tclreadline::TryFromList $word [spice::plot_variables 0]]
                        }

			"spicewish\:\:run" - "run" {
				return ""
			}
			"spicewish\:\:halt" - "halt" {
                                return ""
                        }

		}
	
		::tclreadline::ScriptCompleter  $word $start $end $line 
	}

	::tclreadline::readline customcompleter "spicewish::readline::completer"
}
