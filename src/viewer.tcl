
namespace eval viewer {

	variable wNames ".tclspice"
	variable viewCnt 1

	variable vecUpdate_flag 0

	variable viewTraces
	variable viewTraceScale
	variable view_w
	variable view_traces
	variable view_type
	variable view_plotName
	variable view_traceCnt
	variable view_Xscale

	variable meas_dx
	variable meas_dy
	variable meas_x1
	variable meas_y1
	variable meas_x2
	variable meas_y2

	proc create { plotNum } {
		variable wNames
		variable viewCnt
		variable view_w

		set w [toplevel "${wNames}${viewCnt}"]
		set plotName [spicewish::vectors::get_plotTypeName $plotNum]

		wm title $w "\[$w\] - $plotName - [spice::plot_title $plotNum]"
		
		# measurments
    		frame $w.meas

		blt::graph $w.g -width 1000 -height 250
		eval "$w.g axis configure x -scrollcommand {$w.x  set}"
   		eval "$w.g axis configure y -scrollcommand {$w.y  set}"
		eval "scrollbar $w.x -orient horizontal -command {$w.g axis view x} -width 9"
		eval "scrollbar $w.y -command {$w.g axis view y} -width 9"
		
		# x-axis title
		$w.g axis configure x -title [spicewish::vectors::defaultScale $plotNum]

		create_measurments $w

		# table 
		blt::table $w \
			0,0 $w.meas -fill x -cspan 3 \
			1,1 $w.g -fill both -rspan 2 \
			1,2 $w.y -fill y -rspan 2 \
			3,1 $w.x -fill x

		blt::table configure $w r0 r2 c0 c2 r3 -resize none	

		# graph settings
		Blt_Crosshairs $w.g
		if {0} {
			Blt_ZoomStack $w.g 
		} else {
			blt::ZoomStack $w.g "ButtonPress-3" "ButtonPress-2"
			blt::ZoomStack $w.g "ButtonPress-3" "Shift-ButtonPress-3"
		}
		
		$w.g grid on
		$w.g axis configure x -command { spicewish::viewer::axis_callback } -subdivisions 2

		set view_w($viewCnt) "$w"

		# freq/risetime measurments
		spicewish::meas_markers::bindings $w

		eval "wm protocol $w WM_DELETE_WINDOW {spicewish::viewer::Destroy $w}"

		bind $w <KeyPress-u> { spicewish::viewer::Update }
		
		incr viewCnt
		return [expr $viewCnt -1]
	}

	proc create_measurments { w } {
		variable meas_dx
		variable meas_dy
		variable meas_x1
		variable meas_y1
		variable meas_x2
		variable meas_y2
		
		set meas_dx($w.g) 0.00000000000000
		set meas_dy($w.g) 0.00000000000000
  		set meas_x1($w.g) 0.00000000000000
		set meas_y1($w.g) 0.00000000000000
		set meas_x2($w.g) 0.00000000000000
		set meas_y2($w.g) 0.00000000000000
    
		set w_meas "$w.meas"

		# measurements 
		pack [label $w_meas.dyv -textvariable  spicewish::viewer::meas_dy($w.g) -background "lightgrey" -width 16] -side right
		pack [label $w_meas.dy -text "dy :" -background "lightgrey" ] -side right
		bind  $w_meas.dyv <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "dy :  $::spicewish::viewer::meas_dy($w.g)"}
   	
		pack [label $w_meas.dxv -textvariable spicewish::viewer::meas_dx($w.g) -background "grey" -width 16] -side right
		pack [label $w_meas.dx -text "dx : " -background "grey"] -side right
		bind  $w_meas.dxv <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "dx :  $::spicewish::viewer::meas_dx($w.g)"}
		bind  $w_meas.dxv <ButtonPress-2> { 
			regexp  {(.[0-9A-z]+)} %W w
			if { $::spicewish::viewer::meas_dx($w.g) != 0 } { 
				puts "freq : [spicewish::eng::float_eng [expr 1 / $::spicewish::viewer::meas_dx($w.g)]] Hz" 
			} 
		}
 
		pack [ label $w_meas.y2v -textvariable  spicewish::viewer::meas_y2($w.g)  -background "lightgrey" -width 16  ] -side right
		pack [ label $w_meas.y2 -text "y2 :"  -background "lightgrey"] -side right
		bind  $w_meas.y2v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "y2 :  $::spicewish::viewer::meas_y2($w.g)"}
    
		pack [ label $w_meas.y1v -textvariable  spicewish::viewer::meas_y1($w.g)  -background "grey"  -width 16] -side right
		pack [ label $w_meas.y1 -text "y1 :" -background "grey"  ] -side right
		bind  $w_meas.y1v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "y1 :  $::spicewish::viewer::meas_y1($w.g)"}
    
		pack [ label $w_meas.x2v -textvariable  spicewish::viewer::meas_x2($w.g)  -background "lightgrey"  -width 16 ] -side right
		pack [ label $w_meas.x2 -text "x2 :" -background "lightgrey"  ] -side right
		bind  $w_meas.x2v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "x2 :  $::spicewish::viewer::meas_x2($w.g)"}
    
		pack [ label $w_meas.x1v -textvariable  spicewish::viewer::meas_x1($w.g) -background "grey" -width 16  ] -side right
		pack [ label $w_meas.x1 -text "x1 :" -background "grey"  ] -side right
		bind  $w_meas.x1v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "x1 :  $::spicewish::viewer::meas_x1($w.g)"}

		if {1} { 
			catch {::spicewish::motionMeasure::create $w.g}

		} else {

	 		# cursor motion bindings 
			bind $w.g <Motion> {
				if { $::zoomInfo(%W,B,x) != "" } { 
		   			set ::spicewish::viewer::meas_x2(%W)  [%W xaxis invtransform $::zoomInfo(%W,B,x)]
					set ::spicewish::viewer::meas_y2(%W)  [%W yaxis invtransform $::zoomInfo(%W,B,y)]
					set ::spicewish::viewer::meas_x1(%W)  [%W xaxis invtransform $::zoomInfo(%W,A,x)]
					set ::spicewish::viewer::meas_y1(%W)  [%W yaxis invtransform $::zoomInfo(%W,A,y)]
					set ::spicewish::viewer::meas_dx(%W)  [expr ($::spicewish::viewer::meas_x2(%W) - $::spicewish::viewer::meas_x1(%W))]
					set ::spicewish::viewer::meas_dy(%W)  [expr ($::spicewish::viewer::meas_y2(%W) - $::spicewish::viewer::meas_y1(%W))]
				}
	    		}  
		}


	}

	proc names { } {
		variable view_w
	
		return [array names view_w]
	}

	proc plot { args } { 

		if {$args == ""} {return}
		spice::bltplot $args
	}

	proc oplot { plotTypeName args } {
		
		if {$args == ""} {return}

		set currentPlotType [spice::getplot]

		spice::setplot $plotTypeName

		set l ""
                foreach p $args {
                        set  l "$l\\\"[spicewish::viewer::evaluate $p]\\\" "
                }
                eval spice::bltplot $l

		spice::setplot $currentPlotType
	}

	proc newplot { plotNum args } {

		variable view_w
		variable view_traces
		variable view_type
		variable view_plotName
		variable view_traceCnt
	
		if {$args == ""} {return}
	
		# create a new viewer
		set viewNum [create $plotNum]
		set w $view_w($viewNum)
                set view_traces($viewNum) ""
                set view_type($viewNum) "plot"
                set view_plotName($viewNum) [spicewish::vectors::get_plotTypeName $plotNum]
		set view_traceCnt 0

		eval "addplot $viewNum $plotNum $args"
	}
	
	proc addplot { viewNum plotNum args } { 
		
		variable view_w
		variable view_traces
		variable view_type
		variable view_plotName
		variable view_traceCnt
		variable view_Xscale
		
		if {$args == ""} {return}

		set w $view_w($viewNum)

		set xName  [lindex $args 0]
	        set xType  [lindex $args 1]
	        set xUnits [lindex $args 2]
	        set yName  [lindex $args 3]
	        set yType  [lindex $args 4]
	        set yUnits [lindex $args 5]
	
		set xVarName [spicewish::vectors::create_vecName $plotNum $xName]
		set yVarName [spicewish::vectors::create_vecName $plotNum $yName]
	
		eval $xVarName set ::spice::X_Data
		eval $yVarName set ::spice::Y_Data
	
		lappend view_traces($viewNum) $yName
		set view_Xscale($viewNum) $xName

                $w.g element create "$yName" \
                      	-xdata $xVarName \
                        -ydata $yVarName \
                       	-symbol "circle" \
                      	-linewidth 2 \
                      	-pixels 3 \
                       	-color [trace_colour $view_traceCnt]

		$w.g axis configure x -title $xName

          	incr view_traceCnt
	#	return $w
	}
	
	proc iplot  { args } {
		variable view_type

		eval "set view_w [plot $args]"

		# register callBack
		if {[lsearch $spicewish::stepCallBack_list "spicewish::viewer::stepCallBack"] == -1} {
			lappend spicewish::stepCallBack_list "spicewish::viewer::stepCallBack"
		}

		set viewNum [get_index $view_w]
		set view_type($viewNum) "iplot"
	}

	proc get_index { name } {
		variable view_w

		foreach viewNum [array names view_w] {
			if { $name == $view_w($viewNum) } {
				return $viewNum
			}
		}
		return -1
	}

	proc axis_callback { widget value } {
   
		return [ spicewish::eng::float_eng $value]
	} 

	proc trace_colour { index } {

		set def_colours "red blue orange green magenta brown"

		if {$index < [llength $def_colours]} {
			return [lindex $def_colours $index]
		} else {
			# creates a random colour 
			set randred [format "%03x" [expr {int (rand() * 4095)}]]
			set randgreen [format "%03x" [expr {int (rand() * 4095)}]]
			set randblue [format "%03x" [expr {int (rand() * 4095)}]]
			return "#$randred$randgreen$randblue"
		}	
	}

	proc stepCallBack { } {

		set currentPlotName [spicewish::vectors::get_plotTypeName 0]
	
		#puts $::spice::steps_completed
		# hack !!!
		spicewish::vectors::create 0 "time"
		spicewish::vectors::create 0 "a0_x1_y0"

	}

	proc save_postscript { w } { 
			variable view_plotName

			if {![winfo exists $w]} {
				puts stderr "Error: '$w' doesn't exist"
				return
			}
			
			set plotNum [spicewish::viewer::get_index $w]
			if {$plotNum == -1} {
				puts stderr "Error: invalid viewer"
				return
			}
			
			set fileName "[string range $w 1 end].ps"

			set plotNum [spicewish::vectors::get_plotNum $view_plotName($plotNum)]
			set w "$w.g"

			# save current backround colours etc
			set prevSetting(title)         [$w cget -title]
			set prevSetting(plotrelief)    [$w cget -plotrelief]
			set prevSetting(bg)            [$w cget -background]
			set prevSetting(xaxis_bg)      [$w xaxis cget -background]
			set prevSetting(yaxis_bg)      [$w xaxis cget -background]
			set prevSetting(legend_relief) [$w legend cget -relief]
			set prevSetting(legend_bg)     [$w legend cget -background]
			

			# apply white background
			$w configure -title [spice::plot_title $plotNum]
			$w configure -plotrelief "flat"
			$w configure -background "white"
			$w xaxis configure -background "white"
			$w yaxis configure -background "white"
			$w legend configure -relief "flat"
			$w legend configure -background "white"
			

			# save postscript
			$w postscript output $fileName -maxpect 1

			# reapply inital colours
			$w configure -title $prevSetting(title)
			$w configure -plotrelief $prevSetting(plotrelief)
			$w configure -background $prevSetting(bg)
                        $w xaxis configure -background $prevSetting(xaxis_bg)
                        $w yaxis configure -background $prevSetting(yaxis_bg)
			$w legend configure -relief $prevSetting(legend_relief)
			$w legend configure -background $prevSetting(legend_bg)
	

			puts stdout "created '$fileName'"	
	}

	proc Update { } {
		variable view_traces
		variable view_plotName
		variable vecUpdate_flag
		variable view_Xscale
		variable view_plotName
		
		set vecUpdate_flag 1

		foreach index [names] {

			set plotTypeName $view_plotName($index)
			set plotNum [spicewish::vectors::get_plotNum $plotTypeName]
							
			foreach trace "$view_traces($index) $view_Xscale($index)" {
				set currentPlot [spice::getplot]
				spice::setplot $view_plotName($index)

				::spice::X_Data set ""
				spice::bltplot [evaluate $trace]

				eval $\{::$view_plotName($index):v:$trace\} set \"::spice::Y_Data\"

				spice::setplot $currentPlot
			}

		}
		set vecUpdate_flag 0
		return ""
	}
	
	proc Destroy { w } {
		variable view_traces
		variable view_w
		
		foreach index [names ] {
			if {[string match $view_w($index) $w]} {
				unset view_w($index)
				catch {unset view_traces($index)}
			}
		} 

		destroy $w
	}

	proc evaluate { list } {

	        foreach char "\[ \]" {
	                set repList ""
	                for {set i 0} {$i < [string length $list]} {incr i} {
	                        if {[string range $list $i $i] == $char} {
	                                lappend repList $i
	                        }
	                }
	                set cnt 0
	                foreach index $repList {
	                        set list [string replace $list [expr $index + $cnt] [expr $index + $cnt] "\\$char"]
	                        incr cnt
	                }
	        }
	
	        return $list
	}
}

namespace eval vectors {
	
	proc names { {plot 0} } {
		# traces for current plot
		return 	[spice::plot_variables $plot]	
		
	}

	proc exists { plot name } {

		set list [names $plot]

		if {[lsearch -exact $list $name] != -1} {
			return $name
		} else {
			::spice::Y_Data set ""
                       	uplevel #0 ::spice::bltplot [spicewish::viewer::evaluate $name]

                       	if {[::spice::Y_Data length] != 0} {
                               	return $name
                       	} else {
				return ""
			}
		}
	}

	proc get_plotNum { plotTypeName } {

		for {set i 0} {$i < [avaliable_plotNums]} {incr i} {
                        if {[get_plotTypeName $i] == $plotTypeName} {
                                return $i
                        }
                }
                return -1
	}

	proc get_plotTypeName { plotNum } {
		return [spice::plot_typename $plotNum]
	}

	proc avaliable_plotNames { } {
		
		for {set plotNum 0} {$plotNum < [avaliable_plotNums]} {incr plotNum} {
			lappend list [get_plotTypeName $plotNum]	
		}
		return $list
	}

	proc validVectorList { plotNum args } {
		set validList ""
	
		foreach vector $args {
			set vector [string tolower $vector]
			
			if {[exists $plotNum $vector] != ""} {		
				lappend validList $vector
			}
		}	

		return $validList
	}
	
	proc type { plotNum name } {
		set spice_data [spice::spice_data $plotNum] ;# "{a0 voltage}{time time}"

		set name       [string tolower $name]
		set spice_data [string tolower $spice_data]

		foreach plot_data $spice_data {
			set p_name [lindex $plot_data 0] ;# node name
			set p_type [lindex $plot_data 1] ;# voltage/current/time

			if {$p_name == $name} {
				return $p_type
			}
		}
		return ""
	}

	proc avaliable_plotNums { } {
		set cnt 0
		while {![catch { spice::plot_name $cnt }]} {
			incr cnt
		}
		return $cnt
	}
	
	proc defaultScale { plotNum } {

		return [spice::plot_defaultscale $plotNum]
	}
	
	proc create_vecName { plotNum nodeName } {
		
		set name "::[get_plotTypeName $plotNum]:v:${nodeName}"
	
		if {![info exists $name]} {
			set vecName [blt::vector #auto]
			eval set $name $vecName
		} else {
			eval set vecName \${$name}
		}
		return $vecName
	}


}
