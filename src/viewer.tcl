
namespace eval viewer {

	variable wNames ".tclspice"
	variable viewCnt 1

	variable viewTraces
	variable viewTraceScale
	variable view_w
	variable view_traces
	variable view_type
	variable view_plotName

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
		set plotName [spicewish::vectors::plotTypeName $plotNum]

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
		Blt_ZoomStack $w.g 
		Blt_Crosshairs $w.g 
		$w.g grid on
		$w.g axis configure x -command { spicewish::viewer::axis_callback } -subdivisions 2

		set view_w($viewCnt) "$w"

		# freq/risetime measurments
		spicewish::meas_markers::bindings $w

		eval "wm protocol $w WM_DELETE_WINDOW {spicewish::viewer::Destroy $w}"

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

	proc names { } {
		variable view_w
	
		return [array names view_w]
	}

	proc plot { args } {
		eval "oplot 0 $args"
	}

	proc oplot { plotNum args } {

		variable view_w
		variable view_traces
		variable view_type
		variable view_plotName

		eval "set args \[spicewish::vectors::validVectorList $plotNum $args\]"	

		if {$args == ""} {return}
		
		# create a new viewer
		set viewNum [create $plotNum]
		set w $view_w($viewNum)
		set view_traces($viewNum) ""
		set view_type($viewNum) "plot"
		set view_plotName($viewNum) [spicewish::vectors::plotTypeName $plotNum]

		# add/update default scale 
		set defBLTvec [spicewish::vectors::create $plotNum [spicewish::vectors::defaultScale $plotNum]]

		if {$defBLTvec == ""} {
         		puts "Error: plotting '[spicewish::vectors::defaultScale $plotNum]'"
                	return
              	}

		set traceCnt 0
		#loop for all vectors
		foreach vectorName $args {

			# don't allow duplicate traces

			set BLTvec [spicewish::vectors::create $plotNum "$vectorName"]

			if {$BLTvec == ""} {
				puts "Error: plotting '$vectorName'"
				continue
			}

			lappend view_traces($viewNum) $vectorName

			$w.g element create "$vectorName" \
				-xdata $defBLTvec \
				-ydata $BLTvec \
				-symbol "circle" \
		    		-linewidth 2 \
		    		-pixels 3 \
				-color [trace_colour $traceCnt]
	
			incr traceCnt
		}
		return $w
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

	proc get_plotNum { plotTypeName } {

		for {set i 0} {$i < [spicewish::vectors::avaliable_plots]} {incr i} {
			if {[spicewish::vectors::plotTypeName $i] == $plotTypeName} { 
				return $i
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

		set currentPlotName [spicewish::vectors::plotTypeName 0]
	
		#puts $::spice::steps_completed
		# hack !!!
		spicewish::vectors::create 0 "time"
		spicewish::vectors::create 0 "a0_x1_y0"

	}

	proc Update { } {
		variable view_traces
		variable view_plotName

		foreach index [names] {

			set plotTypeName $view_plotName($index)
			set plotNum [get_plotNum $plotTypeName]
			
			# default scale
			spicewish::vectors::create $plotNum [spicewish::vectors::defaultScale $plotNum]

			foreach trace $view_traces($index) {
				spicewish::vectors::create $plotNum $trace
			}
		}
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
			return "" 
		}
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

	proc plotTypeName { plot } {
		# plot type eg tran1, dc1
		return [spice::plot_typename $plot]
	}
	
	proc type { plot name } {
		set spice_data [spice::spice_data $plot] ;# "{a0 voltage}{time time}"

		set name       [string tolower $name]
		set spice_data [string tolower $spice_data]

		foreach plot $spice_data {
			set p_name [lindex $plot 0] ;# node name
			set p_type [lindex $plot 1] ;# voltage/current/time

			if {$p_name == $name} {
				return $p_type
			}
		}
		return ""
	}

	proc avaliable_plots { } {
		set cnt 0
		while {![catch { spice::plot_name $cnt }]} {
			incr cnt
		}
		return $cnt
	}
	
	proc defaultScale { plot } {

		return [spice::plot_defaultscale $plot]
	}
		
	proc create { plot name } {

		# valid vector 
		if {[exists $plot $name] == ""} {return ""}

		if { ![info exists "::[plotTypeName $plot]:v:${name}"]} {
			set ::[plotTypeName $plot]:v:${name} "[blt::vector #auto]"
		}

		eval "set tmp \${::[plotTypeName $plot]:v:${name}}"

		eval spice::plot_getvector $plot $name $tmp

		return $tmp
	}

}
