
namespace eval meas_dialog {

	variable measCnt 1
	variable nName{} ""
	variable cycleCnt{} ""
	variable freq_d{} ""
	variable riseTime_d{} ""
	variable _propDelay{} ""
	variable _riseTime_20{} ""
	variable VDD 2.5
	

	proc create_widget { } {
		variable measCnt
		variable nName
		
		set w ".meas${measCnt}"
		toplevel $w
			
		pack [label $w.lNode -textvariable [namespace current]::nName($measCnt) -background "grey" -width 20]
		pack [frame $w.fFreq] -fill x
		pack [label $w.fFreq.l -text "Freq :"] -side left
		pack [label $w.fFreq.v -textvariable spicewish::meas_dialog::freq_d($measCnt)] -side right
		pack [frame $w.fRiseTime] -fill x
		pack [label $w.fRiseTime.l -text "RiseTime :"] -side left
		pack [label $w.fRiseTime.v -textvariable [namespace current]::riseTime_d($measCnt)] -side right
		pack [frame $w.fCycleCnt] -fill x
		pack [label $w.fCycleCnt.l -text "CycleCnt :"] -side left
		pack [label $w.fCycleCnt.v -textvariable [namespace current]::cycleCnt($measCnt)] -side right
		pack [button $w.bExit -text "Close"]
		bind $w.bExit <ButtonPress-1> { spicewish::meas_dialog::destroy_widget [winfo parent %W] }
		
	}

	proc destroy_widget { w } {
	
		if {![scan $w ".meas%i" index]} {return}
	
		# Remove all associated triggers
		catch { spice::unregisterTrigger ${index}_propDelay }
		catch {	spice::unregisterTrigger ${index}_riseTime_20 }
		catch {	spice::unregisterTrigger ${index}_riseTime_80 }
		
		after 10 destroy $w     ;# can't destroy widget beacuse of bind to button
	}

	proc TriggerCallBack { args } { 		
		variable nName
		variable freq_d
		variable cycleCnt
		variable riseTime_d
		variable _propDelay
		variable _riseTime_20
		
       	#	set nName [lindex $args 0]           ;# - NodeName
	        set tTime [lindex $args 1]           ;# - Trigger Time
	        set sStep [expr [lindex $args 2] -1] ;# - pop trigger returns steps completed not spice step
	        set mName [lindex $args 5]           ;# - MarkerName

		if {![scan $mName "%i_" index]} { return }
		
	        if {$mName == "${index}_propDelay"} {	
	                set freq_d($index) "[spicewish::eng::float_eng [expr 1/($tTime - $_propDelay($index))]]Hz"
	                set _propDelay($index) $tTime
	                incr cycleCnt($index)
	        }
	        if {$mName == "${index}_riseTime_20"} {
	                set _riseTime_20($index) $tTime
	        }
	
	        if {$mName == "${index}_riseTime_80"} {
	                set riseTime_d($index) "[spicewish::eng::float_eng [expr $tTime - $_riseTime_20($index)]]s"
	        }		
		
	}

	proc stepCallBack { } {
		
	}

	proc registerTrigger { } {

		variable measCnt
		variable nName
		variable VDD
		
		spice::registerTrigger $nName($measCnt) [expr $VDD *0.5] [expr $VDD *0.5] 1 ${measCnt}_propDelay
		spice::registerTrigger $nName($measCnt) [expr $VDD *0.2] [expr $VDD *0.2] 1 ${measCnt}_riseTime_20
		spice::registerTrigger $nName($measCnt) [expr $VDD *0.8] [expr $VDD *0.8] 1 ${measCnt}_riseTime_80
	}

	proc add { nodeName } {
		variable measCnt
		variable nName
		variable cycleCnt
		variable freq_d
		variable riseTime_d
		variable _propDelay
		variable _riseTime_20

		if {$spice::steps_completed == 0 } {	
			return
		}
			
		set nName(${measCnt}) $nodeName
		set freq_d(${measCnt}) 0
		set cycleCnt(${measCnt}) 0
                set riseTime_d(${measCnt}) 0
		set _propDelay(${measCnt}) 0
		set _riseTime_20(${measCnt}) 0
	
		create_widget	
		
		registerTrigger

		incr measCnt
	}

	spice::registerTriggerCallback [namespace current]::TriggerCallBack
}


namespace eval meas_markers {

        variable VDD 2.5

	proc preciseTrigPoint { trig_stepV trig_stepT prev_stepV prev_stepT Vtrigger} {

		set spiceTimeStep [expr $trig_stepT - $prev_stepT]
		set Vrate_through_trig [expr ($trig_stepV - $prev_stepV) / $spiceTimeStep]
		set trigger_delta_time [expr (0 - (($trig_stepV - $Vtrigger) / $Vrate_through_trig))]
		set tTrigger [expr $trig_stepT + $trigger_delta_time ]
	
		return $tTrigger
	}

	proc riseTime { w x y } {

		variable VDD
	
		set index [marker_position $w $x $y]
		
		if {$index == -1} { return }
	
		# find the time when crosses 20% of VDD
		for {set i $index} {$i < [nodeData length]} {incr i} {
			set cur_V [nodeData index $i]
			set cur_T [timeData index $i]
			set prev_V [nodeData index [expr $i -1]]
	                set prev_T [timeData index [expr $i -1]]
	
			if {$cur_V > [expr $VDD *0.2]} {
				set x1 [preciseTrigPoint $cur_V $cur_T $prev_V $prev_T [expr $VDD *0.2]]
				set y1 [expr $VDD *0.2]
				break
			}
		}	
		
		# find the time when crosses 80% of VDD
	        for {set i $index} {$i < [nodeData length]} {incr i} {
		    set cur_V [nodeData index $i]
		    set cur_T [timeData index $i]
		    set prev_V [nodeData index [expr $i -1]]
		    set prev_T [timeData index [expr $i -1]]
		    
		    if {$cur_V > [expr $VDD * 0.8]} {
			set x2 [ preciseTrigPoint $cur_V $cur_T $prev_V $prev_T [expr $VDD *0.8]]
			set y2 [expr $VDD *0.8]
			break
		    }
	        }
		
	        $w line create "m_riseTime" -xdata {$x1 $x2} \
		    -ydata {$y1 $y2} \
		    -color "black" \
		    -label "" \
		    -symbol "cross"
		
		set RiseTime [expr $x2 - $x1]	
		set xd [expr $x1 + (($x2 - $x1) /2)] 
		set yd [expr $y1 + (($y2 - $y1) /2)]
		
	        $w marker create text -text "[spicewish::eng::float_eng $RiseTime]s" \
		    -coords {$xd $yd} \
		    -name "m_riseTime_t" \
		    -anchor "nw"
		
		blt::vector destroy nodeData
	        blt::vector destroy timeData
	}

	proc propDelay { w x y } {

	    variable VDD    

	    set index [marker_position $w $x $y] 
	    
	    if {$index == -1} { return}
	    
	    # find the time when crosses 50% of VDD
		for {set i $index} {$i < [nodeData length]} {incr i} {
			set cur_V [nodeData index $i]
			set cur_T [timeData index $i]
			set prev_V [nodeData index [expr $i -1]]
			set prev_T [timeData index [expr $i -1]]
	
			if {$cur_V > [expr $VDD * 0.5]} {
				set x1 [preciseTrigPoint $cur_V $cur_T $prev_V $prev_T [expr $VDD *0.5]]
				set y1 [expr $VDD * 0.5]
				set index $i
				break
			}
		}	
	
		# second crossing of 50% VDD
	    	set state 0
		for {set i $index} {$i < [nodeData length]} {incr i} {
			set cur_V [nodeData index $i]
			set cur_T [timeData index $i]
			set prev_V [nodeData index [expr $i -1]]
			set prev_T [timeData index [expr $i -1]]
	
			if {$state && ($cur_V > [expr $VDD * 0.5])} {
				set x2 [preciseTrigPoint $cur_V $cur_T $prev_V $prev_T [expr $VDD * 0.5]]
			   	set y2 [expr $VDD * 0.5]			       
		    		set state 0
				break
			} elseif {(!$state) && ($cur_V < [expr $VDD * 0.5])} {
		    		set state 1
			}
	    	}
	    
		$w line create "m_riseTime" -xdata {$x1 $x2} \
	                -ydata {$y1 $y2} \
	                -color "black" \
	                -label "" \
	                -symbol "cross"
	
		set Freq [expr 1/($x2 - $x1)]
	        set xd [expr $x1 + (($x2 - $x1) /2)]
	        set yd [expr $y1 + (($y2 - $y1) /2)]
	
	        $w marker create text -text "[spicewish::eng::float_eng $Freq]Hz" \
	                -coords {$xd $yd} \
	                -name "m_riseTime_t" \
	                -anchor "nw" \
			-xoffset 10 -yoffset 10
	
		blt::vector destroy nodeData
		blt::vector destroy timeData
	}   
	
	proc marker_position { w x y } { 
		
		variable VDD

		# get VDD value from spice
		set VDD [spice::get_value vdd [expr $::spice::steps_completed -1]]

		catch { $w element delete "m_riseTime" }
	        catch { $w marker delete "m_riseTime_t" }

		set x [expr $x - 10]
		set y [expr $y - 28] 

		set currentNode [$w element get current]
		
		if {$currentNode == "" } { return -1 }
	
		set xaxis [$w xaxis invtransform $x]
		set yaxis [$w yaxis invtransform $y]
		
		# get blt vectors
		blt::vector create nodeData
		blt::vector create timeData
	
		spice::spicetoblt time timeData
		spice::spicetoblt $currentNode nodeData
	
		if {![$w element closest $x $y closestData -halo 5.0i]} {return -1}
	
		set prev_V [nodeData index $closestData(index)]
				
		# test if rising edge 20-80%
		if {($prev_V < [expr $VDD *0.2]) || ($prev_V >[expr $VDD * 0.8])} {
			return -1
		}
	
		# test next voltage is greater (rising edge)
		if {$prev_V > [nodeData index [expr $closestData(index) +1]]} {
			return -1
		}
	
		# back track index until rise of waveform
		for {set index $closestData(index)} { $index > 0} {incr index -1} { 
			set cur_V [nodeData index $index]
			if {$cur_V > $prev_V} { break }
			set prev_V [nodeData index $index]
		}
		return $index
	}
	
	proc release { w x y } {

		catch {	$w element delete "m_riseTime" }
		$w marker delete "m_riseTime_t"
	}	
	
	proc bindings { w } {
	
		bind $w <Key-F1> { spicewish::meas_markers::release %W.g %x %y }
		bind $w <Key-F2> { spicewish::meas_markers::riseTime %W.g %x %y }
                bind $w <Key-F3> { spicewish::meas_markers::propDelay %W.g %x %y}
	}
}




namespace eval motionMeasure {

	variable measInfo

	proc create { graph } {
		eval "bind $graph <ButtonPress-1> { [namespace current]::SetMeasPoint %W %x %y }"
		eval "bind $graph <ButtonRelease-1> { [namespace current]::removeMarker %W }"
	}

	proc InitVars { graph } {
		variable measInfo
		set measInfo($graph,A,x) {}
		set measInfo($graph,A,y) {}
		set measInfo($graph,B,x) {}
		set measInfo($graph,B,y) {}
		set measInfo($graph,corner) A
	}

	proc removeMarker { graph } {
		variable measInfo
	        bind $graph <Motion> {}
        	$graph marker delete "zoomOutlineV"
	        $graph marker delete "zoomOutlineH"
	}

	proc SetMeasPoint { graph x y } {
	        variable measInfo

	        if { ![info exists measInfo($graph,corner)] } {
	                InitVars $graph
	        }
	        GetCoords $graph $x $y $measInfo($graph,corner)
	
	        eval "bind $graph <Motion> { [namespace current]::GetCoords %W %x %y B; [namespace current]::Box %W }"
	}

	proc GetCoords { graph x y index } {
		variable measInfo
		if { [$graph cget -invertxy] } {
			set measInfo($graph,$index,x) $y
			set measInfo($graph,$index,y) $x
		
		} else {
			set measInfo($graph,$index,x) $x
			set measInfo($graph,$index,y) $y
    		}
	}

	proc Box { graph } {
		variable measInfo
		if { $measInfo($graph,A,x) > $measInfo($graph,B,x) } {
      			set x1 [$graph xaxis invtransform $measInfo($graph,B,x)]
			set y1 [$graph yaxis invtransform $measInfo($graph,B,y)]
			set x2 [$graph xaxis invtransform $measInfo($graph,A,x)]
			set y2 [$graph yaxis invtransform $measInfo($graph,A,y)]
		} else {
			set x1 [$graph xaxis invtransform $measInfo($graph,A,x)]
			set y1 [$graph yaxis invtransform $measInfo($graph,A,y)]
			set x2 [$graph xaxis invtransform $measInfo($graph,B,x)]
			set y2 [$graph yaxis invtransform $measInfo($graph,B,y)]
		}
		set coords { $x1 $y1 $x2 $y1 $x2 $y2 $x1 $y2 $x1 $y1 }
		if { [$graph marker exists "zoomOutlineV"] } {
			$graph marker configure "zoomOutlineV" -coords "$x1 $y1 $x1 $y2"
			$graph marker configure "zoomOutlineH" -coords "$x1 $y2 $x2 $y2"
		} else {
			set X [lindex [$graph xaxis use] 0]
			set Y [lindex [$graph yaxis use] 0]

		        $graph marker create line -coords "$x1 $y1 $x1 $y2" -name "zoomOutlineV" -linewidth 2
		        $graph marker create line -coords "$x1 $y2 $x2 $y2" -name "zoomOutlineH" -linewidth 2
  		}

		set ::spicewish::viewer::meas_x1($graph) $x1
		set ::spicewish::viewer::meas_y1($graph) $y1
		set ::spicewish::viewer::meas_x2($graph) $x2
		set ::spicewish::viewer::meas_y2($graph) $y2
		set ::spicewish::viewer::meas_dx($graph) [expr $x2 - $x1]
		set ::spicewish::viewer::meas_dy($graph) [expr $y2 - $y1]
	}
}
