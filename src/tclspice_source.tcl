#! /usr/bin/wish
#
# Copyright (c) 2003 MultiGiG.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version, 
# provided that the above copyright notice appear in all copies.
#
# comments : a.dawe@multigig.com
#
# spicewish
# spicewish version : 0.2.1.1
#

# packages
package require BLT
package require Tclx

# globals
set ::spicewish_globals(version) 0.2.1.1
set ::spicewish_globals(fileName) ""
set ::spicewish_globals(include_fileNames) ""
set ::spicewish_globals(viewerCount) 1
set ::spicewish_globals(editor) 0
set ::spicewish_globals(editor_syntaxHighlight) 1
set ::spicewish_globals(editor_statusbar) 1
set ::spicewish_globals(editor_width) 568
set ::spicewish_globals(editor_height) 344
set ::spicewish_externalTraces ""


#========================================================
#========================================================
#   viewer
# sw_viewer_callback_Yaxis
# sw_viewer_clearTraces 
# sw_viewer_png
# sw_viewer_snapshot 
# sw_viewer_postscript 
# sw_viewer_names 
# sw_viewer_traces 
# sw_viewer_create_menu
# sw_viewer_create_measurments
# sw_viewer_create
#
#========================================================
#========================================================

#---------------------------------------------------------------------------------------------------
# convert the yaxis value to engineering units
#  - if $::spicewish_engUnits($viewer) the convert to eng units
#  - else return standard value
#
proc sw_viewer_callback_Yaxis { widget value } {
   
    regexp  {(.[0-9A-z]+)} $widget viewer

    if { $::spicewish_engUnits($viewer) } { 
	return [ sw_float_eng $value]
    } else { 
	return $value
    }
} 


#---------------------------------------------------------------------------------------------------
# remove all the traces from a viewer
#  - also delete all the traces associated blt::vectors
#
proc sw_viewer_clearTraces { viewer } {

    # error traps
    if { ! [ winfo exists $viewer ] } { return }

    set traceList [ sw_viewer_traces $viewer ]
    
    # remove traces
    eval "sw_remove $viewer $traceList"

    # update slots 
    sw_traceplace_slotDimensions $viewer
    sw_traceplace_slotUpdate $viewer

} 


#-------------------------------------------------------------------------------------------------
# uses xwd ( X  Window System window dumping utility ) to store viewer image
#   - converts xwd image to png
#
proc sw_viewer_png { viewer { fileName "" } } { 
    
    if { ! [ winfo exists $viewer ] } { return }
        
    # filename
    if { $fileName == "" } {
	set initialFileName "[ string range $viewer 1 end ].png" 
	set fileName  [ tk_getSaveFile -defaultextension .png  -initialfile $initialFileName   \
			    -filetypes { {{Portable Network Graphic} *.png } { {JPEG} *.jpg } } \
				-parent $viewer  ]
	if { $fileName == "" } { return }  
    }

    set xwdFileName [string range $fileName 0 [ expr [ string last "." $fileName ] -1 ]]
    set xwdFileName "$xwdFileName.xwd"  
    set pngFileName $fileName

    # make sure window is on top
    raise $viewer 
    update

    # grab X window image 
    if { [ catch { exec xwd -silent -name "[ wm title $viewer ]" > $xwdFileName } ] } { return }

    # convert xwd to png
    if { [ catch { exec convert $xwdFileName $pngFileName } ] } { return }

    # remove temp xwd file
    if { [ file exists $xwdFileName ] } { file delete $xwdFileName }
   
} 

#--------------------------------------------------------------------------------------------------
# generates a tcl script of a viewer
#   - saves all plots
#
#     ******* UNDER CONSTRUCTION *******************
#
proc sw_viewer_snapshot { viewer } { 

    # script fileName
    set scriptFileName "viewier.sw_snap.tcl"
    set script_fileId [open $scriptFileName "w"]
    
    # generate script header
    puts $script_fileId "# SpiceWish generated viewer snapshot"
    puts $script_fileId "# -- generated : [ clock format [ clock seconds ]  -format %c ]" 
    puts $script_fileId "package require spice"
    puts $script_fileId "spice_init_gui /home/ad/proj1.cir \n"

    # loop for each trace in the vector 
    foreach traceInfo $::spicewish_plots($viewer) {
	set traceName "_[ lindex $traceInfo 0 ]"
	set traceVector [ lindex $traceInfo 1 ]
	set timeVector "$viewer.time"
	set traceColour [ lindex $traceInfo 2 ]

	set xdata [ $timeVector range 0 [ expr [ $timeVector length ] -1 ] ]
	set ydata [ $traceVector range 0 [ expr [ $traceVector length ] -1 ] ]

	puts $script_fileId "sw_externaltrace_add  $traceName \"$xdata\" \"$ydata\" $traceColour "
    }  

    close $script_fileId
} 


#---------------------------------------------------------------------------------------------------
# create a post script of the viewer
#
# ** Known BUG -  causes a 'Segmentation fault (core dumped)' after the postscript has been genereted
#
proc sw_viewer_postscript { viewer { fileName "" } } {
    
    # error trap
    if { ! [ winfo exists $viewer ] } { return }
    
    # no filename passed 
    if { $fileName == "" } {
	set initialFileName "[ string range $viewer 1 end ].ps" 
	set fileName  [ tk_getSaveFile -defaultextension .ps  -initialfile $initialFileName   \
				-filetypes { {{postscript} *.ps } } \
				-parent $viewer  ]
	if { $fileName == "" } { return }  
    }

    # generate the postscript
    blt::busy hold $viewer
    update
    $viewer.g postscript output $fileName
    update
    blt::busy release $viewer
    update
}

#------------------------------------------------------------------------------------------------------
# returns a list of all the viewers
#
proc sw_viewer_names { } { 

    set returnlist ""
    
    foreach winname [winfo children .] {	
	if {  [ regexp {.tclspice[0-9]+} $winname ] } { lappend returnlist $winname } 	
    }
   
    return $returnlist
} 

#-------------------------------------------------------------------------------------------------------
# returns the names of traces in a viewer
# 
proc sw_viewer_traces { viewer } {
    
    set traceList ""
    
    if { ! [ info exists ::spicewish_plots($viewer) ] } { return "" } 

    for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $::spicewish_plots($viewer) ] } { incr dataHolderCnt } { 
	set traceInfo [ lindex $::spicewish_plots($viewer) $dataHolderCnt ] 
	set traceName [ lindex $traceInfo 0 ] 
	
	lappend traceList $traceName
    }   

    return $traceList 
}

#-------------------------------------------------------------------------------------------------------
# create a menu at the top left of the viewer
#
proc sw_viewer_create_menu { w } {
    
    set w_menu "$w.meas"

    pack [ menubutton $w_menu.mOptions -text "options" -menu $w_menu.mOptions.m  ] -side left
    set optionsMenu [menu $w_menu.mOptions.m -tearoff 0]
    $optionsMenu add command -label "Rename viewer" -state disabled
    $optionsMenu add cascade -label "Save" -menu $w_menu.mOptions.m2 
    set optionsMenuSave [ menu $w_menu.mOptions.m2 -tearoff 0 ]
    $optionsMenuSave add command -label "Snapshot"  -state disabled
    $optionsMenuSave add command -label "Postscript" -command "sw_viewer_postscript $w"  -state disabled
    $optionsMenuSave add command -label "Png" -command "sw_viewer_png $w"
    $optionsMenu add checkbutton -label "Eng units" -variable ::spicewish_engUnits($w) -command "sw_update $w"
    $optionsMenu add checkbutton -label "Trace Place" -variable ::spicewish_tp($w) -command "sw_traceplace_mode $w -1" 
    $optionsMenu add command -label "Close" -command "destroy $w"
 
    pack [ menubutton $w_menu.mTraces -text "traces" -menu $w_menu.mTraces.m ] -side left
    set tracesMenu [menu $w_menu.mTraces.m  -tearoff 0 ]
    $tracesMenu add command -label "Add/Remove  traces" -command "sw_nodeselection_create $w"
    $tracesMenu add command -label "Update traces" -command "sw_update $w"
    $tracesMenu add command -label "Clear traces" -command "sw_viewer_clearTraces $w" 
} 

#--------------------------------------------------------------------------------------------------------------
# creates mesurment values at top of viewer
#
proc sw_viewer_create_measurments { w } {
    
    set ::spicewish_meas_dx($w.g) 0.00000000000000
    set ::spicewish_meas_dy($w.g) 0.00000000000000
    set ::spicewish_meas_x1($w.g) 0.00000000000000
    set ::spicewish_meas_y1($w.g) 0.00000000000000
    set ::spicewish_meas_x2($w.g) 0.00000000000000
    set ::spicewish_meas_y2($w.g) 0.00000000000000
    
    set w_meas "$w.meas"

    # measurements 
    pack [ label $w_meas.dyv -textvariable  spicewish_meas_dy($w.g)  -background "lightgrey" -width 16 ] -side right
    pack [ label $w_meas.dy -text "dy :" -background "lightgrey" ] -side right
    bind  $w_meas.dyv <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "dy :  $::spicewish_meas_dy($w.g)"}
   	
    pack [ label $w_meas.dxv -textvariable  spicewish_meas_dx($w.g)  -background "grey" -width 16 ] -side right
    pack [ label $w_meas.dx -text "dx : " -background "grey" ] -side right
    bind  $w_meas.dxv <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "dx :  $::spicewish_meas_dx($w.g)"}
    bind  $w_meas.dxv <ButtonPress-2> { 
	regexp  {(.[0-9A-z]+)} %W w; puts "freq : [ sw_float_eng [ expr 1 / $::spicewish_meas_dx($w.g) ] ] Hz"}
 
    pack [ label $w_meas.y2v -textvariable  spicewish_meas_y2($w.g)  -background "lightgrey" -width 16  ] -side right
    pack [ label $w_meas.y2 -text "y2 :"  -background "lightgrey"] -side right
    bind  $w_meas.y2v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "y2 :  $::spicewish_meas_y2($w.g)"}
    
    pack [ label $w_meas.y1v -textvariable  spicewish_meas_y1($w.g)  -background "grey"  -width 16] -side right
    pack [ label $w_meas.y1 -text "y1 :" -background "grey"  ] -side right
    bind  $w_meas.y1v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "y1 :  $::spicewish_meas_y1($w.g)"}
    
    pack [ label $w_meas.x2v -textvariable  spicewish_meas_x2($w.g)  -background "lightgrey"  -width 16 ] -side right
    pack [ label $w_meas.x2 -text "x2 :" -background "lightgrey"  ] -side right
    bind  $w_meas.x2v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "x2 :  $::spicewish_meas_x2($w.g)"}
    
    pack [ label $w_meas.x1v -textvariable  spicewish_meas_x1($w.g) -background "grey" -width 16  ] -side right
    pack [ label $w_meas.x1 -text "x1 :" -background "grey"  ] -side right
    bind  $w_meas.x1v <ButtonPress-1> { regexp  {(.[0-9A-z]+)} %W w; puts "x1 :  $::spicewish_meas_x1($w.g)"}

    # cursor motion bindings 
    bind $w.g <Motion> {

	if { $::zoomInfo(%W,B,x) != "" } { 
	    set ::spicewish_meas_x2(%W) [%W xaxis invtransform $::zoomInfo(%W,B,x)]
	    set ::spicewish_meas_y2(%W)  [%W yaxis invtransform $::zoomInfo(%W,B,y)]
	    set ::spicewish_meas_x1(%W)  [%W xaxis invtransform $::zoomInfo(%W,A,x)]
	    set ::spicewish_meas_y1(%W)  [%W yaxis invtransform $::zoomInfo(%W,A,y)]
	    set ::spicewish_meas_dx(%W)  [expr ($::spicewish_meas_x2(%W)  - $::spicewish_meas_x1(%W) )]
	    set ::spicewish_meas_dy(%W)  [expr ($::spicewish_meas_y2(%W)  - $::spicewish_meas_y1(%W) )]
	}
    }  
}  

#--------------------------------------------------------------------------------------------------------
# creates a viewer
#
proc sw_viewer_create { } { 

    # create widget 
    set viewer_w [toplevel ".tclspice$::spicewish_globals(viewerCount)"]
    wm title $viewer_w "\[ .tclspice$::spicewish_globals(viewerCount) \] - $::spicewish_globals(fileName)"
    
    # initial settings
    set ::spicewish_menu($viewer_w) 1
    set ::spicewish_tp($viewer_w) 0 
    set ::spicewish_engUnits($viewer_w) 1

    # measurments
    frame $viewer_w.meas
    
    # trace place
    frame $viewer_w.tracePlace -borderwidth 6
    label $viewer_w.l -text "" -borderwidth 4
    
    # graph
    blt::graph $viewer_w.g -width 1000 -height 200
    eval "$viewer_w.g  axis configure x -scrollcommand { $viewer_w.x  set } "
    eval "$viewer_w.g  axis configure y -scrollcommand { $viewer_w.y  set } "
    eval "scrollbar  $viewer_w.x  -orient horizontal -command { $viewer_w.g  axis view x } -width 9"
    eval "scrollbar $viewer_w.y  -command { $viewer_w.g  axis view y } -width 9"
  
    # graph scale
    
    $viewer_w.g axis configure x -command { sw_viewer_callback_Yaxis } -subdivisions 2
    

    # menu - measurments 
    sw_viewer_create_menu $viewer_w
    sw_viewer_create_measurments  $viewer_w

    # table 
    blt::table $viewer_w \
	0,0 $viewer_w.meas -fill x  -cspan 3 \
	1,0 $viewer_w.tracePlace -fill both \
	2,0 $viewer_w.l -fill both \
	1,1 $viewer_w.g -fill both -rspan 2 \
	1,2 $viewer_w.y -fill y -rspan 2 \
	3,1 $viewer_w.x -fill x

    blt::table configure $viewer_w  r0 r2 c0 c2 r3  -resize none  

    # drag and drop packet for traceplace
    blt::drag&drop source $viewer_w.g -packagecmd { sw_traceplace_dragDropPacket %t %W } -button 2
    set token [blt::drag&drop token  $viewer_w.g -activebackground blue ]
    pack [ label $token.label -text "" ]
    	  
    # higlight trace in legend
    $viewer_w.g legend bind all <Button-2> { %W legend activate [%W legend get current] }
    $viewer_w.g legend bind all <ButtonRelease-2> {  %W legend deactivate [%W legend get current]}

    # highlights the nearest traces to the cursors in legend
    $viewer_w.g element bind all <Enter> { %W legend activate [%W element get current] }
    $viewer_w.g element bind all <Leave> { %W legend deactivate [%W element get current] }


    # graph settings
    Blt_ZoomStack $viewer_w.g 
    Blt_Crosshairs $viewer_w.g 
    $viewer_w.g grid on
    
    # trace place
    sw_traceplace_mode $viewer_w $::spicewish_tp($viewer_w) 

    # shortcut keys
    bind $viewer_w <Control-KeyPress-r> { sw_run }       ;#- run the simulation
    bind $viewer_w <Control-KeyPress-h> { sw_halt }      ;#- stop the simulation
    bind $viewer_w <Control-KeyPress-u> { sw_update } ;#- update the viewer
    bind $viewer_w <Control-KeyPress-c> { 
	raise .
	#wm geometry . "[winfo width .]x[winfo height .]+%X+%Y"
	#update
    }       
    bind $viewer_w <Control-KeyPress-t> { sw_traceplace_mode %W } ;#- toggles the traceplace grid on and off

    bind $viewer_w <Control-KeyPress-m> { 
	if { $::spicewish_menu(%W) } { 
	    set ::spicewish_menu(%W) 0
	    blt::table forget %W.meas
	} else { 
	    set ::spicewish_menu(%W) 1
	    blt::table %W 0,0 %W.meas -fill x  -cspan 3 \
	    } 
    } ;#- toggle menu on and off


    # drag a drop target so that packets can be added form the editor   
    eval "blt::drag&drop target $viewer_w handler string { sw_traceplace_remove .tclspice1 %W %v }"
    eval "blt::drag&drop target $viewer_w.l handler string { sw_traceplace_remove .tclspice1 %W %v }"

    # drag a drop target so that packets can be added form the editor   
    eval "blt::drag&drop target $viewer_w.g handler string { sw_plot $viewer_w %v }"

    incr ::spicewish_globals(viewerCount)

    return $viewer_w
} 



#=========================================================
#=========================================================
#    trace place
# sw_traceplace_canvasUpdate
# sw_traceplace_dragDropPacket
# sw_traceplace_traces
# sw_traceplace_createUniqueVector 
# sw_traceplace_slotUpdate
# sw_traceplace_add
# sw_traceplace_remove
# sw_traceplace_slotDimensions
# sw_traceplace_wheelMove 
# sw_traceplace_mode 
# sw_traceplace_create 
#
#=========================================================
#=========================================================


#--------------------------------------------------------------------------------------------------------
# update the canvas makers at the left hand side of each slot 
# - the markers display the colour of the traces in its associated slots
#
proc sw_traceplace_canvasUpdate { viewer } {

    # error traps 
    if { ! [ winfo exists $viewer ] } { return } 
    if { ! [ winfo exists $viewer.tracePlace.slot1 ] } { return } 
    if { ! [ info exists ::spicewish_tp_plots($viewer) ] } { return } 

    set markersTagName "tracePlaceMarker"
    
    # loop foreach slot
    for { set slotCnt 1 } { $slotCnt <= 15 } { incr slotCnt } {
	
	# remove all previous coloured markers
	$viewer.tracePlace.slot$slotCnt.c delete $markersTagName
	
	# slot height
	set slotHeight [ winfo height $viewer.tracePlace.slot$slotCnt ]

	# number of traces in slot
	set slotsTraces [ sw_traceplace_traces $viewer $slotCnt ] 
	set numTraces [ llength $slotsTraces  ] 

	# no trace in the slot 
	if { $slotsTraces == "" } { continue } 

	# loop foreach trace in the slot 
	for { set dataHolderCnt 0 } { $dataHolderCnt < $numTraces } { incr dataHolderCnt } {

	    set traceName [ lindex $slotsTraces $dataHolderCnt ] 
	    set traceColour [ sw_trace_colour $viewer $traceName ] 
	    
	    set y1 [ expr $dataHolderCnt * ( $slotHeight / $numTraces ) ] 
	    set y2 [ expr ($dataHolderCnt + 1) * ( $slotHeight / $numTraces ) ] 
	    
	    # create canvas rectangle marker
	    $viewer.tracePlace.slot$slotCnt.c create rectangle \
		0 $y1 10 $y2 \
		-tags $markersTagName \
		-fill $traceColour 
	} 
    }
}


#---------------------------------------------------------------------------------------------------------
# cratetes a packet of a trace that can be dragged and dropped into a traceplaces slot 
#   activated when
#     - button 3 over trace in graph
#     - button 3 over legend name
#
#
proc sw_traceplace_dragDropPacket { token graph { slot "" } } {
    
    regexp  {(.[0-9A-z]+)} $graph viewer

    if { ! $::spicewish_tp($viewer) } { return } ;#- returns if traceplace grid not active
	

    if { [$viewer.g  legend activate] != ""} {
	# moving traces from either legend or graph 
	$token.label configure -text [$viewer.g legend activate] -background [ $viewer.g element cget [$graph legend activate ] -color ] -height 1  
	return [ $viewer.g legend activate ] 

    } else { 
	# moving traces from a slot 
	if { [ llength "[ sw_traceplace_traces $viewer $slot ]" ] > 1  } {
	    # multiple traces in slot 
	    $token.label configure -text "Traces"  -background "red"

	} else {
	    # single trace in slot 
	    set traces [ sw_traceplace_traces $viewer $slot ]
	    set traces_colour  [ sw_trace_colour $viewer $traces ] 
	    $token.label configure -text $traces  -background $traces_colour

	} 
	set traces  [ sw_traceplace_traces $viewer $slot ]
	
	return "{$traces}"
	
    }  
}


#----------------------------------------------------------------------
# returns a list of all the traces in a slot 
#
proc sw_traceplace_traces { viewer slot } { 
    
    set traceList ""
    
    # error traps 
    if { ! [ winfo exists $viewer ] } { return "" } 
    if { ! [ info exists ::spicewish_tp_plots($viewer) ] } { return "" } 

    set slotInfo [ lindex $::spicewish_tp_plots($viewer) $slot ]

    for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $slotInfo ] } { incr dataHolderCnt } { 
	set traceInfo [ lindex $slotInfo $dataHolderCnt ] 
	set traceName [ lindex $traceInfo 0 ] 
	
	lappend traceList $traceName
    }   
    return $traceList 
} 

#-----------------------------------------------------------------------
# create a unique blt vector name
#  - spice's node name carn't be used due to illegal characters eg '#'
#
proc sw_traceplace_createUniqueVector { viewer } {
    
    # error traps 
    if { ! [ winfo exists $viewer ] } { return "" } 

    set cnt 1
    while { 1 } {
	set vectorName "$viewer.tp.trace$cnt"	
	if { [ blt::vector names ::$vectorName ] == "" } { break }  

	if { $cnt > 1000 } { break } ;#- just in case
	incr cnt
    }
    return $vectorName
}



#-----------------------------------------------------------------------------------------
# upadates all the trace in the traceplace grid
#
proc sw_traceplace_slotUpdate { viewer } {
    
    #error trap 
    if { ! [ info exists ::spicewish_tp_plots($viewer) ] } { return }

    # loop for each slot
    for { set slotCnt 0 } { $slotCnt <= 15 } { incr slotCnt } {

	set slotDataHolder [ lindex $::spicewish_tp_plots($viewer) $slotCnt ]
	
	# slot dimensions
	set slotinfo [ lindex $::spicewish_tp_dimensions($viewer) $slotCnt ]
	set slotHeight [ lindex $slotinfo 1 ] 
	set slotStartPos [ lindex $slotinfo 0 ] 


	set minHeight ""
	set maxHeight ""

	# loop for each trace in the slot 
	for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $slotDataHolder ] } { incr dataHolderCnt } {
	    set holder_traceInfo [ lindex $slotDataHolder $dataHolderCnt ]
	    set holder_traceName [ lindex $holder_traceInfo 0 ]
	    set holder_traceVector [ lindex $holder_traceInfo 1 ]
	 
	    if { $holder_traceName == "" } { continue }
	    
	    # update blt vector
	    spice::spicetoblt $holder_traceName $holder_traceVector
	    
	    # maximum and minmum height of current trace 
	    # $bltvector(min) cannot be used due to '.' in vector name
	    for { set traceSizeCnt 0 } { $traceSizeCnt < [ $holder_traceVector length ] } { incr traceSizeCnt } {
		set value [ $holder_traceVector range $traceSizeCnt $traceSizeCnt ]
		if { $traceSizeCnt == 0 } {
		    set traceMin $value
		    set traceMax $value
		} else {
		    if { $value < $traceMin } { set traceMin $value }
		    if { $value > $traceMax } { set traceMax $value }
		}		
	    }

	    # minimum height of all traces in slot
	    if { ( $traceMin <  $minHeight )  || ( $minHeight  == "" )  } { set minHeight $traceMin  } 

	    # maximum height of all traces in 
	    if { ( $traceMax >  $maxHeight )  || ( $maxHeight  == "" )  } { set maxHeight $traceMax} 
	}


	#------
	# rescale all traces in the slot 
	if { ( $maxHeight == "" )  || ( $minHeight == "" ) } { continue }
	
	# height + scale factor
	set height [ expr $maxHeight - $minHeight ]
	set factor [ expr $slotHeight / $height ]

	for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $slotDataHolder ] } { incr dataHolderCnt } {
	    set holder_traceInfo [ lindex $slotDataHolder $dataHolderCnt ]
	    set holder_traceVector [ lindex $holder_traceInfo 1 ]
	    
	    if { $holder_traceVector == "" } { continue }

	    eval "$holder_traceVector expr { $holder_traceVector - $minHeight } "
	    eval "$holder_traceVector expr { $holder_traceVector * $factor }"
	    eval "$holder_traceVector expr { $slotStartPos + $holder_traceVector } "
	}
    }
    # update the time vector
    spice::spicetoblt time $viewer.tp.time

    # update the canvas markers
    sw_traceplace_canvasUpdate $viewer
} 



#------------------------------------------------------------------------------------------
# add traces to a traceplace slot
# - reads data holder ::spicewish_tp_plots($viewier)
#
proc sw_traceplace_add { viewer slot args } {

    # error traps
    if { ! [ winfo exists $viewer ] } { return "" } 
    
    # make sure that the data holder exists
    if { ! [ info exists ::spicewish_tp_plots($viewer) ] } { 
	set ::spicewish_tp_plots($viewer) "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}"
    }

    # loop for each trace to be added
    foreach trace $args {
	
	set trace [ string tolower $trace ] ;#- lowercase

	# remove trace if it exists in another slot
	sw_traceplace_remove $viewer $trace 
	
	# test that it's a valid spice node
	if { ! [ sw_spice_validNode $trace ] } { continue }

	# test that trace not already added
	if { [ lcontain [ sw_traceplace_traces $viewer $slot ]  $trace ] } { continue }

	# data holder info
	set traceName $trace
	set traceColour [ sw_trace_colour $viewer $trace ]
	set traceVector [ sw_traceplace_createUniqueVector $viewer ]
	
	# blt vectors
	blt::vector create $traceVector
	blt::vector create $viewer.tp.time ;#- make sure exists
	
	# add to data holder
	set slotsTraces [ lindex $::spicewish_tp_plots($viewer)  $slot ] 
	set ammendedList "$slotsTraces { $trace $traceVector $traceColour } "
	set ::spicewish_tp_plots($viewer) [ lreplace $::spicewish_tp_plots($viewer) $slot $slot $ammendedList ]

	# add trace to the viewer
	if { [ $viewer.g element exists $traceName] } { 
	    $viewer.g element configure $traceName \
		-ydata $traceVector \
		-xdata $viewer.tp.time
	} 
    }

    # update slots 
    sw_traceplace_slotDimensions $viewer
    sw_traceplace_slotUpdate $viewer
} 

#-------------------------------------------------------------------------------------------
# remove traces from the traceplace grid
#
proc sw_traceplace_remove { viewer args } {
  
    # error traps 
    if { ! [ winfo exists $viewer ] } { return } 
    if { ! [ info exists ::spicewish_tp_plots($viewer) ] } { return }

    foreach trace $args  {

	set trace [ string tolower $trace ] ;#- lowercase
	
	# loop through all the slots 
	for { set slotCnt 1 } { $slotCnt <= 15 } { incr slotCnt } {
	    
	    set slotDataHolder [ lindex $::spicewish_tp_plots($viewer) $slotCnt ]


	    # loop for each trace in the slot 
	    for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $slotDataHolder ] } { incr dataHolderCnt } {
		set holder_traceInfo [ lindex $slotDataHolder $dataHolderCnt ]
		set holder_traceName [ lindex $holder_traceInfo 0 ]
		set holder_traceVector [ lindex $holder_traceInfo 1 ]

		if { $holder_traceName == $trace } {

		    # remove from data holder 
		    set ammendedList [ lreplace $slotDataHolder $dataHolderCnt $dataHolderCnt  ]
		    set ::spicewish_tp_plots($viewer) [ lreplace $::spicewish_tp_plots($viewer) $slotCnt $slotCnt $ammendedList ]
		   
		    # destroy blt vector
		    blt::vector destroy $holder_traceVector
		    
		    # clear the vectors in the graph 
		    
		}
	    }
	}
    }

    # update canvas markers 
    sw_traceplace_canvasUpdate $viewer
} 

#--------------------------------------------------------------------------------------------
# create a data holder with the size of each slot 
#   - ::spicewish_tp_dimensions($viewer)
#   - eg {} {0.0 100.0 } {50.0 50.0 } {0.0 50.0 } {75.0 25.0 } {50.0 25.0 } {25.0 25.0 } {0.0 25.0 } {87.5 12.5 } {75.0 12.5 } {62.5 12.5 } {50.0 12.5 } {37.5 12.5 } {25.0 12.5 } {12.5 12.5 } {0.0 12.5 }
#
proc sw_traceplace_slotDimensions { viewer } {
    
    global row_heights

    # error traps
    if { ! [ winfo exists $viewer ] } { return } 
    if { ! [ winfo exists $viewer.tracePlace.slot1 ] } { return }
    
    set widget "$viewer.tracePlace"

    # calclate the total height
    set totalHeight 0
    for { set rowCnt 0 } { $rowCnt < 15 } { incr rowCnt } {
	set totalHeight [ expr $totalHeight +  [ grid rowconfigure $widget $rowCnt -minsize ] ]
    }
    set factor [ expr $totalHeight / 100.0 ] 

    # clear data holder
    set ::spicewish_tp_dimensions($viewer) "{}"

    # calculate the height of each slot 
    for { set slotCnt 1 } { $slotCnt <= 15 } { incr slotCnt } {
	
	set slot_w "$widget.slot$slotCnt"
	
	set grid_info [grid info $slot_w ] 
	set row_span_info  [lindex $grid_info 9]
	set row_info [lindex $grid_info 5]
	
	set calcRowHeight 0 
	set calcRowStart 0
	for { set rowCnt 0 } { $rowCnt < 15 } { incr rowCnt } {
	    
	    if { ( $rowCnt >= $row_info ) && ( $rowCnt < ( $row_info + $row_span_info ) ) } {
		set calcRowHeight [ expr $calcRowHeight + [ grid rowconfigure $widget $rowCnt -minsize ] ]
	    }
	    
	    if {  $rowCnt < $row_info } {
		set calcRowStart [ expr $calcRowStart + [ grid rowconfigure $widget $rowCnt -minsize ] ]
	    }
	     
	}
	set calcRowHeight [ expr  ( $calcRowHeight / $factor ) ]
	set calcRowStart [ expr 100 - ( $calcRowStart / $factor ) - $calcRowHeight  ]

	lappend ::spicewish_tp_dimensions($viewer)  "$calcRowStart $calcRowHeight " 
    }
}

#---------------------------------------------------------------------------------------------
# recalculates the size the slot when mouse wheel moved over it 
#   - wheel up increases the slots size
#   - wheel down decreases the slots size
#
proc sw_traceplace_wheelMove { viewer slot direction } {
    
    # error traps 
    if { ! [ winfo exists $viewer ] } { return "" } 

    global row_heights

    set columns 3
    set rows [expr pow(2, $columns) ]

    set widget "$viewer.tracePlace"
    set slot_w "$viewer.tracePlace.slot$slot"
    
    set grid_info [grid info $slot_w ] 
    set col_info [lindex $grid_info 3]
    set row_span_info  [lindex $grid_info 9]
    set row_info [lindex $grid_info 5]
   
    scan [winfo geometry $widget ] "%dx%d+%d+%d" width total_hei xpos ypos


    # stores current slot sizes
    for {set rows_cnt  0 } {  $rows_cnt < $rows } {  incr rows_cnt} {	
	set row_heights($rows_cnt) [	grid rowconfigure $widget $rows_cnt -minsize ]
	if {$row_heights($rows_cnt) < 10} {set row_heights($rows_cnt) 10}
    }
    
    set allocated_height 0
    
    # increases / decreases selected slots
    for { set rows_cnt $row_info} {$rows_cnt < [expr ($row_info + $row_span_info)  ] } {incr rows_cnt } {	
	
	set valholder $row_heights($rows_cnt) 
	
	if {$direction == "up" } {
	    set valholder [expr  $valholder * 1.1]   ;#- scale up
	} else {
	    set valholder [expr $valholder / 1.1]	 ;#- scale down  
	}
	
	set row_heights_holder($rows_cnt)  $valholder                        ;  #- store back in working holder
	set allocated_height [expr $allocated_height +  $valholder]      ;  #- add the height
    }	
    
    # Return if this would exceed the screen size limit by itself (even without makeing all the other rows 0 height)
    if {$allocated_height >= [ expr $total_hei - 10  ] } {puts "hit limit $allocated_height";  return }
    
    # recalculates the slots sizes
    for {set rows_cnt_outer 0 } { $rows_cnt_outer < $rows} {incr rows_cnt_outer } {
	
	if { [catch {set temp $row_heights_holder($rows_cnt_outer)} ] }  {
	    
	    # Find unallocated heights total (i.e. the total of the elements not yet decided)
	    set unallocated_height 0
	    set allocated_height 0
	    
	    for { set rows_cnt 0 } { $rows_cnt < $rows } { incr rows_cnt} {
		if { [catch {set temp $row_heights_holder($rows_cnt)} ] }  {
		    set unallocated_height [ expr $unallocated_height + $row_heights($rows_cnt) ]     
		} else {
		    set allocated_height [ expr $allocated_height +  $row_heights_holder($rows_cnt) ]	    
		    # case where a slot has been given a value above
		}
	    }	   
	    set scale_to_apply [ expr ($total_hei - $allocated_height) / $unallocated_height ] 
	    set row_heights_holder($rows_cnt_outer)  [expr  ( $row_heights($rows_cnt_outer) * $scale_to_apply) ]
	}
    }
        
    # if the first row 
    # need to be if all the rows are less than the miniumum allowed then set to minimum
    if { ( $row_info == 0 ) } {
	# loop through all the rows checking if they have all dropped below the minumum
	set temp 0 
	    for { set temp_cnt 0 } { $temp_cnt < $rows } { incr temp_cnt } {
		if { $row_heights_holder($temp_cnt) <  [expr $total_hei / $rows ] } { incr temp } 
	    }
	if { $temp == $rows } {
	    for { set temp_cnt 0 } { $temp_cnt < $rows } { incr temp_cnt } {
		set row_heights_holder($temp_cnt)  [expr $total_hei / $rows ]
	    }
	}
    }
    
    # reapply's the new slot sizes
    for {set rows_cnt_outer 0 } { $rows_cnt_outer < $rows} {incr rows_cnt_outer } {
	grid rowconfigure $widget $rows_cnt_outer -minsize $row_heights_holder($rows_cnt_outer)
    }
    
    sw_traceplace_slotDimensions $viewer
    sw_traceplace_slotUpdate $viewer
}

#-------------------------------------------------------------------------------------------
# toggles the trace place grid on and off
#  - if a value 1 or 0 is passed, then the grid is set to that value
#  - if value = '-1' then the global variable $::spicewish_tp($viewer) value is used (need for menu checkbutton)
#  - if no value is passed then the grid is toggled on and off (KeyPress-t)
#
proc sw_traceplace_mode { viewer { mode "" } } { 
    
    # error traps 
    if { ! [ winfo exists $viewer ] } { return "" } 
    
    
    if { $mode != "" } {
	# traceplace modes passed by $mode 
	if { ( $mode == 1 ) || ( $mode == 0 ) } { 
	    set ::spicewish_tp($viewer) $mode 
	} elseif { ( $mode != -1 ) }  { return } 
		  
    } else { 
	# toggle traceplace grid on and off 
	if { $::spicewish_tp($viewer) } {
	    set ::spicewish_tp($viewer) 0
	} else { set ::spicewish_tp($viewer) 1 } 
    }

    # turn traceplace grid on or off depending on $::spicewish_tp($viewer)
    if { $::spicewish_tp($viewer)  == 0 } { 
	# trace place off
	if { [ info exists ::spicewish_plots($viewer) ] } { 
	    for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $::spicewish_plots($viewer) ] } { incr dataHolderCnt } { 
		set traceInfo [ lindex $::spicewish_plots($viewer) $dataHolderCnt ] 
		set traceName [ lindex $traceInfo 0 ] 
		set traceVector [ lindex $traceInfo 1 ]
		
		if { $traceName == "" } { continue }
		
		$viewer.g element configure $traceName \
		    -xdata $viewer.time \
		    -ydata $traceVector 
	    }
	}

	# configure graphs yaxis back to normal + unpack the grid
	$viewer.g axis configure y -min "" -max ""
	blt::table forget $viewer.tracePlace 
	return 
    } else {
	# trace place on 
	# - reset the traces to have to blt vector bound to them 
	foreach trace [ sw_viewer_traces $viewer ] {
	    if { $trace == "" } { continue }
	    $viewer.g element configure $trace \
		-xdata "" \
		-ydata ""
	}
	
	#- bind the bltvectors to traces that are in slots 
	if { [ info exists ::spicewish_tp_plots($viewer) ] } { 
	    # loop for each slot 
	    for { set slotCnt 0 } { $slotCnt <= 15 } { incr slotCnt } {
		set slotDataHolder [ lindex $::spicewish_tp_plots($viewer) $slotCnt ]
		
		# loop for each trace in the slot 
		for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $slotDataHolder ] } { incr dataHolderCnt } {
		    set holder_traceInfo [ lindex $slotDataHolder $dataHolderCnt ]
		    set holder_traceName [ lindex $holder_traceInfo 0 ]
		    set holder_traceVector [ lindex $holder_traceInfo 1 ]
		    
		    if { $holder_traceName == "" } { continue }

		    # re bind the blt vectors to the trace
		    $viewer.g element configure $holder_traceName \
			-xdata $viewer.tp.time \
			-ydata $holder_traceVector
		}
	    }
	}
	
	# configure the yaxis to 0 - 100 + pack the traceplace grid
	$viewer.g axis configure y -min 0 -max 100
	blt::table $viewer 1,0 $viewer.tracePlace -fill both 

	# create traceplace widget if it doesn't exists
	if { ! [ winfo exists $viewer.tracePlace.slot1 ] } { 
	    sw_traceplace_create $viewer
	} 
    }
}

#--------------------------------------------------------------------------------------------
# create the traceplace grid at the left hand side of the viewer
#
proc sw_traceplace_create { viewer } {
    
    # error traps 
    if { ! [ winfo exists $viewer ] } { return "" } 
 
    set columns 3
    set rows [expr pow(2, $columns) ]
    set min_row_hei  1
    set max_traces_per_slot 8 

     for {set columns_cnt 0 } {$columns_cnt <= $columns} {incr columns_cnt } {
	
	set  row_span_holder  [ expr  int (pow(2, $columns) / pow(2, $columns_cnt) ) ]
	
	for {set row_cnt 0} { $row_cnt <= [ expr (pow(2, $columns_cnt) - 1)   ] } {incr row_cnt } {
	    
	    set slotnum [ expr int (int (  [expr  $row_cnt* $row_span_holder] / [expr  pow(2, ($columns - $columns_cnt)) ])   + [expr  int (pow(2, $columns_cnt))]) ] 
	    
	    set w $viewer.tracePlace

	    # button 
	    frame $w.slot$slotnum 
	    set button_w [ button $w.slot$slotnum.b -padx 3 -pady 0 ]
	    set canvas_w [ canvas $w.slot$slotnum.c -background grey -width 2 -height 10 ]
	    
	    pack $canvas_w -side left -fill y 
	    pack $button_w -side right -fill both -expand 1
		 
	    # wheel move
	    eval "bind $button_w <ButtonPress-1> { } "
	    eval "bind $button_w <ButtonPress-4> { sw_traceplace_wheelMove $viewer $slotnum up }"
	    eval "bind $button_w <ButtonPress-5> { sw_traceplace_wheelMove $viewer $slotnum down }"

	    # drag and drop source
	    eval "blt::drag&drop source $button_w -packagecmd { sw_traceplace_dragDropPacket %t %W $slotnum } -button 2"

	    set token [blt::drag&drop token $button_w -activebackground blue ]
	    pack [ label $token.label -text "" ]

	    

	    # drag and drop target
	    eval "blt::drag&drop target $button_w handler string { sw_traceplace_add $viewer $slotnum %v }"



	    # grid
	    grid configure $w.slot$slotnum -row [ expr $row_cnt * $row_span_holder  ] \
		-column [ expr ($columns - $columns_cnt) * $max_traces_per_slot ] \
		-sticky "nsew" \
		-rowspan $row_span_holder

	    grid columnconfigure "$w"  [expr $columns_cnt * $max_traces_per_slot] -weight 1
	    grid rowconfigure "$w" $row_cnt -weight 1
	}
    } 
    
    # default row size
    for {set rows_cnt  0 } {  $rows_cnt < $rows } {  incr rows_cnt} {
	set ::row_heights($rows_cnt) 10
	grid rowconfigure $w $rows_cnt -minsize 10
    }

    # bug when the traceplace grid is first made resizing the slots cause the viewer to grow larger
    update ;# required to get size of slots ( winfo height )
    wm geometry $viewer "[expr [ winfo width $viewer ] +1]x[ winfo height $viewer]"

} 




#========================================================
#========================================================
#        add external traces
# sw_externaltrace_createUniqueVector
# sw_externaltrace_add
# sw_externaltrace_addFile
#
#========================================================
#========================================================

#----------------------------------------------------------------------------------------------------
#
#
proc sw_externaltrace_createUniqueVector { } {

    set cnt 1
    while { 1 } {
	set vectorName "sw.external.trace$cnt"	
	if { [ blt::vector names ::$vectorName ] == "" } { break }  

	if { $cnt > 1000 } { break } ;#- just in case
	incr cnt
    }
    return $vectorName
}


#----------------------------------------------------------------------------------------------------
# add another trace to the node selection dialog
#
proc sw_externaltrace_add { traceName xdata ydata { traceColour "" } } { 

    set traceName [ string tolower $traceName ] ;#- lowercase

   # test that trace name not used by spice 
    

    # test tracename is unique
    set traceVectorName [ sw_externaltrace_createUniqueVector ]
    set traceVectorTimeName "$traceVectorName.time"
    
    # create blt vector
    blt::vector create $traceVectorName
    blt::vector create $traceVectorTimeName 

    # load data into 
    $traceVectorName set $xdata
    $traceVectorTimeName set $ydata
 
    # trace colour - creates unique colour is none passed
    if { $traceColour == "" } { 
	set traceColour "red"
    } 

    # store traces in data holder
    lappend  ::spicewish_externalTraces "$traceName $traceVectorName $traceColour  $traceVectorTimeName"

} 


#--------------------------------------------------------------------------------------------------
#
#
proc sw_externaltrace_addFile { vecName fileName  { startLine ""} } {
    
    # creates a unique vectors using passed name
    set vecName [ string tolower $vecName ]
    set vectorName "$vecName"
    
    # test that file exists 
    if { ! [ file exists $fileName ] } { 
	puts "can't read file '$fileName' "
	return 
    }

    # test spice
    
    # test that vecName is not already a external vector
    set vectorList [ blt::vector names ]
    foreach vector $vectorList  {
	set vectorL [ string tolower $vector ] ;#- lowercase
	set nameL [ string tolower $vectorName ] ;#- lowercase
	if { [ string match $vectorL $nameL ] } { 
	    puts "vector '$vecName' already added as an external vector"
	    return
	}
    }
    
    # -- read file 
    set xdata ""
    set ydata ""
    set fileId [open $fileName r]
    
    set lineCnt 0
    while { ! [ eof $fileId ] } {
	gets $fileId linestring
	foreach { dTime dPeriod } $linestring { }
	if { $lineCnt > $startLine  } {
	    lappend xdata $dTime
	    lappend ydata $dPeriod
	}
	incr lineCnt
    }
    
    # -- save to external vector
    blt::vector create ::$vectorName
    blt::vector create ::$vectorName\_time
    ::$vectorName set $ydata
    ::$vectorName\_time set $xdata
} 



#========================================================
#========================================================
#        node selection dialog 
# sw_nodeselection_create 
# sw_nodeselection_clearTraces
# sw_nodeselection_markTraces
# sw_nodeselection_loadSelection
#
#========================================================
#========================================================

#---------------------------------------------------------------------------
# create a pop up dialog to select required nodes
#  - if the name of a viewer is passed then 
#      - the viewer traces will be highlighted 
#      - the viewer will be updated with the highlighted traces
#
proc sw_nodeselection_create { { viewer "" } } {

    # error traps
    if { $::spice::steps_completed < 1 } {  spice::step 1  } 
    if { $viewer != "" } { if { ! [ winfo exists $viewer ] } { return }  }
    

    set dialogName ".trace_selection_box"

    catch {destroy $dialogName }
    set w [ toplevel $dialogName ] 

    wm title $w "Select Traces"
    
    pack [ blt::tabnotebook $w.tnb ]  -fill both -expand 1 
        
    set tabList [ sw_spice_nodeTypes  ]
    
    #foreach tabs $tabList  
    for { set tabCount 0 } { $tabCount < [ llength $tabList ] } { incr tabCount } {
	set tabs [ lindex $tabList $tabCount ]
    
	$w.tnb insert end -text [ string totitle $tabs] 
	$w.tnb configure -tearoff 0
	
	frame $w.tnb.$tabs  ;#- frame for each tab 
	
	$w.tnb tab configure $tabCount -window $w.tnb.$tabs -fill both
	    
	# eval required due to use of '$tabs' in widgets names
	eval "blt::hierbox $w.tnb.$tabs.hierbox  -hideroot true -yscrollcommand { $w.tnb.$tabs.vert set } -xscrollcommand { $w.tnb.$tabs.horz set } -background white"
	eval "scrollbar $w.tnb.$tabs.vert -orient vertical -command { $w.tnb.$tabs.hierbox yview } -width 10"
	eval "scrollbar $w.tnb.$tabs.horz -orient horizontal -command { $w.tnb.$tabs.hierbox  xview } -width 10"
	
	# plot / update button
	if { $viewer == "" } {  set buttonText "Plot" 
	} else { set buttonText "Update \[ $viewer \]" }

	button $w.tnb.$tabs.b -text $buttonText  \
	    -command "sw_nodeselection_loadSelection $w $viewer"  

	eval "bind $w <KeyPress-Return> { sw_nodeselection_loadSelection $w }"  
	
	# table 
	blt::table $w.tnb.$tabs   \
	    0,0 $w.tnb.$tabs.hierbox  -fill both \
	    0,1 $w.tnb.$tabs.vert  -fill y \
	    1,0 $w.tnb.$tabs.horz  -fill x \
	    2,0 $w.tnb.$tabs.b -fill x -cspan 2
	
	blt::table configure $w.tnb.$tabs   c1 r1 r2  -resize none  
		
	# bind button 1 to the select the traces from the list
	$w.tnb.$tabs.hierbox bind all <Button-1> {
	    set hierbox_path %W
	    set index [  $hierbox_path index current]
	    
	    if { [ $hierbox_path entry cget $index -labelfont] == {times 10}} {
		$hierbox_path entry configure $index -labelfont {times 12 bold}
	    } else {
		$hierbox_path entry configure $index -labelfont {times 10}
	    }   
	}
    }

    # load nodes into dialog 
    foreach nodeInfo [ spice::spice_data ] {
	set node [ lindex $nodeInfo 0 ] 
	set type  [ lindex $nodeInfo 1 ] 
	
	if { [ string tolower $type ]  == "time" } { continue  } ;#- don't add the time vector
	set node [ string tolower $node ]
	$w.tnb.$type.hierbox insert -at 0 end $node -labelfont {times 10}     
    }    

    # if name of viewer to update is passed
    if { $viewer != "" } {	
	sw_nodeselection_markTraces $viewer $w
    }
} 


#--------------------------------------------------------------------------------------------
# clears all the highlighted traces in the node selection dialog
#
proc sw_nodeselection_clearTraces { w } { 
    
    # error traps 
    if { ! [ winfo exists $w ] } { return } 

    # loop through each tab
    foreach tabs [ sw_spice_nodeTypes ]  {
	# loop foreach spice node
	for {set i 1} { $i <= [ $w.tnb.$tabs.hierbox index end ] } {incr i} {
	    $w.tnb.$tabs.hierbox entry configure $i -labelfont {times 10}
	}
    }    
} 



#---------------------------------------------------------------------------------------------
# highlight traces from a viewer in node selection dialog
#
proc sw_nodeselection_markTraces { viewer w } {
    
    # error traps 
    if { ! [ winfo exists $w ] } { return } 
    
    # remove any highlighted traces 
    sw_nodeselection_clearTraces $w

    set traceList [ sw_viewer_traces $viewer ]

    foreach tabs [ sw_spice_nodeTypes ]  {
	
	# loop foreach trace 
	foreach trace $traceList { 
	    set nodeIndex [ $w.tnb.$tabs.hierbox find -name $trace ] 
	    if { $nodeIndex == "" } { continue }
	    
	    $w.tnb.$tabs.hierbox entry configure $nodeIndex -labelfont {times 12 bold}
	}
    }
} 


#----------------------------------------------------------------------------------------------
# load the highligted nodes in node selection dialog in to a new viewer
#
proc sw_nodeselection_loadSelection { w { viewer "" } } {

    set list ""
    foreach tabs [ sw_spice_nodeTypes ]  {
	for {set i 1} { $i <= [ $w.tnb.$tabs.hierbox index end ] } {incr i} {
	    
	    if {[ $w.tnb.$tabs.hierbox entry cget $i -labelfont] == {times 12 bold}} {
		lappend list [ $w.tnb.$tabs.hierbox get $i ]
	    }
	}
    }

    # error no traces seleted
    # if { [ llength $list ] == 0 } { return } 
    
    # if updating a viewer remove any deselected traces 
    if { $viewer != "" } { 
	set currentTraceList [ sw_viewer_traces $viewer ]
	foreach currentTrace $currentTraceList {
	    if { ! [ lcontain $list $currentTrace ] } {
		sw_remove $viewer $currentTrace
	    }
	}
	
    }

    # load nodes in to a new viewer
    eval "sw_plot $viewer $list" 
    
    catch { destroy $w }
}



#========================================================
#========================================================
#       spice
# sw_spice_validNode 
# sw_spice_nodeTypes
#
#========================================================
#========================================================

#----------------------------------------------------------------------------------
# tests if trace is a valid spice node 
#
proc sw_spice_validNode { trace } {
    
    if { $::spice::steps_completed <1 } { spice::step 1 } 
    
    set trace [ string tolower $trace ]
    set spiceNodeList  [ spice::spice_data ] ;#- all lowercase
    
    for { set i 0 } { $i< [ llength $spiceNodeList] }  { incr i } { 
	if { [ lcontain [ lindex [ lindex $spiceNodeList $i ] 0 ]  $trace] } { return 1 }
    } 
    return 0
} 



#------------------------------------------------------------------------------------
# returns a list of different types of nodes
#  - eg voltage current
#  - doesn't return time 
#
proc sw_spice_nodeTypes { } {
    
    if { $::spice::steps_completed <1 } { spice::step 1 } 
    
    set list [spice::spice_data] ;#- list of nodes
    
    set typeList ""
    # loop foreach vector
    foreach type $list { 
	set type [ lindex $type 1 ]
	set typeL [ string tolower $type ] 
	
	if { ! [ lcontain $typeList $typeL  ] } { 
	    if { $typeL == "time" } { continue }
	    lappend typeList $typeL 
	}  
    }

    return $typeList 
}


#========================================================
#========================================================
#       trace 
# sw_trace_createUniqueVector
# sw_trace_colour
# sw_trace_vectorUpdate
# sw_trace_type
#
#========================================================
#========================================================

#-----------------------------------------------------------------------
# create a unique blt vector name
#  - spice's node name carn't be used due to illegal characters eg '#'
#
proc sw_trace_createUniqueVector { viewer } {
    
    # error traps 
    if { ! [ winfo exists $viewer ] } { return "" } 

    set cnt 1
    while { 1 } {
	set vectorName "$viewer.trace$cnt"	
	if { [ blt::vector names ::$vectorName ] == "" } { break }  

	if { $cnt > 1000 } { break } ;#- just in case
	incr cnt
    }
    return $vectorName
}

#--------------------------------------------------------------------------------
# returns the colour associted with trace
#  - creates a colour if none assigned
#
proc sw_trace_colour { viewer trace } { 
        
    set default_colour(default) "grey"
    set default_colour(0) "red"
    set default_colour(1) "blue"
    set default_colour(2) "orange"
    set default_colour(3) "green"
    set default_colour(4) "magenta"
    set default_colour(5) "brown"
    
    # test if trace has already been assigend a colour
    if { [ info exists ::spicewish_plots($viewer) ] } {  
	for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $::spicewish_plots($viewer) ] } { incr dataHolderCnt } { 
	    set holder_traceInfo [ lindex $::spicewish_plots($viewer) $dataHolderCnt ] 
	    set holder_traceName [ lindex $holder_traceInfo 0 ]
	    if { $holder_traceName == $trace } { 
		return [ lindex $holder_traceInfo 2 ] 
	    }
	}   
    }

    # get the number of traces in teh viewer
    set traceIndex [ llength [ sw_viewer_traces $viewer ] ] 

    #create unique new colour
    if { [info exists default_colour($traceIndex)] } {
	set colour $default_colour($traceIndex)  ;#- if within number of traces
    } else { 
	# creates a random colour 
	set randred [format "%03x" [expr {int (rand() * 4095)}]]
	set randgreen [format "%03x" [expr {int (rand() * 4095)}]]
	set randblue [format "%03x" [expr {int (rand() * 4095)}]]
	set colour "#$randred$randgreen$randblue"
    }

    return $colour 
} 


#----------------------------------------------------------------------------------------------------
# update blt vector with current spice data
#  - if external added file just returns 
#
proc sw_trace_vectorUpdate { viewer traceName } {

    if { ! [ winfo exists $viewer ] } { return } 
    
    # test that data holder exists
    if { ! [ info exists ::spicewish_plots($viewer) ] } { return }    

    # lowercase
    set traceName [ string tolower $traceName ]

    # loop till trace found in data holder
    for { set traceCnt 0 } { $traceCnt < [ llength $::spicewish_plots($viewer) ] } { incr traceCnt } { 
	set traceInfo [ lindex $::spicewish_plots($viewer) $traceCnt ]
	
	set holder_traceName [ lindex $traceInfo 0 ]
	set holder_traceVector [ lindex $traceInfo 1 ]
	set holder_traceTimeVector [ lindex $traceInfo 3 ]

	if { $holder_traceName == $traceName } {
	   
	    if { [ sw_spice_validNode $traceName ] } { 
		# spice node
		spice::spicetoblt  $holder_traceName $holder_traceVector
		spice::spicetoblt time $holder_traceTimeVector

	    } else {
		# external node
		
		# do nothing 
	    }
	} 
    }
} 


#----------------------------------------------------------------------------------------------------
# return the traces type 
# either 
#  - spice
#  - externally added trace
#
proc sw_trace_type { traceName } {
    
    # search spice plot list 
    if {  [ sw_spice_validNode $traceName ] } {
	return "spice"
    }

    # search external node list 
    for { set extCnt 0 } { $extCnt < [ llength $::spicewish_externalTraces ] } { incr extCnt } { 
	set traceInfo [ lindex $::spicewish_externalTraces $extCnt ]
	set holder_traceName [ lindex $traceInfo 0 ]
	
	if { $traceName == $holder_traceName } {
	    return "external"
	} 
	
    }

    return "" 
} 


#========================================================
#========================================================
#       
# sw_plot
# sw_remove
# sw_update
#
#========================================================
#========================================================

#----------------------------------------------------------------------------------------------------
# adds traces to a viewer
#  - if first arg is the name of a viewer the traces are added to it,   else creates a new viewer
#   - add the trace to the viewers data holder 
#   - create a unique colour for the trace
#   - create a unique vector for the trace ( see sw_trace_createUniqueVector )
#   - format { traceName vectorName traceColour  timeVector} 
#
proc sw_plot { args } {
     
    # loop for each of the args
    for { set traceCnt 0 } { $traceCnt < [ llength $args ] } { incr traceCnt }  {
	set trace [ lindex $args $traceCnt ] 

	# check if first arg is the name of viewer to update
	if { $traceCnt == 0 } { 
	    if { [ winfo exists $trace ] } { 
		set viewer $trace 
		continue
	    } else {
		set viewer [ sw_viewer_create  ] 
	    }
	}

	set traceName [ string tolower $trace ] ;#- lowercase to match spice 

	switch [ sw_trace_type $traceName ]  { 
	    
	    "" { 
		# not valid trace 
		#puts "$trace not valid "
	    }
	    
	    "spice" {
		
		# test that data holder exists
		if { !  [ info exists ::spicewish_plots($viewer) ] } {    set ::spicewish_plots($viewer) ""  }    
			
		# test that trace hasn't been prevously added
		if { [ lsearch [ sw_viewer_traces $viewer ] $traceName ] != -1 } { 
		    #puts "trace $traceName already added "
		    continue }
		
		# generate trace colour 
		set traceColour [ sw_trace_colour $viewer $traceName ]
		
		# get unique vector name 
		set traceVectorName [ sw_trace_createUniqueVector $viewer ] 
		set traceTimeVectorName "$viewer.time"
		
		blt::vector create $traceVectorName
		blt::vector create $traceTimeVectorName  ;# make sure that time vector exists 
	    }

	    "external" {
		
		# loop to find trace in external data holder
		for { set extCnt 0 } { $extCnt < [ llength $::spicewish_externalTraces ] } { incr extCnt } { 
		    set traceInfo [ lindex $::spicewish_externalTraces  $extCnt ]
		    set holder_traceName        [ lindex $traceInfo 0 ]
		    
		    if { $traceName == $holder_traceName } { 
			set traceVectorName         [ lindex $traceInfo 1 ]
			set traceColour                   [ lindex $traceInfo 2 ]
			set traceTimeVectorName [ lindex $traceInfo 3 ]
		    }	    
		}
	    }
	    
	}
	# add trace to the data holder
	lappend ::spicewish_plots($viewer) "$traceName $traceVectorName $traceColour $traceTimeVectorName"
    }

    # update the viewer
    sw_update $viewer

    return $viewer
} 


#-----------------------------------------------------------------------------------------------
# removes traces from a viewer
#  - removes trace from the viewers data holder
#  - removes trace form the viewer
#  - delete the traces unique blt vector 
#
proc sw_remove { viewer args } {

    if { ! [ winfo exists $viewer ] } { return } 

    # test that data holder exists
    if { ! [ info exists ::spicewish_plots($viewer) ] } { return }    
    
    # loop for each of the passed traces
    foreach trace $args {

	set trace [ string tolower $trace ] ;#- lowercase

	# search through the data holder for the trace 
	for { set dataHolderCnt 0 } { $dataHolderCnt < [ llength $::spicewish_plots($viewer) ] } { incr dataHolderCnt } { 
	    set traceInfo [ lindex $::spicewish_plots($viewer) $dataHolderCnt ] 
	    set holder_traceName [ lindex $traceInfo 0 ]
	    set holder_traceVector [ lindex $traceInfo 1 ]
	    
	    if { $holder_traceName == "" } { continue } 
	    
	    if { $trace == $holder_traceName } { 
		# remove from data holder
		set ::spicewish_plots($viewer) [ lreplace $::spicewish_plots($viewer) $dataHolderCnt $dataHolderCnt  ]
		
		#remove from graph 
		if { [ $viewer.g element exists $trace ] } { $viewer.g element delete $trace } 
		
		# delete vector 
		blt::vector destroy $holder_traceVector
	    }   
	}    
	
	# remove trace from traceplace grid
	sw_traceplace_remove $viewer $trace
    }
} 

#----------------------------------------------------------------------
# updates the blt vectors for all the traces in a viewer
#  - if no args passed then updates up all the viewer present
#
proc sw_update { args } {
  
    if { $args == "" }  {
	set viewersList [ sw_viewer_names ]
    } else {  
	set viewersList $args
    }

    # loop for each of the viewer to update
    foreach viewer $viewersList {
	
	# test that data holder exists
	if { !  [ info exists ::spicewish_plots($viewer) ] } { continue  } 

	# flag to update spice time
#	set spiceTimeUpdate 0

	# loop for each of the traces in the viewer
	for { set traceCnt 0 } { $traceCnt < [ llength $::spicewish_plots($viewer) ] } { incr traceCnt } { 
	    set traceInfo [ lindex $::spicewish_plots($viewer) $traceCnt ]

	    set traceName [ lindex $traceInfo 0 ]
	    set traceVector [ lindex $traceInfo 1 ]
	    set traceColour [ lindex $traceInfo 2 ]
	    set timeVector [ lindex $traceInfo 3 ]

	    if { $traceName == "" } { continue  }
	    
	    
	    # update blt vector
	    #spice::spicetoblt  $traceName $traceVector
	    sw_trace_vectorUpdate $viewer $traceName


	    # test that the Trace has been added to the graph 
	    if { ! [ $viewer.g element exists $traceName ] } { 
		$viewer.g  element create $traceName \
		    -ydata $traceVector \
		    -xdata $timeVector \
		    -symbol "circle" \
		    -linewidth 2 \
		    -pixels 3 \
		    -color $traceColour
	    }
	    
	} 
#	# update the spice time vector
#	spice::spicetoblt time $timeVector
    }
  
    # update traceplace traces is grid enabled
    foreach viewer $viewersList {
	if { $::spicewish_tp($viewer) } {
	    sw_traceplace_slotUpdate	$viewer
	}
    }
} 




#========================================================
#========================================================
#       spice
# sw_run
# sw_halt 
#========================================================
#========================================================

#-------------------------------------------------------------------------------------------------------
# runs the simualtion
#
proc sw_run { { run_untill "" } } {
    
    if { $run_untill != "" } { 
	spice::stop when time > $run_untill 
    }

    if { $::spice::steps_completed < 1  } {
	spice::bg run 
    } else  {
	spice::bg resume 
    }
    if { [ winfo exists .controls.brun ] } { .controls.brun configure -text "Resume" }
} 


#-------------------------------------------------------------------------------------------------------
# halts the simulation
#
proc sw_halt { } {

    spice::halt 
    sw_update ;#- updates all the viewers
} 



#========================================================
#========================================================
#   controls
# sw_controls_create 
# sw_controls_exit
# sw_controls_editor_lineNumber
# sw_controls_editor_popupMenu 
# sw_controls_editor_active_editors
# sw_controls_editor_create_statusbar
# sw_controls_editor_create
# sw_controls_editor_dragDropPacket 
# sw_controls_editor_toggle
# sw_controls_editor_loadFile 
# sw_controls_editor_revertBuffer
# sw_controls_editor_saveFile 
# sw_controls_editor_includeFiles
# sw_controls_editor_syntaxHighlighting_toggle
# sw_controls_editor_syntaxHighlighting
#
#========================================================
#========================================================

#------------------------------------------------------------------------------
# main control dialog 
#
proc sw_controls_create { fileName } {
    
    if { [ winfo exists .controls ] } { 
	# SpiceWish being initialized again 
	wm deiconify .  ;# raise main console 
	
	# re source spice file 
	spice::halt
	spice::stop
	spice::reset 
	spice::source $fileName
	.controls.brun configure -text "Run"
	return 
    }

    set ::spicewish_globals(fileName) $fileName
    spice::source $fileName

    # control buttons
    frame .controls
    pack [ button .controls.brun -text "Run" -command { sw_run } -width 8 ] -fill x
    pack [ button .controls.bhalt -text "Halt" -command { sw_halt } ] -fill x
    pack [ button .controls.bplot -text "Plot" -command { sw_nodeselection_create } ] -fill x
    pack [ button .controls.bviewer -text "Editor" -command { sw_controls_editor_toggle } ] -fill x
    pack [ button .controls.bexit -text "Exit" -command { sw_controls_exit } ] -fill x
    
    # create spice file editor widget + load file + highlight paremeters
    set editor_w [ sw_controls_editor_create $fileName ]

    #   sw_controls_editor_loadFile $editor_w $fileName
    #   sw_controls_editor_syntaxHighlighting 
    #   $editor_w.text configure -state disabled ;#- disabled no editting for now

    # table
    blt::table . \
	0,0 .controls -anchor n -fill x 
    blt::table configure . c0 -resize none

    # set editor to initial condition
    if { $::spicewish_globals(editor) } { 
	blt::table . 	0,1 $editor_w -fill both 
    }

    # window title 
    wm title . "spicewish - $fileName"

    # window dimensions
    update
    wm geometry . "90x[ expr [ winfo height .controls ] + 10 ]"

    wm deiconify .  ;#- required for magic use


    # keyboard shortcuts
    if {1} {
	bind . <KeyPress-r> { sw_run }
	bind . <KeyPress-h> { sw_halt }
	bind . <KeyPress-p> { sw_nodeselection_create }
	bind . <KeyPress-u> { sw_update }
	bind . <KeyPress-e> { sw_controls_editor_toggle }
    }
}


#---------------------------------------------------------------------------------------------
# exits spicewish
# if using within magic then just the SpiceWish widgets are destroyed with out using exit
#
proc sw_controls_exit { } {

    set widget_list [ winfo children . ]

    # test if tk console ( .tkcon ) is being used 
    if [ info exists ::CAD_HOME ] {

	# kill only the spicewish based windows
	foreach winname $widget_list  {
	    # viewer windows
	    if {  [ regexp {.tclspice[0-9]+} $winname ] } {
		wm withdraw $winname
	    } 
	    if { $winname == ".controls" } {  wm withdraw "."  }  
	 }
    } else {
	exit
    }
} 

#---------------------------------------------------------------------------------------------
# sets the line number at the bottom of the status bar 
#
proc sw_controls_editor_lineNumber { w } {
    
    set lineIndex [ $w index insert ] 
    set lineIndex [ split $lineIndex . ]
	   
    set lineNum [ lindex $lineIndex 0 ]
    set colNum [ lindex $lineIndex 1 ] 
    
    set totalLineNum [ expr [ $w index end ] - 1 ]
    
    set percentage [ expr round( ( $lineNum / $totalLineNum ) * 100 ) ] 
    switch $percentage { 
	"0"       { set ::sw_editor_lineInfo($w) "L${lineNum}--C${colNum}--Top" }
	"100"   { set ::sw_editor_lineInfo($w) "L${lineNum}--C${colNum}--Bot" }
	default { set ::sw_editor_lineInfo($w) "L${lineNum}--C${colNum}--${percentage}%" }
    }
} 


#-----------------------------------------------------------------------------------------------
# displays a pop up options dialog inside text window 
#
proc sw_controls_editor_popupMenu { w x y { _x ""} { _y ""} } { 
 

    # test if the cursor is mositioned over a selected piece of text
    # piece of text must be a valid spice node as well
    set sel_coords [ $w tag ranges sel ]
 
    if { [ $w tag ranges sel ] != "" } {
    	
	set node [ $w get [lindex $sel_coords 0] [lindex $sel_coords 1] ]
	
	# test if valid node
	if { [ sw_spice_validNode $node ] } {  

	    set coords_first [ $w bbox [lindex [ $w tag ranges sel ] 0 ] ]
	    set coords_sec [ $w bbox [lindex [ $w tag ranges sel ] 1 ] ]
	    set sel_x1 [lindex $coords_first 0]
	    set sel_y1 [lindex $coords_first 1]
	    set sel_x2 [expr [lindex $coords_sec 0] + [lindex $coords_sec 2] ]
	    set sel_y2 [expr [lindex $coords_sec 1] + [lindex $coords_sec 3] ]
	    
	    if { ($_x > $sel_x1) && ( $_x < $sel_x2 ) && ($_y > $sel_y1) && ( $_y < $sel_y2 ) } { return }
	}
    }


    catch {destroy .m}                     ;# destroy old pop up      
    menu .m -tearoff 0
    
    .m add command -label "Drag and drop" -state disabled

    .m add command -label "New viewer" -command "sw_viewer_create"
    .m add checkbutton -label "Syntax highlighting" \
	-variable spicewish_globals(editor_syntaxHighlight) \
	-command "sw_controls_editor_syntaxHighlighting_toggle"

    .m add checkbutton -label "Status bar" \
	-variable spicewish_globals(editor_statusbar) \
	-command "sw_controls_editor_create_statusbar"

    .m add command -label "Save" -command "sw_controls_editor_saveFile $w" 
    .m add command -label "Revert Buffer " -command "sw_controls_editor_revertBuffer $w"
    
    tk_popup .m $x $y 

}
 
#----------------------------------------------------------------------------------------
# returns a list of all the open editors (tabs)
#
proc sw_controls_editor_active_editors { } {
    
    set editors_list ""

    if { $::spicewish_globals(include_fileNames) == "" } { 
	set editors_list ".editor"
    } else { 
	set editors_list ".editor.tnb.0"
	for {set i 1} {$i <= [ llength $::spicewish_globals(include_fileNames) ]} {incr i } {
	    set editors_list "$editors_list .editor.tnb.$i"
	} 	
    } 
    
    return $editors_list 
}


#-----------------------------------------------------------------------------------------
# generates the status bars for the bottom of each  text editor
# no arg passed will toggle staus bars on and off depending on $::spicewish_globals(editor_statusbar)
#
proc sw_controls_editor_create_statusbar { { w "" } } {
    
     # toggles passed .editor.text 

    if { $w != "" } { 
	if { ! [ winfo exists $w.status ] } { 
	    # status bar 
	    frame $w.status -relief sunken -borderwidth 1

	    # - line number
	    set ::sw_editor_lineInfo($w.text) "L0--C0--Top" 
	    pack [ frame $w.status.line -relief sunken -borderwidth 1 ]  -fill x -expand 1
	    pack [ label $w.status.line.lv -textvariable sw_editor_lineInfo($w.text) ] -side left

	    # - line number - bindings 
	    bind $w.text <Button> { sw_controls_editor_lineNumber %W } 
	    bind $w.text <ButtonRelease> { sw_controls_editor_lineNumber %W }
	    bind $w.text <KeyRelease> { sw_controls_editor_lineNumber %W }
	} 
    } else {
	set w [ sw_controls_editor_active_editors ]
    }
    
    # pack/forget status bar depending on $::spicewish_globals(editor_statusbar)
    foreach _w $w { 
	if { $::spicewish_globals(editor_statusbar) } { 
	    # pack status bar 
	    blt::table $_w 2,0 $_w.status -fill x -cspan 2 
	} else { 
	    # unpack status bar 
	    blt::table forget $_w.status
	}
    }
    
    return $w.status    
} 


#----------------------------------------------------------------------------
# creates a text widget to view the spice file(s)
#
proc sw_controls_editor_create { fileName } {
    
    # editor dimensions
    set editorWidth 568
    set editorHeight 344
    
    set includeFiles [ sw_controls_editor_includeFiles $fileName ]

    set ::spicewish_globals(include_fileNames) $includeFiles
    

    #create text widget with sccrollbars
    set w [ frame .editor ]
    set w_s $w
    # create a tabset if any include
    if { [ llength $includeFiles ] != 0 } { 
    
	pack [ blt::tabnotebook $w.tnb ]  -fill both -expand 1 
        
	set tabList "$fileName $includeFiles"
	
	#foreach tabs $tabList  
	for { set tabCount 0 } { $tabCount < [ llength $tabList ] } { incr tabCount } {
	    
	    set w $w_s
	    set tabs [ lindex $tabList $tabCount ]
	    $w.tnb insert end -text $tabs 
	    $w.tnb configure -tearoff 0
	    frame $w.tnb.$tabCount  ;#- frame for each tab 
	    
	    $w.tnb tab configure $tabCount -window $w.tnb.$tabCount -fill both
	    set w "$w.tnb.$tabCount"

	    text $w.text -background LemonChiffon 
	    
	    eval "$w.text configure -yscrollcommand { $w.y set } -xscrollcommand { $w.x set }"
	    eval "scrollbar $w.y -command { $w.text yview } -orient vertical -width 10"
	    eval "scrollbar $w.x -command { $w.text xview } -orient horizontal -width 10"
	    
	    # mouse wheel bindings
	    bind $w.text <Button-5>               [list %W yview scroll 5 units]
	    bind $w.text <Button-4>               [list %W yview scroll -5 units]
	    bind $w.text <Shift-Button-5>       [list %W yview scroll 1 units]
	    bind $w.text <Shift-Button-4>       [list %W yview scroll -1 units]
	    bind $w.text <Control-Button-5>  [list %W yview scroll 1 pages]
	    bind $w.text <Control-Button-4>  [list %W yview scroll -1 pages]
	    bind $w.text <KeyPress-Next>    [list %W yview scroll 5 units]
	    bind $w.text <KeyPress-Prior>    [list %W yview scroll -5 units]
	    
	   
	    # table
	    blt::table $w \
		0,0 $w.text -fill both \
		0,1 $w.y -fill y \
		1,0 $w.x -fill x 

	    # add status bar 
	    set w_status [ sw_controls_editor_create_statusbar $w ]
	    
	    blt::table configure $w c1 r1 r2  -resize none
	
	    # create a drag a drop packet (source) 
	    eval "blt::drag&drop source $w.text -packagecmd { sw_controls_editor_dragDropPacket %t %W } -button 3"
	    set token [blt::drag&drop token $w.text -activebackground blue ]
	    pack [ label $token.label -text "" ]
	    
	    # popup menu 
	    bind $w.text <Button-3> { sw_controls_editor_popupMenu %W %X %Y %x %y }

	    # load file + add syntax highlighting 
	    sw_controls_editor_loadFile $w $tabs
	    sw_controls_editor_syntaxHighlighting  $w.text

	    # $w.text configure -state disabled ;#- disabled no editting for nowDOUTDOUT
	}
	
	return $w

    } else {
	text $w.text -background LemonChiffon 
	
	eval "$w.text configure -yscrollcommand { $w.y set } -xscrollcommand { $w.x set }"
	eval "scrollbar $w.y -command { $w.text yview } -orient vertical -width 10"
	eval "scrollbar $w.x -command { $w.text xview } -orient horizontal -width 10"
	
	# mouse wheel bindings
	bind $w.text <Button-5>               [list %W yview scroll 5 units]
	bind $w.text <Button-4>               [list %W yview scroll -5 units]
	bind $w.text <Shift-Button-5>       [list %W yview scroll 1 units]
	bind $w.text <Shift-Button-4>       [list %W yview scroll -1 units]
	bind $w.text <Control-Button-5>  [list %W yview scroll 1 pages]
	bind $w.text <Control-Button-4>  [list %W yview scroll -1 pages]
	bind $w.text <KeyPress-Next>    [list %W yview scroll 5 units]
	bind $w.text <KeyPress-Prior>    [list %W yview scroll -5 units]
	    
	
	# table
	blt::table $w \
	    0,0 $w.text -fill both \
	    0,1 $w.y -fill y \
	    1,0 $w.x -fill x 
	
	# add status bar 
	set w_status [ sw_controls_editor_create_statusbar $w ]
	

	blt::table configure $w c1 r1 r2 -resize none

	# create a drag a drop packet (source) 
	eval "blt::drag&drop source $w.text -packagecmd { sw_controls_editor_dragDropPacket %t %W } -button 3"
	set token [blt::drag&drop token $w.text -activebackground blue ]
	pack [ label $token.label -text "" ]

	# popup menu 
	bind $w.text <Button-3> { sw_controls_editor_popupMenu %W %X %Y %x %y}
	

	# load file + add syntax highlighting
	sw_controls_editor_loadFile $w $fileName
	sw_controls_editor_syntaxHighlighting  $w.text

	#$w.text configure -state disabled ;#- disabled no editting for now
    }

    return $w
} 


#-----------------------------------------------------------------------------
# creates a drag and drop packet for 
#
proc sw_controls_editor_dragDropPacket { token widget } {
    

    # current cursor position in text box
    set _x  [ expr [winfo pointerx $widget ] - [winfo rootx $widget] ]
    set _y  [ expr [winfo pointery $widget ] - [winfo rooty $widget] ]
 
    # retrive marked nodes string 
    set sel_coords [ $widget tag ranges sel ] 

    if { $sel_coords == "" } { return } ;#- returns when no text is selected 

    # make sure that the  cursor is positioned over the selected nodes text  
    # bbox   x y width height
    set coords_first [ $widget bbox [lindex [ $widget tag ranges sel ] 0 ] ]
    set coords_sec [ $widget bbox [lindex [ $widget tag ranges sel ] 1 ] ]
    set sel_x1 [lindex $coords_first 0]
    set sel_y1 [lindex $coords_first 1]
    set sel_x2 [expr [lindex $coords_sec 0] + [lindex $coords_sec 2] ]
    set sel_y2 [expr [lindex $coords_sec 1] + [lindex $coords_sec 3] ]
    
    if { ! (($_x > $sel_x1) && ( $_x < $sel_x2 ) && ($_y > $sel_y1) && ( $_y < $sel_y2 )) } { return }


    set node [ $widget get [lindex $sel_coords 0] [lindex $sel_coords 1] ]
    
    # exit if selected text is not node in spice deck
    if { ! [ sw_spice_validNode $node ] } { return } 
    
    # create token ( drag drop packet )
    $token.label configure -text $node  -background "grey"
    
    set node [ string tolower $node ] 

    return "{$node}"
}


#-----------------------------------------------------------------------------
# toggles the editor widget on and off 
#  - if a 1 or 0 is passed then the widget is set to that state
#
proc sw_controls_editor_toggle { { mode "" } } {
    
    # editor dimensions
    set editorWidth 568
    set editorHeight 344

    # decide the display state of the editor widget
    if { $mode != "" } {
	if { ( $mode == 1 ) || ( $mode == 0 ) } {
	    set ::spicewish_globals(editor) $mode
	} else { return } 
    } else {
	if { $::spicewish_globals(editor) } { 
	    set ::spicewish_globals(editor) 0
	} else { 
	    set ::spicewish_globals(editor) 1
	}
    }
    
    # enable / disable the editor widget
    if { $::spicewish_globals(editor) } {
	# enable editor
	blt::table . 0,1 .editor -fill both 
	update
	set width [ expr 80 + $::spicewish_globals(editor_width) ]
	set height  $::spicewish_globals(editor_height)
	wm geometry . "$width\x$height"
    } else {
	# disable editor
	set ::spicewish_globals(editor_width) [ winfo width .editor ] ;#- grab width of editor window if resized
	set ::spicewish_globals(editor_height) [ winfo height .editor ] ;#- grab height
	blt::table forget .editor
	blt::table . 0,0 .controls -fill x
	update
	wm geometry . "90x[ expr [ winfo height .controls ] + 10 ]"
    }
} 

#--------------------------------------------------------------------------------
# loads spice file into editor
#
proc sw_controls_editor_loadFile { w fileName} {
  
    
    if { ! [ file exists $fileName]  } { return }
    
    set w $w.text
    set f [open $fileName]
    $w delete 1.0 end
    while { ! [eof $f] } {
	$w insert end [read $f 10000] 
    }
    close $f
    
}

#-------------------------------------------------------------------------------
# reverts text editor back to buffer 
#
proc sw_controls_editor_revertBuffer { w } {
    
    set fileNames "$::spicewish_globals(fileName) $::spicewish_globals(include_fileNames)"
    
    if { [ llength $fileNames ] == 1 } { 
	# single file open ( no .includes )
	set fileName [lindex $fileNames 0]
	set f [open $fileName]
	$w delete 1.0 end
	while { ! [eof $f] } {
	    $w insert end [read $f 10000] 
	}
	close $f

	# reapply syntax highlighting 
	sw_controls_editor_syntaxHighlighting $w
	
	
    } else {
	# multiple files open ( .includes )
	for { set i 0 } { $i < [ llength $fileNames ] } { incr i } {
	    set w ".editor.tnb.${i}.text" 
	    set fileName [ lindex $fileNames $i ] 
	    set f [open $fileName]
	    $w delete 1.0 end
	    while { ! [eof $f] } {
		$w insert end [read $f 10000] 
	    }
	    close $f

	    # reapply syntax highlighting 
	    sw_controls_editor_syntaxHighlighting $w
	}
    }
    

}

#-------------------------------------------------------------------------------
# saves text editor to file 
#
proc sw_controls_editor_saveFile { w } {

    set fileNames "$::spicewish_globals(fileName) $::spicewish_globals(include_fileNames)"
    
    if { [ llength $fileNames ] == 1 } { 
	# single file open ( no .includes )
	set fileName [lindex $fileNames 0]

	# check write permission
	if { ! [ file writable $fileName ] } {
	    tk_messageBox -title "Write permission" -icon error -message "'$fileName' is not writable"
	} else { 
	    
	    set data [$w get 1.0 {end -1c}]
	    set fileid [open $fileName w]
	    puts -nonewline $fileid $data
	    close $fileid
	    puts "Saved file '$fileName'"
	}
    } else {
	# multiple files open ( .includes )
	for { set i 0 } { $i < [ llength $fileNames ] } { incr i } {
	    set fileName [ lindex $fileNames $i ] 

	    if { ! [ file writable $fileName ] } {
		tk_messageBox -title "Write permission" -icon error -message "'$fileName' is not writable"
	    } else { 

		set w ".editor.tnb.${i}.text" 	
		set data [$w get 1.0 {end -1c}]
		set fileid [open $fileName w]
		puts -nonewline $fileid $data
		close $fileid
		puts "Saved file '$fileName'"
	    }
	}
    }
} 

#-------------------------------------------------------------------------------
# returns a list of .include files in the sourced file 
# 
proc sw_controls_editor_includeFiles { fileName } { 

    if { ! [ file exists $fileName ] }  { return }

    set fileid [ open $fileName "r" ]  
    set filedata [read $fileid]
    set lines [split $filedata "\n"] 
    
    set includeFileList ""

    foreach line $lines { 

	set comLine [ string range $line 0  [ expr [ string first " " $line ] -1  ] ] 
	set comLine [ string tolower $comLine ]
	if { $comLine == ".include" } {

	    set includeFile [ string range $line [ expr [ string first " " $line ] +1 ]  end ]

	    # strip away spaces around file name
	    while { [string first " " $includeFile ] != -1 } {
		set includeFile [ string replace $includeFile [string first " " $includeFile ] [string first " " $includeFile ] ]
	    }

	    # test if file already in list
	    if { ! [ lcontain $includeFileList $includeFile ] } { 
		lappend includeFileList $includeFile
	    }
	} 
    }  
    
    
    return $includeFileList 
} 


#--------------------------------------------------------------------------------
# toggles the syntax highlighting on and off
#
proc sw_controls_editor_syntaxHighlighting_toggle { } {
    
    # work out the number of files open  ( tabs )
    set _w [ sw_controls_editor_active_editors ]
    
    foreach w $_w { 
	if { $::spicewish_globals(editor_syntaxHighlight) } { 
	    # re highlight all nodes
	    sw_controls_editor_syntaxHighlighting $w.text
	} else { 
	    # delete all tags 
	    eval "$w.text tag delete [ $w.text tag names ]"      
	}
    }
}


#--------------------------------------------------------------------------------
# highlight in different coolour the different parameters
#
proc sw_controls_editor_syntaxHighlighting { w } {


    if { ! $::spicewish_globals(editor_syntaxHighlight) } { return } ;#- syntax highlighting disabled 

    #set w "$w"
    
    # number of lines
    scan [$w index end] %d nl
    
    # tag names
    set tagName "parameters"

    # loop foreah line
    set lastLineColour ""
    set count 0
    while { $count < $nl } { 
	set lineText [$w get "$count.0 linestart" "$count.0 lineend"] 
	set firstChar [ string range $lineText 0 0 ]
	set firstString [ string range $lineText 0 [ string first " " $lineText ] ]

	switch $firstChar {
	    "" {
		# blank line 
		set lastLineColour ""
	    }
	    "+" {
		# continuation line
		if { $lastLineColour != "" } {
		    $w tag add $tagName.$count "$count.0 linestart" "$count.0 lineend"
		    $w tag configure $tagName.$count -foreground $lastLineColour
		}
	    }
	    "C" - "c" {
		set lineColour "red"
		$w tag add $tagName.$count "$count.0 linestart" "$count.0 lineend"
		$w tag configure $tagName.$count -foreground $lineColour
		set lastLineColour $lineColour
	    }
	    "M" - "m" { 
		set lineColour "green"
		$w tag add $tagName.$count "$count.0 linestart" "$count.0 lineend"
		$w tag configure $tagName.$count -foreground $lineColour
		set lastLineColour $lineColour
	    }
	    "V" - "v" {
		set lineColour "blue"
		$w tag add $tagName.$count "$count.0 linestart" "$count.0 lineend"
		$w tag configure $tagName.$count -foreground $lineColour
		set lastLineColour $lineColour
	    }
	    "L" - "l" {
		set lineColour "orange"
		$w tag add $tagName.$count "$count.0 linestart" "$count.0 lineend"
		$w tag configure $tagName.$count -foreground $lineColour
		set lastLineColour $lineColour
	    }
	    "R" - "r" {
		set lineColour "purple"
		$w tag add $tagName.$count "$count.0 linestart" "$count.0 lineend"
		$w tag configure $tagName.$count -foreground $lineColour
		set lastLineColour $lineColour
	    }
	    "." {
		switch $firstString {
		    ".MODEL " - ".model " {
			set lineColour "darkgreen"
			$w tag add $tagName.$count "$count.0 linestart" "$count.0 lineend"
			$w tag configure $tagName.$count -foreground $lineColour
			set lastLineColour $lineColour
		    }
		}
		
	    }
	} ;#- end switch $firstChar
	incr count 
    }
} 


#--------------------------------------------------------------------------------
# export main proc to namespace spicewish::
#
namespace eval spicewish {

    proc run { args}                         { eval "sw_run $args" } 
    proc halt { }                                { sw_halt }
    proc plot { args }                        { eval "sw_plot $args" }
    proc remove { args }                  { eval "sw_remove $args" }
    proc update { args }                   { eval "sw_update $args" }
    proc tp { viewer mode }              { eval "sw_traceplace_mode $viewer $mode" }
    proc tp_add { viewer slot args } { eval "sw_traceplace_add  $viewer $slot $args" }
    proc tp_remove { viewer args }  { eval "sw_traceplace_remove $viewer $args" }
    proc init_gui { fileName }             { eval "sw_controls_create $fileName" } 
    proc exit { }                                 { sw_controls_exit }
    proc version { }                           { puts " $::spicewish_globals(version)\n" } 

}





#-------------------------------------------------------------------------------------------
#
#
proc sw_float_eng { num {forceunit ""} } {
    
    set significant_digits 4 ;#- mininum signficant digits to show - might show more e.g.  125 will be shown no matter what
    #lookup hashes
    set table_eng_float_MSC(T) 1e+12
    set table_eng_float_MSC(G) 1e+9
    set table_eng_float_MSC(M) 1e+6
    set table_eng_float_MSC(K) 1e+3
    set table_eng_float_MSC(\n) 1; # Will need to change?
    set table_eng_float_MSC(m) 1e-3
    set table_eng_float_MSC(u) 1e-6
    set table_eng_float_MSC(\u03BC) 1e-6 
    set table_eng_float_MSC(n) 1e-9
    set table_eng_float_MSC(p) 1e-12
    set table_eng_float_MSC(f) 1e-15
    set table_eng_float_MSC(a) 1e-18
    
    #inverse hash  
    set table_float_eng_MSC(1e+12)" T"
    set table_float_eng_MSC(1e+9) " G"
    set table_float_eng_MSC(1e+6) " M"
    set table_float_eng_MSC(1e+3) " K"
    set table_float_eng_MSC(1)    " "
    set table_float_eng_MSC(1e-3) " m"
    set table_float_eng_MSC(1e-6) " \u03BC"
    set table_float_eng_MSC(1e-9) " n"
    set table_float_eng_MSC(1e-12) " p"
    set table_float_eng_MSC(1e-15) " f"
    set table_float_eng_MSC(1e-18) " a" 
    
    
    set indexes [lsort -real -decreasing "[array names table_float_eng_MSC]"]
    
    set out {}; # If no suffix found, return number as is
    
    if "$num != 0" {
	foreach index $indexes {
	    
	    set newnum "[expr {$num / $index}]"
	    set whole 0
	    set fraction 0; # In case of integer input
	    
	    # regexp {^(\d+)} $newnum whole; # Messy!
	    # Uses stupid variable to save having to do two regexp's
	    regexp {^(\d+)\.*(\d*)} $newnum stupid whole fraction
	    
	    if {[expr {[string match {*e*} $newnum] == 0}] && \
		    [expr {$whole >0 }] } {
		set suf "$table_float_eng_MSC($index)" 
		
		#work out how many decimal places to show - fixed to 4 significant digits
		set decimal_length_to_show [ expr ( ($significant_digits - 1 ) - [ string length $whole ] ) ]
		if { $decimal_length_to_show >= 0 } {
		    
		    if { $fraction == "" } {
			set newnm $whole
		    } else {
			#need some decimal places to give the resolution required
			set newnum "$whole.[string range $fraction 0 [ expr ($decimal_length_to_show)] ]"
		    }
		    
		} else {
		    #no decimal places
		    set newnum $whole  ;#- just use the whole, no decimals
		}
		
		set out "$newnum$suf"
		return "$out"
		break
	    } else {
		#puts "[expr {fmod($newnum,1)}] is not big enough"
	    }
	}
    }
    if {[string equal $out {}]} {
	#puts {Appropriate suffix not found}
	return $num
    }
} 


#----------------------------------------------------------------------------------------
#
#   
proc sw_plot_command_readline_completer { word start end line } {
    
    set match {} ;#- no match found yet
    set shortest_string_len 1000 ;#- will always get shorter
    set matches 0 ;#- totaliser for matches
    
    #check for "plot" command line
    
    if { [ regexp {\s+sw_plot} " $line " ]   ||  [  regexp {\s+spicewish\:\:plot} " $line " ] } {
	#found plot near the start of the line
	
	#check if punter is asking for a variable
	if { $start >= 5 } {
	    
	    #-find the last item typed so far
	    set arglist [split $line " "]
	    set lastarg [lindex $arglist end]
	    
	    #-search against the possible spice variables
	    set possibility_list {}	   
	    
	    set varlist [spice::plot_variables 0]  ;#- get current spice traces
	    
	    foreach possibility $varlist {
		#check for exact match
		if {$lastarg == $possibility} { 
		    set possibility_list $possibility
		    break
		}
		
		#check for close match
		if { [string match "${lastarg}*" $possibility] } {
		    lappend possibility_list $possibility ;#- add to list
		    incr matches ;#-add another match
		    #update shortest len
		    if { [string length $possibility] < $shortest_string_len } {
			set shortest_string_len  [string length $possibility]
		    }
		} 				
	    }
	    
	    #check for exact match
	    if { $lastarg == $possibility } { set match $lastarg } else { set match "" }
	    
	    #for multiple matches, find the longest common bit of string to return  readline style
	    #-loop for each character
	    set done 0
	    for {set charindex 0} { $charindex < $shortest_string_len } { incr charindex } {
		
		#-get char from one string
		set char [string index [lindex $possibility_list 0] $charindex]
		
		#check in each of other strings for same character
		for {set cnt 1} {$cnt < $matches} {incr cnt} {
		    
		    set char_alt [string index [lindex $possibility_list $cnt] $charindex]
		    
		    if {$char != $char_alt } {
			#found a mismatch, so exit now
			set done 1
			break
		    }
		    if { $done == 1} { break }
		}
		if { $done == 1} { break }
	    }
	    
	    #return the longest common portion which matches 
	    set common_bit [string range [lindex $possibility_list 0] 0 [expr $charindex-1] ]
	    
	    #sort the return values into alphabetic order
	    set possibility_list [lsort -dictionary $possibility_list]
	    
	    #return parameters for the readline
	    return [list $common_bit $possibility_list]  ;#
	    
	} ;#- end of if starting at the right place
	
    } ;#- end of if got plot in the line
    
    ::tclreadline::ScriptCompleter  $word $start $end $line    
}

::tclreadline::readline customcompleter  "sw_plot_command_readline_completer" ;#- see routine above







