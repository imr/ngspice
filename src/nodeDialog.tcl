
namespace eval nodeDialog {

	variable w ".nodeSelect"
	variable plot_title
	variable selection
	variable selected "const"
	variable selectionCnt

	proc create { } {
		variable w

		if {[winfo exists $w]} {
			raise $w
			Update
			return
		}

		toplevel $w
		
		wm title $w "Vectors"

		# Plot names
		#pack [frame $w.plots] -fill x
		frame $w.plots
		pack [listbox $w.plots.lst -height 4 \
			-width 50 \
			-yscrollcommand {.nodeSelect.plots.sx set} \
			] -fill x -expand 1 -side left

		pack [scrollbar $w.plots.sx -orient vertical \
			-width 9 \
			-command {.nodeSelect.plots.lst yview}\
			] -fill y -side right

		# vector names
		#pack [frame $w.nodes] -fill both -expand 1
		frame $w.nodes
		pack [blt::hierbox $w.nodes.hb -hideroot 1 \
			-yscrollcommand {.nodeSelect.nodes.sx set} \
			-selectbackground "#e6e6e6" \
			] -fill both -expand 1 -side left

		pack [scrollbar $w.nodes.sx -orient vertical \
			-width 9 \
			-command {.nodeSelect.nodes.hb yview} \
			] -fill y -side right

		# control frame
		#pack [frame $w.fcont] -fill x
		frame $w.fcont
		pack [label $w.fcont.lselected -text "Selected: "] -side left
                pack [button $w.fcont.bNodeCnt -textvariable spicewish::nodeDialog::selectionCnt \
                                -padx 1m \
                                -command spicewish::nodeDialog::clearSelection \
                                ] -side left
                pack [button $w.fcont.bPlot -text " Plot " \
                                -command spicewish::nodeDialog::plotSelection ]


		# blt table
                blt::table $w \
                        0,0 $w.plots -fill x \
                        1,0 $w.nodes -fill both \
                        2,0 $w.fcont -fill x

		blt::table configure $w r0 r2 -resize none

		# bindings
		$w.nodes.hb bind all <Button-1> {
			set hierbox_path %W
			set index [$hierbox_path index current]
			set nodeName [$hierbox_path entry cget $index -label]
			set plotTypeName [spice::plot_typename [.nodeSelect.plots.lst curselection]]
			set listPos [lsearch $::spicewish::nodeDialog::selection($plotTypeName) $nodeName]

			if { [ $hierbox_path entry cget $index -labelcolor] == "black"} {
				$hierbox_path entry configure $index -labelcolor "blue"
				$hierbox_path configure -selectforeground blue
	
				if {$listPos == -1 } {
					lappend spicewish::nodeDialog::selection($plotTypeName) $nodeName
				}
	 		} else {
				$hierbox_path entry configure $index -labelcolor "black"
				$hierbox_path configure -selectforeground black

                               	set spicewish::nodeDialog::selection($plotTypeName) [lreplace $::spicewish::nodeDialog::selection($plotTypeName) $listPos $listPos]
                               
	
		   	}   
			set spicewish::nodeDialog::selectionCnt [llength $::spicewish::nodeDialog::selection($plotTypeName)]
		}

		bind $w.plots.lst <ButtonRelease-1> {
			spicewish::nodeDialog::selectPlot %W  
		}

		# mouse wheel bindings
		bind $w.nodes.hb <Button-5> [list %W yview scroll 5 units]
		bind $w.nodes.hb <Button-4> [list %W yview scroll -5 units]

		wm protocol $w WM_TAKE_FOCUS { spicewish::nodeDialog::Update }
	
		Update
	}

	proc plotSelection { } {
		variable selection

		set plotNum [.nodeSelect.plots.lst curselection]
		set plotTypeName [spice::plot_typename $plotNum]

		if {$selection($plotTypeName) == ""} {return}

		eval "spicewish::viewer::oplot $plotTypeName $selection($plotTypeName)"
	}

	proc loadSelection { } {
		variable selection
		variable selectionCnt

		set plotTypeName [spice::plot_typename [.nodeSelect.plots.lst curselection]]

		set w_hb ".nodeSelect.nodes.hb"

		for {set index 1} { $index <= [$w_hb index end] } {incr index} {
			if {[$w_hb index $index] == "" } {continue}

			set name [$w_hb entry cget $index -label]
			set listPos [lsearch $selection($plotTypeName) $name]

			if {$listPos != -1} {
				$w_hb entry configure $index -labelcolor "blue"
			}
		}
		set selectionCnt [llength $selection($plotTypeName)]	
	}

	proc clearSelection { } {
		variable selection

		set plotTypeName [spice::plot_typename [.nodeSelect.plots.lst curselection]]
		set selection($plotTypeName) ""

		populateTree [.nodeSelect.plots.lst curselection]
	}

	proc Update { } {
		variable plot_title
		variable selection
		variable selected
		variable w

		if {![winfo exists $w]} {return}

		# get current plots titles
		set index 0
		while { ![catch {set t [spice::plot_name $index]} ] } {

	       	       	set p_title [spice::plot_title $index]
			set p_typeName [spice::plot_typename $index]

			set plot_title($index) "$p_typeName :$p_title"

			if {![info exists selection($p_typeName)]} {
				set selection($p_typeName) ""
			}
			
	                incr index
		}

		# populate current plot window
		.nodeSelect.plots.lst delete 0 end
		
                for {set i 0} {$i < [array size plot_title]} {incr i} {
                        .nodeSelect.plots.lst insert end $plot_title($i)
                }

		# list position of selected plot
		set listPos [spicewish::vectors::get_plotNum $selected]
		if {$listPos < 0} {set listPos 0}
	
		.nodeSelect.plots.lst selection set $listPos
		.nodeSelect.plots.lst see $listPos

		populateTree $listPos
		
        }	
	
	proc populateTree { index } {
		
		set w ".nodeSelect.nodes.hb"
		$w delete 0
		
		set defaultScale [spice::plot_defaultscale $index]
		
		foreach node [spice::plot_variables $index] {
			set text ": [spicewish::vectors::type $index $node], real, [spice::plot_datapoints $index]"
			if {$node == $defaultScale} {
				set text "$text \[default scale\]"
			}

			$w insert -at 0 end $node -labelfont {times 9} -text $text -labelcolor "black"
		}
		loadSelection
	}
	
	proc selectPlot { w } {
		variable selected
		set index [$w curselection]
		set selected [spicewish::vectors::get_plotTypeName $index]
		
		populateTree $index
	}
}
