

namespace eval nodeDialog {

	variable w ".nodeSelect"
	variable plot_title
	variable selection
	variable selectionCnt

	proc create { } {
		variable w

		if {[winfo exists $w]} {
			raise $w
			return
		}

		toplevel $w
		
		pack [frame $w.plots] -fill x
		pack [listbox $w.plots.lst -height 3 \
			-width 50 \
			-yscrollcommand {.nodeSelect.plots.sx set} \
			] -fill x -expand 1 -side left

		pack [scrollbar $w.plots.sx -orient vertical \
			-width 9 \
			-command {.nodeSelect.plots.lst yview}\
			] -fill y -side right


		pack [frame $w.nodes] -fill both -expand 1
		pack [blt::hierbox $w.nodes.hb -hideroot 1 \
			-yscrollcommand {.nodeSelect.nodes.sx set} \
			-selectbackground "#e6e6e6" \
			] -fill both -expand 1 -side left

		pack [scrollbar $w.nodes.sx -orient vertical \
			-width 9 \
			-command {.nodeSelect.nodes.hb yview} \
			] -fill y -side right

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
		pack [frame $w.fcont] -fill x
		pack [button $w.fcont.bNodeCnt -textvariable spicewish::nodeDialog::selectionCnt \
				-padx 1m \
				-command spicewish::nodeDialog::clearSelection \
				] -side left
		pack [button $w.fcont.bPlot -text "Plot" \
				-command spicewish::nodeDialog::plotSelection ] 

		bind $w.plots.lst <ButtonRelease-1> { spicewish::nodeDialog::selectPlot %W  }

		wm protocol $w WM_TAKE_FOCUS { spicewish::nodeDialog::currentPlots }
	
		currentPlots
	}

	proc plotSelection { } {
		variable selection

		set plotNum [.nodeSelect.plots.lst curselection]
		set plotTypeName [spice::plot_typename $plotNum]
		eval "spicewish::viewer::oplot $plotNum $selection($plotTypeName)"
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

	proc currentPlots { } {
		variable plot_title
		variable selection

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

		.nodeSelect.plots.lst delete 0 end

                for {set i 0} {$i < [array size plot_title]} {incr i} {
                        .nodeSelect.plots.lst insert end $plot_title($i)
                }

		.nodeSelect.plots.lst selection set 0
		populateTree 0
		
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
		set index [$w curselection]
		populateTree $index
	}
}
