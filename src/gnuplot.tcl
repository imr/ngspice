
namespace eval gnuplot {

	proc writeDataFile { plotNum args } {

		set fileName "gnuplot.dat"
		set fileId [open $fileName "w"]
		
		set plot_list "[spicewish::vectors::defaultScale $plotNum] $args"

		puts $fileId "// $plot_list"

		for {set dPoint 0} {$dPoint < [spice::plot_datapoints $plotNum]} {incr dPoint} {

			for {set i 0} {$i < [llength $plot_list]} {incr i} {
				set vecName [lindex $plot_list $i]
				puts -nonewline $fileId "[spice::plot_get_value $vecName $plotNum $dPoint]\t"
			}
			puts $fileId ""
		}
	
			
		close $fileId
	}

	proc writeCommandFile { plotNum args } {
		set fileName "gnuplot.cmd"
		set fileId [open $fileName "w"]

		puts $fileId "set title \"[spice::plot_title $plotNum]\""
	
		puts $fileId "set xlabel \"[spicewish::vectors::defaultScale $plotNum]\""
		puts $fileId "set ylabel \" \""
		puts $fileId "plot \\"

		for {set pCnt 0} {$pCnt < [llength $args]} {incr pCnt} {
			set plotName [lindex $args $pCnt]

			if {$pCnt ==[expr [llength $args] -1]} {
				puts $fileId "\"gnuplot.dat\" using 1:[expr $pCnt +2] with lines title \"$plotName\""
			} else  {
				puts $fileId "\"gnuplot.dat\" using 1:[expr $pCnt +2] with lines title \"$plotName\", \\"
			}
		} 

		puts $fileId "pause -1 \"Hit return to exit\""

		close $fileId
	}

	proc plot { args } {
		set plotNum 0		

		eval "set validVecs \[spicewish::vectors::validVectorList $plotNum $args \]"

		if {$validVecs == ""} {
			puts "Error: no valid vectors"
			return
		}
		eval writeCommandFile $plotNum $validVecs
		eval writeDataFile $plotNum $validVecs
	}	
}
