# Script to run Spicepp from Tcl
# 07/06/04 - ad
#
# version 0.1.2
#
# note:
#   edit variable spicepp_dir to for dir contaning spicepp code
#   spicepp avaliable at http://sourceforge.net/projects/tclspice
#

namespace eval spicepp {

	variable version 0.1.3
	variable spicepp_dir $::spice_library/spice
	variable outFile_extension ".pp-cir"	
	variable timeStamp_prefix "* spicepp:"
	variable Hspice_structures ".meas .param .lib .globals"

	#------------------------------------------------------
	#
	# returns
	#  - empty string if an error
	#  - name of passed file if in spice3f5 syntax
	#  - or name of translated file
	#
	proc convert { fileName } {
		
		variable outFile_extension
		variable spicepp_dir

		if {![file exists $fileName]} {
			puts "spicepp: '${fileName}' dosen't exist"
			return ""
		}

		# output fileName
		set strLen [expr [string length $fileName] - [string length [file extension $fileName]] -1]
		set outFile [string range $fileName 0 $strLen]
		set outFile "${outFile}${outFile_extension}"

		# check that input file contains Hspice statements
		if {![checkHSpiceFormat $fileName]} {
			puts "spicepp: '$fileName' doesn't contain any Hspice statements."
			return $fileName
		}

		# check modified time stamp in previous generated output files
		#  against modified time of input file
		if {![checkTimeStamp $fileName $outFile]} {
			puts "spicepp: '$fileName' hasn't been modified since last translation"
			return $outFile
		}

		# conversion Start Time
		set sTime [clock seconds]
		
		# execute Spicepp on given file
		puts "spicepp: converting '$fileName' > '$outFile'"

		if {[catch {exec ${spicepp_dir}/spicepp.pl $fileName > $outFile} errResult]} {
			puts "spicepp couldn't convert '$fileName'"
			puts $errResult
			return ""
		}
		
		# add time stamp to output file
		addTimeStamp $fileName $outFile

		# Conversion Time
		#puts "Spicepp : [expr [clock seconds] - $sTime]s"
		
		return $outFile
	}

	#-----------------------------------------
	# adds the modified time of orginal spice deck to second line
	#  of output file
	#
	proc addTimeStamp { inpFile outFile } {
	
		variable timeStamp_prefix

		set mTime [file mtime $inpFile]
		
		set fileId [open $outFile "r"]
		set fileData [read $fileId]
		set fileData [split $fileData "\n"]
		close $fileId

		set fileId [open $outFile "w"]
		set lineCnt 0

		foreach lineData $fileData {
			
			if {$lineCnt ==1} {
				puts $fileId "$timeStamp_prefix $mTime"
			}
			puts $fileId $lineData	
			incr lineCnt
		}
		close $fileId
	}

	#-------------------------------------------
	# Checks modified time stamp in output file against
	#  modified time of input file 
	proc checkTimeStamp { inpFile outFile } {
		
		variable timeStamp_prefix
	
		# check output file exists
		if {![file exists $outFile]} { 
			return 1
		}
			
		set fileId [open $outFile "r"]
		set fileData [read $fileId]
		set fileData [split $fileData "\n"]
		close $fileId
	
		set timeStampData [lindex $fileData 1]
		set timeStampData [string range $timeStampData [string length $timeStamp_prefix] end]
		# remove spaces
		eval set timeStampData \"$timeStampData\"
		
		if {![string is integer $timeStampData]} { 			
			return 1 
		}
		
		# compare modified times
		if { $timeStampData == [file mtime $inpFile] } {
			return 0
		}
		return 1
	}

	#---------------------------------------------
	# parses the spice file for any Hspice stuctures
	#  .meas .lib etc 
	proc checkHSpiceFormat { fileName } {

		variable Hspice_structures

		set fileId [open $fileName "r"]
		set fileData [read $fileId]
		set fileData [split $fileData "\n"]
		close $fileId

		for {set lineCnt 0} {$lineCnt < [llength $fileData]} {incr lineCnt} {
			set line [lindex $fileData $lineCnt]
			
			# get line structre
			set line [string range $line 0 [expr [string first " " $line] -1]]
		
			# lowercase
			set line [string tolower $line]				

			# skip line if '*' as lsearch returns a 0
			if {$line == "*"} { continue } 		
	
			# check line structure against Hspice list
			if {[lsearch $Hspice_structures $line] != -1} {
		#		puts "match found line $lineCnt : '$line'"
				return 1
			}
		}
		return 0
	}

}
