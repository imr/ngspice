
namespace eval eng {

	proc float_eng { num {forceunit ""} } {

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
					set decimal_length_to_show [expr (($significant_digits - 1) - [string length $whole])]

					if {$decimal_length_to_show >= 0} {
		    
						if { $fraction == "" } {
							set newnm $whole
		    				} else {
							#need some decimal places to give the resolution required
							set newnum "$whole.[string range $fraction 0 [expr ($decimal_length_to_show)]]"
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
			return $num
    		}
	} 
}
