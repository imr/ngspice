*ng_script

*
.control

  source mc_ring_circ.net  		$ source the input file


  save buf                        	$ we just need buf, save memory by more than 10x

    tran 5p 50n 0

plot v(buf)
end
*quit
** ngspice on this file to check for basic functionality **


