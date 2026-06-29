Test sequences for ngspice control structures
*vectors are used (except foreach)
*start in interactive mode

.control

* simple test for label, goto
  echo
  let loop = 0
  label starthere 
  echo start "$&loop"
  let loop = loop + 1
  if loop < 3
    goto starthere
  end  
  echo end "$&loop"   
  
  quit
.endc
