Test sequences for ngspice control structures
*vectors are used (except foreach)
*start in interactive mode

.control

* test for label, goto   
  echo
  let index = 0    
  label starthere2
  let loop = 0   
  echo We are at start with index "$&index" and loop "$&loop"
  if index < 6
    label inhere
    let index = index + 1
    if loop < 3
      let loop = loop + 1
      if index > 1
        echo jump2
        goto starthere2
      end  
    end
    echo jump
    goto inhere
  end 
  echo We are at end with index "$&index" and loop "$&loop"   
  
* test goto in while loop
  echo
  let loop = 0
  if 1    ; outer loop to allow nested forward label 'endlabel'
    while loop < 10
      if loop > 5
        echo jump
        goto endlabel
      end
      let loop = loop + 1
    end
    echo before  ; never reached
    label endlabel
    echo after "$&loop"  
  end  

  
  quit
.endc
