Test sequences for ngspice control structures
*vectors are used (except foreach)
*start in interactive mode

.control

* test for while, repeat, if, break
  let loop = 0
  while loop < 4
    let index = 0
    repeat
      let index = index + 1
      if index > 4
        break
      end  
    end
    echo index "$&index"   loop "$&loop"
    let loop = loop + 1
  end


* test sequence for while, dowhile  
  let loop = 0
  echo
  echo enter loop with  "$&loop" 
  dowhile loop < 3
    echo within dowhile loop "$&loop"     
    let loop = loop + 1 
  end
  echo after dowhile loop "$&loop" 
  echo  
  let loop = 0
  while loop < 3
    echo within while loop "$&loop"   
    let loop = loop + 1
  end
  echo after while loop "$&loop"  
  let loop = 3
  echo
  echo enter loop with  "$&loop" 
  dowhile loop < 3
    echo within dowhile loop "$&loop"     $ output expected
    let loop = loop + 1 
  end
  echo after dowhile loop "$&loop"   
  echo  
  let loop = 3
  while loop < 3
    echo within while loop "$&loop"       $ no output expected   
    let loop = loop + 1
  end
  echo after while loop "$&loop"   
 
  
* test sequence for foreach
  echo
  foreach outvar 0 0.5 1 1.5
    echo parameters: $outvar   $ foreach parameters are variables, not vectors!
  end  
  
* test for if ... else ... end  
  echo
  let loop = 0 
  let index = 1 
  dowhile loop < 10
    let index = index * 2
    if index < 128
      echo "$&index" lt 128
    else
      echo "$&index"  ge 128  
    end
    let loop = loop + 1
  end   
  
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
  
* test for label, nested goto
  echo
  let loop = 0
  label starthere1 
  echo start nested "$&loop"
  let loop = loop + 1
  if loop < 3
    if loop < 3
      goto starthere1
    end  
  end  
  echo end "$&loop"   
  
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
  if 1    $ outer loop to allow nested forward label 'endlabel'
    while loop < 10
      if loop > 5
        echo jump
        goto endlabel
      end
      let loop = loop + 1
    end
    echo before  $ never reached
    label endlabel
    echo after "$&loop"  
  end  

*test for using variables
* simple test for label, goto
  echo
  set loop = 0
  label starthe 
  echo start $loop
  let loop = $loop + 1  $ expression needs vector at lhs
  set loop = "$&loop"   $ convert vector contents to variable
  if $loop < 3
    goto starthe
  end  
  echo end $loop
.endc
