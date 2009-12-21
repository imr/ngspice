*test sequence for repeat
.control
  let loop = 0
  while loop < 4
    let index = 0
    repeat
      let index = index + 1
      if index > 4
        break
      end  
    end
    echo
    echo index "$&index"   loop "$&loop"
    let loop = loop + 1
  end
.endc