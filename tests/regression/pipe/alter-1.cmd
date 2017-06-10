set prompt = ""
let fail_count = 0
source alter-1.cir
op

if @v1[sin][2] <> 100meg
   echo "ERROR: tst100 failed"
   let fail_count = fail_count + 1
end

# space after '[' is mandatory
# alter @v1[sin] = [0 1 113meg]
#
# if @v1[sin][2] <> 113meg
#    echo "ERROR: tst113 failed"
#    let fail_count = fail_count + 1
# end

alter @v1[sin] = [ 0 1 114meg ]
if @v1[sin][2] <> 114meg
   echo "ERROR: tst114 failed"
   let fail_count = fail_count + 1
end

alter @v1[sin]=[ 0 1 115meg ]

if @v1[sin][2] <> 115meg
   echo "ERROR: tst115 failed"
   let fail_count = fail_count + 1
end

alter @v1[sin] =[ 0 1 116meg ]

if @v1[sin][2] <> 116meg
   echo "ERROR: tst116 failed"
   let fail_count = fail_count + 1
end

alter @v1[sin]= [ 0 1 117meg ]

if @v1[sin][2] <> 117meg
   echo "ERROR: tst117 failed"
   let fail_count = fail_count + 1
end

# this will cause a warning, because
#   the first ']' will be a separate word
#   instead of beeing fused with the "[sin"
alter @v1[sin ]= [ 0 1 118meg ]

if @v1[sin][2] <> 118meg
   echo "ERROR: tst118 failed"
   let fail_count = fail_count + 1
end

# check old syntax without '=' too
alter v1 sin [ 0 1 119meg ]

if @v1[sin][2] <> 119meg
   print @v1[sin]
   echo "ERROR: tst119 failed"
   let fail_count = fail_count + 1
end

# check old syntax with '=' too
alter v1 sin = [ 0 1 120meg ]

if @v1[sin][2] <> 120meg
   echo "ERROR: tst120 failed"
   let fail_count = fail_count + 1
end

alter v1 sin =[ 0 1 121meg ]

if @v1[sin][2] <> 121meg
   echo "ERROR: tst121 failed"
   let fail_count = fail_count + 1
end

alter v1 sin= [ 0 1 122meg ]

if @v1[sin][2] <> 122meg
   echo "ERROR: tst122 failed"
   let fail_count = fail_count + 1
end

alter v1 sin=[ 0 1 123meg ]

if @v1[sin][2] <> 123meg
   echo "ERROR: tst123 failed"
   let fail_count = fail_count + 1
end

if fail_count > 0
  echo "ERROR: $&fail_count tests failed"
  quit 1
else
  echo "INFO: all tests passed"
  quit 0
end
