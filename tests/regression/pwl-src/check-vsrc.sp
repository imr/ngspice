check-vsrc

.control

begin

if $argc ne 4
  echo "Error: usage: vv seq td ..pass vectors vv seq and time, and td"
  goto bottom
end

set vv =  $argv[1]
set seq = $argv[2]
set td  = $argv[3]

echo "checking $vv"

set expected_hits = $argv[4]

let vv_length  = length( time )
let seq_length = length( $seq )

define match(x,gold,relerr) abs(x-gold) <= abs(relerr*gold)


*--------------------
* compare golden seq with actual voltage

let m = 0
let k = 0

while k < vv_length

  let tmp_t = time [k]  
  let tmp_v = $vv [k]

  while tmp_t >= $seq [m+2] + $td
     let m = m + 2
  end

  let t0 = $seq [m+0] + $td
  let v0 = $seq [m+1]
  let t1 = $seq [m+2] + $td
  let v1 = $seq [m+3]

  if tmp_t <= t0
    let vgold = v0
  else
    let vgold = v0 + ((v1 - v0)/(t1 - t0)) * (tmp_t - t0)
  end

  if not match(tmp_v, vgold, 1.0e-9)
    echo "ERROR: v[ $&k ] @ $&tmp_t is $&tmp_v expected $&vgold"
  end

  let k = k + 1
end


*--------------------
* now search for the breakpoint hits

let m = 0
let k = 0
let hit = 0

while k < vv_length  and  m < seq_length

  let tmp_t = time [k]
  let tmp_s = $seq [m] + $td;

  if match(tmp_t, tmp_s, 1e-9)
    let hit = hit + 1
    echo "INFO: hit at $&tmp_s"
  end

  if tmp_t <= tmp_s
    let k = k + 1
  else
    let m = m + 2
  end

end

if hit ne $expected_hits
  echo "error: breakpoint hits mismatch, actual $&hit, expected $expected_hits"
else
  echo "info: breakpoint seem to be ok"
end

echo ""

label bottom

* Local Variables:
* mode: spice
* End:

end
