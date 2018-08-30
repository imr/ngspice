*test alterparam
.param vv = 1
.param rr = 'vv + 1'

R1 1 0 {rr + 1}
v1 1 0 1

.subckt subr in out rint1 = 6
  .param rint = 5
  .param rint2 = 8
  R0 in out 'rint'
  R1 in out 'rint1'
  R2 in out 'rint2'
.ends

Xr 2 0 subr rint = 7 rint1 = 9
v2 2 0 1

.control
op
print v1#branch v2#branch
echo
listing expand
alterparam vv = 2
reset
op
print v1#branch v2#branch
echo
listing expand
alterparam subr rint = 13
alterparam subr rint1 = 15
alterparam subr rint2 = 17
reset
op
print v1#branch v2#branch
echo
listing expand
.endc

.end
