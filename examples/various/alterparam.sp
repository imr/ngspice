*test alterparam
.param vv = 1
.param rr = 'vv + 1'

R1 1 0 {rr + 1}
v1 1 0 1

.subckt subr in out rint1 = 10
  .param rint = 5
  .param rint2 = 99
  R0 in out 'rint'
  R1 in out 'rint1'
  R2 in out 'rint2'
.ends

Xr 2 0 subr rint = 7 rint1 = 15
v2 2 0 1

.control
op
print v1#branch v2#branch
echo
listing expand
remcirc
alterparam vv = 2
mc_source
op
print v1#branch v2#branch
echo
listing expand
alterparam subr rint = 11
mc_source
op
print v1#branch v2#branch
echo
listing expand
.endc

.end
