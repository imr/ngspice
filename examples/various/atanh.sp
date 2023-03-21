atanh test for real and complex inputs
v1 1 0 dc 0 ac 1
r1 1 2 50
c1 2 0 40f
l1 2 3 50p
c2 3 0 40f
l2 3 4 50p
r2 4 0 50
.control
* real numbers
let invec = vector(1999)
compose invec start=-0.999 stop=0.999 step=0.001
let outvec = atanh(invec)
setscale invec
set xbrushwidth=3
plot outvec
ac dec 100 1g 100g
let X=v(4); complex vector to be worked
*complex numbers
let Y1=X+X^3/3+X^5/5+X^7/7+X^9/9+X^11/11; pseudo atanh
Let Y2 = atanh(X)
plot polar Y1
plot polar Y2
plot polar Y1 Y2

.endc
.end
