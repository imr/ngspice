Code Model Test - 3d Table Model
* simple table, generated manually
*
*** analysis type ***
.control
dc V1 0.1 1.4 0.1 V3 4.1 4.5 0.2
plot v(1) v(2) v(11, 10)
.endc
*
*** input sources ***
*
v1 1 0 DC 1.5
*
v2 2 0 DC 1.5
*
v3 3 0 DC 1.5

**** table model ***
a1 1 2 3 %id(10 11) table1
.model table1 table3d (offset=0.0 gain=1.0 order=2 file="test-3d-1.table")
*
*

*** resistors to ground ***
r1 1 0 1k
r2 2 0 1k

r3 10 11 10

r10 10 0 1k
*
*
.end
