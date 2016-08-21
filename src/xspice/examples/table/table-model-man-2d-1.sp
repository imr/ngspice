Code Model Test - 2d Table Model
* simple input table, generated manually
*
*** analysis type ***
.control
dc V1 0.1 6.9 0.2 V2 0.5 7.5 0.5
plot v(1) v(2) v(11, 10)
.endc
*
*** input sources ***
*
v1 1 0 DC 1.5
*
v2 2 0 DC 1.5
*
*** table model ***
a1 1 2 %id(10 11) table1
.model table1 table2d (offset=0.0 gain=1.0 order=3 file="test-2d-1.table")
*
*

*** resistors to ground ***
r1 1 0 1k
r2 2 0 1k

r3 10 11 1k

r10 10 0 1k
*
*
.end
