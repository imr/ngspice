* Shared netlist for several example circuits using XSPICE-based 74HC00 model

.SUBCKT ONEBIT 1 2 3 4 5 6
*   NODES:  INPUT(2), CARRY-IN, OUTPUT, CARRY-OUT, VCC
X1   1  2  7  6  0 74HC00
X2   1  7  8  6  0 74HC00
X3   2  7  9  6  0 74HC00
X4   8  9 10  6  0 74HC00
X5   3 10 11  6  0 74HC00
X6   3 11 12  6  0 74HC00
X7  10 11 13  6  0 74HC00
X8  12 13  4  6  0 74HC00
X9  11  7  5  6  0 74HC00
.ENDS ONEBIT

.SUBCKT TWOBIT 1 2 3 4 5 6 7 8 9
*   NODES:  INPUT - BIT0(2) / BIT1(2), OUTPUT - BIT0 / BIT1,
*           CARRY-IN, CARRY-OUT, VCC
X1   1  2  7  5 10  9   ONEBIT
X2   3  4 10  6  8  9   ONEBIT
.ENDS TWOBIT

.SUBCKT FOURBIT 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
*   NODES:  INPUT - BIT0(2) / BIT1(2) / BIT2(2) / BIT3(2),
*           OUTPUT - BIT0 / BIT1 / BIT2 / BIT3, CARRY-IN, CARRY-OUT, VCC
X1   1  2  3  4  9 10 13 16 15   TWOBIT
X2   5  6  7  8 11 12 16 14 15   TWOBIT
.ENDS FOURBIT

*** POWER
VCC   99  0   DC 3.3V

*** ALL INPUTS
VIN1A  1  0   DC 0 PULSE(0 3 0 5NS 5NS   20NS   50NS)
VIN1B  2  0   DC 0 PULSE(0 3 0 5NS 5NS   30NS  100NS)
VIN2A  3  0   DC 0 PULSE(0 3 0 5NS 5NS   50NS  200NS)
VIN2B  4  0   DC 0 PULSE(0 3 0 5NS 5NS   90NS  400NS)
VIN3A  5  0   DC 0 PULSE(0 3 0 5NS 5NS  170NS  800NS)
VIN3B  6  0   DC 0 PULSE(0 3 0 5NS 5NS  330NS 1600NS)
VIN4A  7  0   DC 0 PULSE(0 3 0 5NS 5NS  650NS 3200NS)
VIN4B  8  0   DC 0 PULSE(0 3 0 5NS 5NS 1290NS 6400NS)

*** DEFINE NOMINAL CIRCUIT
X1     1  2  3  4  5  6  7  8  9 10 11 12  0 13 99 FOURBIT

.option noinit acct
.TRAN 500p 6400NS
* save inputs
.save V(1) V(2) V(3) V(4) V(5) V(6) V(7) V(8)

.control
pre_set strict_errorhandling
unset ngdebug
*save outputs and specials
save x1.x1.x1.7 V(9) V(10) V(11) V(12) V(13)
run
rusage
* plot the inputs, use offset to plot on top of each other
plot  v(1) v(2)+4 v(3)+8 v(4)+12 v(5)+16 v(6)+20 v(7)+24 v(8)+28
* plot the outputs, use offset to plot on top of each other
plot  v(9) v(10)+4 v(11)+8 v(12)+12 v(13)+16
.endc
