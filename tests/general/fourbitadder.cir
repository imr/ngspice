4 bit adder

* Models:
.MODEL dmod D
.MODEL qmod NPN(level=1 BF=75 RB=100 CJE=1PF CJC=3PF)

.options noacct

.SUBCKT NAND 1 2 3 4
* noeuds: entrees(2) sortie vcc
q1 9 5 1 qmod
d1clamp 0 1 dmod
q2 9 5 2 qmod
d2clamp 0 2 dmod
rb 4 5 4k
r1 4 6 1.6k
q3 6 9 8 qmod
r2 8 0 1k
rc 4 7 130 
q4 7 6 10 qmod
dvbedrop 10 3 dmod
q5 3 8 0 qmod
.ends NAND 

.SUBCKT ONEBIT 1 2 3 4 5 6 
* noeuds entrees(2) ,carryin, sortie, carryout, vcc
x1 1 2 7 6 NAND
x2 1 7 8 6 NAND
x3 2 7 9 6 NAND
x4 8 9 10 6 NAND
x5 3 10 11 6 NAND   
x6 3 11 12 6 NAND
x7 10 11 13 6 NAND
x8 12 13 4 6 NAND
x9 11 7 5 6 NAND
.ends ONEBIT

.SUBCKT TWOBIT 1 2 3 4 5 6 7 8 9
* noeuds 
x1 1 2 7 5 10 9 ONEBIT
x2 3 4 10 6 8 9 ONEBIT
.ends TWOBIT

.SUBCKT FOURBIT 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

x1 1 2 3 4 9 10 13 16 15 TWOBIT
x2 5 6 7 8 11 12 16 14 15 TWOBIT
.ends FOURBIT



* Inputs/Supplies:

vcc 99 0 DC 5V
VIN1A 1 0 DC 0 pulse(0 3 0 10ns 10ns   10ns   50ns)
VIN1B 2 0 DC 0 pulse(0 3 0 10ns 10ns   20ns  100ns)
VIN2A 3 0 DC 0 pulse(0 3 0 10ns 10ns   40ns  200ns)
VIN2B 4 0 DC 0 pulse(0 3 0 10ns 10ns   80ns  400ns)
VIN3A 5 0 DC 0 pulse(0 3 0 10ns 10ns  160ns  800ns)
VIN3B 6 0 DC 0 pulse(0 3 0 10ns 10ns  320ns 1600ns)
VIN4A 7 0 DC 0 pulse(0 3 0 10ns 10ns  640ns 3200ns)
VIN4B 8 0 DC 0 pulse(0 3 0 10ns 10ns 1280ns 6400ns)

* Circuit description:
x1 1 2 3 4 5 6 7 8 9 10 11 12 0 13 99 FOURBIT
rbit0 9 0 1k
rbit1 10 0 1k
rbit2 11 0 1k
rbit3 12 0 1k
rcout 13 0 1k

* Analysys:
.tran 1ns 6ns
.print tran v(1)


.end

 
