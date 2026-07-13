6-line coupled multiconductor with ECL drivers
vemm mm 0 DC -0.4
vepp pp 0 DC 0.4
vein_left  lin  0 PULSE (-0.4 0.4 0N 1N 1N 7N 200N)
vein_right rin  0 PULSE (-0.4 0.4 2N 1N 1N 7N 200N)

* upper 2 lines
x1 lin 0 1 1outn  ECL
x2 mm 0 2 2outn   ECL
x7 7 0 7r 7routn  ECL
x8 8 0 8r 8routn  ECL

c7r 7r 0 0.1P
c8r 8r 0 0.1P

* lower 2 lines
x11 pp 0 11 11outn  ECL
x12 rin 0 12 12outn  ECL
x5  5 0 5l 5loutn  ECL
x6  6 0 6l 6loutn ECL

c5l 5l 0 0.1P
c6l 6l 0 0.1P

p1 1 2 3 4 5 6  0  7 8 9 10 11 12  0  pline

.model pline cpl
+C = 0.900000P  -0.657947P -0.0767356P -0.0536544P -0.0386514P -0.0523990P
+                1.388730P -0.607034P  -0.0597635P -0.0258851P -0.0273442P
+                             1.39328P  -0.625675P -0.0425551P -0.0319791P
+                                         1.07821P  -0.255048P -0.0715824P
+                                                     1.06882P  -0.692091P
+                                                                0.900000P
+L = 0.868493E-7 0.781712E-7 0.748428E-7 0.728358E-7 0.700915E-7 0.692178E-7
+                0.866074E-7 0.780613E-7 0.748122E-7 0.711591E-7 0.701023E-7
+                            0.865789E-7 0.781095E-7 0.725431E-7 0.711986E-7
+                                        0.867480E-7 0.744242E-7 0.725826E-7
+                                                    0.868022E-7 0.782377E-7
+                                                                0.868437E-7
+R = 0.2  0    0    0    0    0
+         0.2  0    0    0    0
+              0.2  0    0    0
+                   0.2  0    0
+                        0.2  0
+                             0.2
+G = 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+
+length = 2

*XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
.SUBCKT ECL EIN GND 9 8
*    Input-GND-OUTP-OUTN
RIN  1 2   0.077K
REF  5 6   0.077K
R1   7 N   1.0K
R2   P 3   0.4555K
R3   P 4   0.4555K
R4   8 N   0.615K
R5   9 N   0.615K
RL1  8 GND 0.093K
RL2  9 GND 0.093K
LIN  EIN 1 0.01U
LREF 5 GND 0.01U
CIN  1 GND 0.68P
CL1  8 GND 1P
CL2  9 GND 1P
Q1 3 2 7 JCTRAN
Q2 4 6 7 JCTRAN
Q3 P 3 8 JCTRAN
Q4 P 4 9 JCTRAN
VEP  P GND DC 1.25
VEN  N GND DC -3
.ENDS ECL

.control
TRAN 0.1N 20N
plot V(3) V(5) V(8) V(11) V(12)
.endc
.MODEL JCTRAN NPN BF=150 VAF=20 IS=4E-17 RB=300 RC=100 CJE=30F CJC=30F
+               CJS=40F VJE=0.6 VJC=0.6 VJS=0.6 MJE=0.5 MJC=0.5
+               MJS=0.5 TF=16P TR=1N
.END
