Mixed single and coupled transmission lines
c1g 1 0 1P
l11a 1 1a 6e-9
r1a7 1a 7 0.025K
rin6 in 6 0.075K
l67 6 7 10e-9
c7g 7 0 1P
P2  1 7  0 2 8 0 PLINE
.MODEL PLINE CPL
+R = 2.25     0
+             2.25
+L = 0.6e-6   0.05e-6
+             0.6e-6
+G = 0 0 0
+C =  1.2e-9  -0.11e-9
+               1.2e-9
+length = 0.03
c2g 2 0 0.5P
r2g 2 0 0.05K
r23 2 3 0.025K
l34 3 4 5e-9
c4g 4 0 2P
l89 8 9 10e-9
c9g 9 0 1P
Y1  9 0 10 0 txline
.model  txline txl R = 1 L =0.6e-6 G = 0 C= 1.0e-9 length=0.04
l1011 10 11 10e-9
c11g 11 0 0.5P
r11g 11 0 0.05K
r1112 11 12 0.025K
l1213 12 13 5e-9
c13g 13 0 2P
r1116 11 16 0.025K
l1617 16 17 5e-9
c17g 17 0 2P
P1    4 2 13 17   0   5 14 15 18   0   PLINE1

.MODEL PLINE1 CPL
+R = 3.5  0    0    0
+         3.5  0    0
+              3.5  0
+                   3.5
+L =
+1e-6    0.11e-6 0.03e-6  0
+        1e-6    0.11e-6  0.03e-6
+                1e-6     0.11e-6
+                            1e-6
+G = 0 0 0 0 0 0 0 0 0 0
+C =
+1.5e-9 -0.17e-9 -0.03e-9   0
+         1.5e-9 -0.17e-9 -0.03e-9 
+                  1.5e-9 -0.17e-9
+                           1.5e-9
+length = 0.02

D1   5  0 dmod
D2  14  0 dmod
D3  15  0 dmod
D4  18  0 dmod

.model dmod d

VES in 0 PULSE (0 5 0 1.1ns 0.1ns 0.9ns 200ns)

.control
TRAN 0.2N 10.0N 
plot v(3) v(6) v(7) v(8) v(11) v(15)
.endc

.END
