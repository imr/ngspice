6.3inch 4 lossy lines CPL model -- R load 

Ra    1    2                       1K
Rb    0    3                       1K
Rc    0    4                       1K
Rd    0    5                       1K
Re    6    0                       1Meg
Rf    7    0                       1Meg
Rg    8    0                       1Meg
Rh    9    0                       1Meg
*
P1    2 3 4 5  0  6 7 8 9 0  LOSSYMODE
*
*
VS1   1    0   PWL(15.9NS 0.0 16.1Ns 5.0 31.9Ns 5.0 32.1Ns 0.0) 
*
.OPTION NOACCT
.TRAN   0.2NS  50NS 0  0.05N 
.PRINT TRAN V(2) V(7) V(9)
.MODEL LOSSYMODE   CPL
+R=0.3      0        0        0 
+          0.3       0        0 
+                   0.3       0  
+                            0.3
+L=9e-9    5.4e-9    0        0
+           9e-9    5.4e-9    0
+                   9e-9     5.4e-9
+                            9e-9
+G=0 0 0 0 0 0 0 0 0 0
+C=3.5e-13 -3e-14    0        0
+          3.5e-13 -3e-14     0    
+                   3.5e-13  -3e-14
+                            3.5e-13
+length=6.3

.END
