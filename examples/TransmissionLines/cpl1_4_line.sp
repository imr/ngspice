MOSdriver -- 6.3inch 4 lossy line CPL model -- C load 

m1     1     2      6       1  mp1p0  w = 36.0u l=1.0u
m2     1     3      7       1  mp1p0  w = 36.0u l=1.0u
m3     1     4      8       1  mp1p0  w = 36.0u l=1.0u
m4     1     10     5       1  mp1p0  w = 36.0u l=1.0u
m5     1     11     13      1  mp1p0  w = 36.0u l=1.0u
m6     1     12     13      1  mp1p0  w = 36.0u l=1.0u

m7     0     2      6       0  mn0p9  w = 18.0u l=0.9u
m8     0     3      7       0  mn0p9  w = 18.0u l=0.9u
m9     0     4      8       0  mn0p9  w = 18.0u l=0.9u
m10    0     10     5       0  mn0p9  w = 18.0u l=0.9u
m11    14    11     13      0  mn0p9  w = 18.0u l=0.9u
m12    0     12     14      0  mn0p9  w = 18.0u l=0.9u

*
CN5  5     0  0.025398e-12 
CN6  6     0  0.007398e-12 
CN7  7     0  0.007398e-12 
CN8  8     0  0.007398e-12 
CN9  9     0  0.097398e-12 
CN10 10    0  0.007398e-12 
CN11 11    0  0.003398e-12 
CN12 12    0  0.004398e-12 
CN13 13    0  0.008398e-12 
CN14 14    0  0.005398e-12 

*
P1    5 6 7 8 0      9 10 11 12 0 pline

*
*
vdd    1    0   DC  5.0
v3     3    0   DC  5.0
*
VS1 2  0 PULSE ( 0 5 15.9NS 0.2NS 0.2NS 15.8NS 32NS)
VS2 4  0 PULSE (0 5 15.9NS 0.2NS 0.2NS 15.8NS 32NS )
*
.control
TRAN 0.2N 47.9N 0 0.05N
plot V(5) V(6) V(7) V(8) V(9) V(10) V(11) V(12)
.endc
.MODEL mn0p9 NMOS VTO=0.8 KP=48U GAMMA=0.30 PHI=0.55 LAMBDA=0.00 CGSO=0 CGDO=0
+CJ=0 CJSW=0 TOX=18000N LD=0.0U
.MODEL mp1p0 PMOS VTO=-0.8 KP=21U GAMMA=0.45 PHI=0.61 LAMBDA=0.00 CGSO=0 CGDO=0 
+CJ=0 CJSW=0 TOX=18000N LD=0.0U
.MODEL PLINE cpl
+R=0.03     0       0      0 
+          0.03     0      0 
+                  0.03    0 
+                         0.03
+L=9e-9    5.4e-9   0      0
+           9e-9   5.4e-9  0
+                   9e-9  5.4e-9
+                         9e-9
+G=0 0 0 0 0 0 0 0 0 0
+C=3.5e-13 -3e-14   0      0
+          3.5e-13 -3e-14  0
+                  3.5e-13 -3e-14
+                         3.5e-13
+length=6.3
.END
