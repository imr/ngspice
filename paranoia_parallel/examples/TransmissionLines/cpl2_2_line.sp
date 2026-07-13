MOSdriver -- 24inch 2 lossy lines CPL model -- C load 

m1     0     268    299     0  mn0p9  w = 18.0u l=1.0u
m2    299    267    748     0  mn0p9  w = 18.0u l=1.0u
m3     0     168    648     0  mn0p9  w = 18.0u l=0.9u
m4     1     268    748     1  mp1p0  w = 36.0u l=1.0u
m5     1     267    748     1  mp1p0  w = 36.0u l=1.0u
m6     1     168    648     1  mp1p0  w = 36.0u l=1.0u
*
CN648  648   0  0.025398e-12 
CN651  651   0  0.007398e-12 
CN748  748   0  0.025398e-12 
CN751  751   0  0.009398e-12 
CN299  299   0  0.005398e-12 
*
P1  648 748 0  651 751 0  PLINE 
*
vdd    1    0   DC  5.0
VK   267    0   DC  5.0
*
*VS 168  0  PWL 4 15.9N 0.0 16.1n 5.0 31.9n 5.0 32.1n 0.0
*VS 268  0  PWL 4 15.9N 0.0 16.1n 5.0 31.9n 5.0 32.1n 0.0
*
VS1 168  0  PULSE (0 5 15.9N 0.2N 0.2N 15.8N 60N)
VS2 268  0  PULSE (0 5 15.9N 0.2N 0.2N 15.8N 60N)
*
.control
TRAN 0.2N 47.9NS 0 1N
plot v(648) v(651) v(751)
.endc
*
.MODEL PLINE CPL
+R=0.2       0 
+           0.2
+L=9.13e-9  3.3e-9
+           9.13e-9
+G=0 0 0
+C=3.65e-13 -9e-14
+           3.65e-13
+length=24
*******************     MODEL SPECIFICATION    **********************
.MODEL mn0p9 NMOS VTO=0.8 KP=48U GAMMA=0.30 PHI=0.55 LAMBDA=0.00 CGSO=0 CGDO=0
+            CJ=0 CJSW=0 TOX=18000N LD=0.0U
.MODEL mp1p0 PMOS VTO=-0.8 KP=21U GAMMA=0.45 PHI=0.61 LAMBDA=0.00 CGSO=0 CGDO=0 
+           CJ=0 CJSW=0 TOX=18000N LD=0.0U
.END
