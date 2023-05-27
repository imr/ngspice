MOSdriver -- 6.3inch 4 lossy line LTRA model -- C load 

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
* Subcircuit test
* test is a subcircuit that models a 4-conductor transmission line with
* the following parameters: l=9e-09, c=2.9e-13, r=0.3, g=0,
* inductive_coeff_of_coupling k=0.6, inter-line capacitance cm=3e-14,
* length=6.3. Derived parameters are: lm=5.4e-09, ctot=3.5e-13.
* 
* It is important to note that the model is a simplified one - the
* following assumptions are made: 1. The self-inductance l, the
* self-capacitance ctot (note: not c), the series resistance r and the
* parallel capacitance g are the same for all lines, and 2. Each line
* is coupled only to the two lines adjacent to it, with the same
* coupling parameters cm and lm. The first assumption implies that edge
* effects have to be neglected. The utility of these assumptions is
* that they make the sL+R and sC+G matrices symmetric, tridiagonal and
* Toeplitz, with useful consequences (see "Efficient Transient
* Simulation of Lossy Interconnect", by J.S.  Roychowdhury and
* D.O Pederson, Proc. DAC 91).

* It may be noted that a symmetric two-conductor line is
* represented accurately by this model.

* Subckt node convention:
* 
*            |--------------------------|
*      1-----|                          |-----n+1
*      2-----|                          |-----n+2
*         :  |   n-wire multiconductor  |  :
*         :  |          line            |  :
*    n-1-----|(node 0=common gnd plane) |-----2n-1
*      n-----|                          |-----2n
*            |--------------------------|


* Lossy line models
.model mod1_test ltra rel=1.2 nocontrol r=0.3 l=2.62616456193e-10 g=0 c=3.98541019688e-13 len=6.3
.model mod2_test ltra rel=1.2 nocontrol r=0.3 l=5.662616446e-09 g=0 c=3.68541019744e-13 len=6.3
.model mod3_test ltra rel=1.2 nocontrol r=0.3 l=1.23373835171e-08 g=0 c=3.3145898046e-13 len=6.3
.model mod4_test ltra rel=1.2 nocontrol r=0.3 l=1.7737383521e-08 g=0 c=3.01458980439e-13 len=6.3

* subcircuit m_test - modal transformation network for test
.subckt m_test 1 2 3 4 5 6 7 8
v1 9 0 0v
v2 10 0 0v
v3 11 0 0v
v4 12 0 0v
f1 0 5 v1 0.371748033738
f2 0 5 v2 -0.601500954587
f3 0 5 v3 0.601500954587
f4 0 5 v4 -0.371748036544
f5 0 6 v1 0.60150095443
f6 0 6 v2 -0.371748035044
f7 0 6 v3 -0.371748030937
f8 0 6 v4 0.601500957402
f9 0 7 v1 0.601500954079
f10 0 7 v2 0.37174803072
f11 0 7 v3 -0.371748038935
f12 0 7 v4 -0.601500955482
f13 0 8 v1 0.371748035626
f14 0 8 v2 0.601500956073
f15 0 8 v3 0.601500954504
f16 0 8 v4 0.371748032386
e1 13 9 5 0 0.371748033909
e2 14 13 6 0 0.601500954587
e3 15 14 7 0 0.601500955639
e4 1 15 8 0 0.371748036664
e5 16 10 5 0 -0.60150095443
e6 17 16 6 0 -0.371748035843
e7 18 17 7 0 0.371748032386
e8 2 18 8 0 0.601500957319
e9 19 11 5 0 0.601500955131
e10 20 19 6 0 -0.371748032169
e11 21 20 7 0 -0.371748037896
e12 3 21 8 0 0.601500954513
e13 22 12 5 0 -0.371748035746
e14 23 22 6 0 0.60150095599
e15 24 23 7 0 -0.601500953534
e16 4 24 8 0 0.371748029317
.ends m_test

* Subckt test
.subckt test 1 2 3 4 5 6 7 8
x1 1 2 3 4 9 10 11 12 m_test
o1 9 0 13 0 mod1_test
o2 10 0 14 0 mod2_test
o3 11 0 15 0 mod3_test
o4 12 0 16 0 mod4_test
x2 5 6 7 8 13 14 15 16 m_test
.ends test
*
x1  5 6 7 8 9 10 11 12 test
*
*
vdd    1    0   PULSE (0 5 0Ns 0.1Ns 0.1Ns 600Ns 800Ns)
v3     3    0   PULSE (0 5 0Ns 0.1Ns 0.1Ns 600Ns 800Ns)

.model mn0p9 nmos  LEVEL=1 vto=0.8V kp=48u gamma=0.3 phi=0.55 lambda=0.00
+                 PHI=0.55 LAMBDA=0.00 CGSO=0 CGDO=0 CGBO=0
+                 CJ=0 CJSW=0 TOX=18000N NSUB=1E16 LD=0.0U

.model mp1p0 pmos  vto=-0.8V kp=21u gamma=0.45 phi=0.61 lambda=0.00
+                 PHI=0.61 LAMBDA=0.00 CGSO=0 CGDO=0 CGBO=0
+                 CJ=0 CJSW=0 TOX=18000N NSUB=3E16 LD=0.0U 

VS1  2   0  PULSE (0 5 15.9Ns 0.2Ns 0.2Ns 15.8Ns 32Ns)
VS2  4   0  PULSE (0 5 15.9Ns 0.2Ns 0.2Ns 15.8Ns 32Ns)

.control
option noinit
TRAN 0.1N 47.9N
rusage
plot v(5) v(6) v(7) v(8) v(9) v(10) v(11) v(12)
.endc
*

.END

