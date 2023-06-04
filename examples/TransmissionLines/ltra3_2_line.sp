MOSdriver -- 24inch 2 lossy lines LTRA model -- C load 

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
* Subcircuit test
* test is a subcircuit that models a 2-conductor transmission line with
* the following parameters: l=9.13e-09, c=2.75e-13, r=0.2, g=0,
* inductive_coeff_of_coupling k=0.36144, inter-line capacitance cm=9e-14,
* length=24. Derived parameters are: lm=3.29995e-09, ctot=3.65e-13.
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
.model mod1_test ltra rel=1.2 nocontrol r=0.2 l=5.83005279316e-09 g=0 c=4.55000000187e-13 len=24
.model mod2_test ltra rel=1.2 nocontrol r=0.2 l=1.24299471863e-08 g=0 c=2.75000000373e-13 len=24

* subcircuit m_test - modal transformation network for test
.subckt m_test 1 2 3 4
v1 5 0 0v
v2 6 0 0v
f1 0 3 v1 0.707106779721
f2 0 3 v2 -0.707106782652
f3 0 4 v1 0.707106781919
f4 0 4 v2 0.707106780454
e1 7 5 3 0 0.707106780454
e2 1 7 4 0 0.707106782652
e3 8 6 3 0 -0.707106781919
e4 2 8 4 0 0.707106779721
.ends m_test

* Subckt test
.subckt test 1 2 3 4
x1 1 2 5 6 m_test
o1 5 0 7 0 mod1_test
o2 6 0 8 0 mod2_test
x2 3 4 7 8 m_test
.ends test
*
x1  648 748  651 751  test
*
*
vdd    1    0   DC  5.0
VK   267    0   DC  5.0
*
VS1 168  0  PULSE (0 5 15.9N 0.2N 0.2N 15.8N 60N)
VS2 268  0  PULSE (0 5 15.9N 0.2N 0.2N 15.8N 60N)
*
.control
TRAN 0.2N 47.9NS
rusage
set color0=white
set xbrushwidth=3
PLOT v(648) v(651) v(751)
.endc
*
.model mn0p9 nmos  LEVEL=1 vto=0.8V kp=48u gamma=0.3 phi=0.55 lambda=0.0
+                 PHI=0.55 LAMBDA=0.00 CGSO=0 CGDO=0 CGBO=0
+                 CJ=0 CJSW=0 TOX=18000N NSUB=1E16 LD=0.0U

.model mp1p0 pmos  LEVEL=1 vto=-0.8V kp=21u gamma=0.45 phi=0.61 lambda=0.0
+                 PHI=0.61 LAMBDA=0.00 CGSO=0 CGDO=0 CGBO=0
+                 CJ=0 CJSW=0 TOX=18000N NSUB=3E16 LD=0.0U

.END
