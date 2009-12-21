BJT ft Test

vce 1 0 dc 3.0
vgain 1 c dc 0.0
f 0 2 vgain -1000
l 2 b 1g
c 2 0 1g
ib 0 b dc 0.0 ac 1.0
ic 0 c 0.01
q1 c b 0 bfs17

.control
foreach myic 0.5e-3 1e-3 5e-3 10e-3 50e-3 100e-3
 alter ic = $myic
 ac dec 10 10k 5g
end
*foreach mytf 50p 100p 150p 200p 250p 300p
* altermod q.x1.q1 tf = $mytf
* ac dec 10 10k 5g
*end
plot abs(ac1.vgain#branch) abs(ac2.vgain#branch) abs(ac3.vgain#branch) abs(ac4.vgain#branch) abs(ac5.vgain#branch) abs(ac6.vgain#branch) ylimit 0.1 100 loglog
.endc

*****************************************************************
* SPICE2G6 MODEL OF THE NPN BIPOLAR TRANSISTOR BFS17 (SOT-23)   *
* REV: 98.1                  DANALYSE GMBH BERLIN (27.07.1998)  *
*****************************************************************
.SUBCKT BFS17C 1 2 3
Q1   6 5 7 BFS17 1.000
LC   1 6 0.350N
L1   2 4 0.400N
LB   4 5 0.500N
L2   3 8 0.400N
LE   8 7 0.600N
CGBC 4 6 70.00F
CGBE 4 8 0.150P
CGCE 6 8 15.00F
.ENDS
.MODEL BFS17 NPN (level=1 IS=0.480F NF=1.008 BF=99.655 VAF=90.000 IKF=0.190
+ ISE=7.490F NE=1.762 NR=1.010 BR=38.400 VAR=7.000 IKR=93.200M
+ ISC=0.200F NC=1.042
+ RB=1.500 IRB=0.100M RBM=1.200
+ RE=0.500 RC=2.680
+ CJE=1.325P VJE=0.700 MJE=0.220 FC=0.890
+ CJC=1.050P VJC=0.610 MJC=0.240 XCJC=0.400
+ TF=56.940P TR=1.000N PTF=21.000
+ XTF=68.398 VTF=0.600 ITF=0.700
+ XTB=1.600 EG=1.110 XTI=3.000
+ KF=1.000F AF=1.000)

.end
