Plot inner small signal parameter

v1 1 0 dc 10.0
rc 1 c 2k
vb b 0 dc 0.6
q1 c b 0 bfs17

.MODEL BFS17 NPN (level=1 IS=0.48F NF=1.008 BF=99.655 VAF=90.000 IKF=0.190
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

.control
save @q1[gm] @q1[pi] @q1[go]
save @q1[qbe] @q1[qbc]
save @q1[cmu] @q1[cpi]
dc vb 0.4 1 0.01
plot @q1[gm] @q1[gpi] @q1[go]
plot @q1[qbe] @q1[qbc]
plot @q1[cmu] @q1[cpi]
.endc

.end
