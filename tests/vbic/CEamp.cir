VBIC Pole Zero Test

Vcc    3 0 5
Rc     2 3 1k
Rb     3 1 200k
I1     0 1 AC 1 DC 0
Vmeas  4 2 DC 0
Cshunt 4 0 .1u

Q1  2 1 0 0 N1

.OPTIONS NOACCT

.ac dec 100 0.1Meg 10G
.print ac db(i(vmeas))
.print ac ph(i(vmeas))
.pz 1 0 4 0 cur pz
.print pz all


.MODEL N1 NPN LEVEL=4 
+ IS=1e-16 IBEI=1e-18 IBEN=5e-15 IBCI=2e-17 IBCN=5e-15 ISP=1e-15 RCX=10
+ RCI=60 RBX=10 RBI=40 RE=2 RS=20 RBP=40 VEF=10 VER=4 IKF=2e-3 ITF=8e-2
+ XTF=20 IKR=2e-4 IKP=2e-4 CJE=1e-13 CJC=2e-14 CJEP=1e-13 CJCP=4e-13 VO=2
+ GAMM=2e-11 HRCF=2 QCO=1e-12 AVC1=2 AVC2=15 TF=10e-12 TR=100e-12 TD=2e-11 RTH=300

.END
