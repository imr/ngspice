100W VDMOS power amplifier
*100W into 8Ω at less than .1% THD
*72° phase margin @ 950kHz
*Adjust R7 for 15mA quiescent current through Q1/Q2
*R24 & R25 are optional output offset trimming
*
VTamb tamb 0 25
MQ1 +V N010 N012 tn tcn IRFP240 thermal
X1 tcn tamb case-ambient
MQ2 -V N020 N017 tp tcp IRFP9240 thermal
X2 tcp tamb case-ambient
R1 OUT N017 .33
R2 N012 OUT .33
C1 OUT N016 100n
R3 N016 0 10
R4 N010 N009 470
R5 N020 N019 470
V1 +V 0 50
V2 0 -V 50
Q3 N009 N006 N005 0 MJE350
Q4 N006 N006 N004 0 MJE350
R6 +V N005 100
R7 N009 N019 820
Q5 N019 N023 N024 0 MJE340
R8 +V N004 100
R9 N024 -V 100
Q6 N022 N021 N024 0 MJE340
C2 N023 N019 18p
C3 N022 N021 18p
R10 N006 N022 10K
Q7 N023 N015 N008 0 MJE350
Q8 N021 N011 N008 0 MJE350
R13 N023 -V 3.9K
R14 N021 -V 3.9K
Q9 N008 N003 N001 0 MJE350
R15 +V N001 470
R16 N002 N001 1K
Q10 N003 N002 +V 0 MJE350
R17 N003 N007 10K
R18 N007 0 10K
C4 +V N007 47u
R19 OUT1 N011 27K
R20 N011 N018 1K
C5 N018 0 100u
C6 N015 0 330p
R21 N015 N014 2.2K
R22 N014 0 47K
C7 N014 N013 2.2u
Vin N013 0 ac 0 dc 0 SINE(0 {V} 1K)
RLOAD OUT 0 8
R24 +V N011 3.7Meg
R25 N011 -V 6.1Meg
V3 OUT OUT1 dc 0 ac 1
C8 OUT1 N011 3p
*
.param V=1.44 ; 100W RMS
.save @r1[i] @r2[i] v(out1) v(out) @rload[i] v(tn) v(tp) v(tcn) v(tcp) inoise_spectrum
.control
op
print v(out) @r1[i] @r2[i]
ac dec 100 1 10Meg
plot db(V(out)/V(out1))
set units=degrees
plot unwrap(ph(V(out)/V(out1)))
tran 1u 1000m
fourier 1K V(out)
plot v(out)*@rload[i]
settype temperature v(tn) v(tp) v(tcn) v(tcp)
plot v(tn) v(tp) v(tcn) v(tcp)
linearize v(out)
fft v(out)
plot db(v(out)) xlimit 0 20k
alter v3 ac = 0
alter vin ac = 1
noise V(out) Vin dec 10 10 100K
setplot noise2
plot inoise_spectrum
.endc
*
.model IRFP240 VDMOS nchan
+ Vto=4 Kp=5.9 Lambda=.001 Theta=0.015 ksubthres=.27
+ Rd=61m Rs=18m Rg=3 Rds=1e7
+ Cgdmax=2.45n Cgdmin=10p a=0.3 Cgs=1.2n
+ Is=60p N=1.1 Rb=14m XTI=3
+ Cjo=1.5n Vj=0.8 m=0.5
+ tcvth=0.0065 MU=-1.27 texp0=1.5
+ Rthjc=0.4 Cthj=5e-3
+ mtriode=0.8
.model IRFP9240 VDMOS pchan
+ Vto=-4 Kp=8.8 Lambda=.003 Theta=0.08 ksubthres=.35
+ Rd=180m Rs=50m Rg=3 Rds=1e7
+ Cgdmax=1.25n Cgdmin=50p a=0.23 Cgs=1.15n
+ Is=150p N=1.3 Rb=16m XTI=2
+ Cjo=1.3n Vj=0.8 m=0.5
+ tcvth=0.004 MU=-1.27 texp0=1.5
+ Rthjc=0.4 Cthj=5e-3
+ mtriode=0.6
*
.model MJE340 NPN(Is=1.03431e-13 BF=172.974 NF=.939811 VAF=27.3487 IKF=0.0260146 ISE=4.48447e-11 Ne=1.61605 Br=16.6725
+ Nr=0.796984 VAR=6.11596 IKR=0.10004 Isc=9.99914e-14 Nc=1.99995 RB=1.47761 IRB=0.2 RBM=1.47761 Re=0.0001 RC=1.42228
+ XTB=2.70726 XTI=1 Eg=1.206 CJE=1e-11 VJE=0.75 Mje=.33 TF=1e-09 XTF=1 VTF=10 ITF=0.01 CJC=1e-11 VJC=.75 MJC=0.33 XCJC=.9
+ Fc=0.5 CJS=0 VJS=0.75 MJS=0.5 TR=1e-07 PTF=0 KF=1e-15 AF=1)
.model MJE350 PNP(Is=6.01619e-15 BF=157.387 NF=.910131 VAF=23.273 IKF=0.0564808 Ise=4.48479e-12 Ne=1.58557 BR=0.1
+ NR=1.03823 VAR=4.14543 IKR=.0999978 ISC=1.00199e-13 Nc=1.98851 RB=.1 IRB=0.202965 RBM=0.1 Re=.0710678 Rc=.355339
+ XTB=1.03638 XTI=3.8424 Eg=1.206 Cje=1e-11 Vje=0.75 Mje=0.33 TF=1e-09 XTF=1 VTF=10 ITF=0.01 Cjc=1e-11 Vjc=0.75
+ Mjc=0.33 XCJC=0.9 Fc=0.5 Cjs=0 Vjs=0.75 Mjs=0.5 TR=1e-07 PTF=0 KF=1e-15 AF=1)
*
.subckt case-ambient case amb
rcs case 1 0.1
csa 1 0 30m
rsa 1 amb 1.3
.ends

.end

