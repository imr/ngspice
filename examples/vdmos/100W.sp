100W VDMOS power amplifier
*100W into 8Ω at less than .1% THD
*72° phase margin @ 950kHz
*Adjust R7 for 15mA quiescent current through Q1/Q2
*R24 & R25 are optional output offset trimming
*
VTamb tamb 0 25
MQ1 +V N010 N012 tn IRFP240
X1 tn tamb junction-ambient
MQ2 -V N020 N017 tp IRFP9240
X2 tp tamb junction-ambient
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
R7 N009 N019 890
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
C4 +V N007 47µ
R19 OUT1 N011 27K
R20 N011 N018 1K
C5 N018 0 100µ
C6 N015 0 330p
R21 N015 N014 2.2K
R22 N014 0 47K
C7 N014 N013 2.2µ
Vin N013 0 ac 0 dc 0 SINE(0 {V} 1K)
RLOAD OUT 0 8
R24 +V N011 3.67Meg
R25 N011 -V 6.1Meg
V3 OUT OUT1 dc 0 ac 1
C8 OUT1 N011 3p
*
.param V=1.44 ; 100W RMS
.save @r1[i] @r2[i] v(out1) v(out) @rload[i] v(tn) v(tp) inoise_spectrum
.control
op
print v(out) @r1[i] @r2[i]
*dc temp -55 125 1
ac dec 100 10 10Meg
plot db(V(out)/V(out1))
set units=degrees
plot ph(V(out)/V(out1))
tran 1u 1000m
fourier 1K V(out)
plot v(out)*@rload[i]
plot v(tn) v(tp)
linearize v(out)
fft v(out)
plot db(v(out)) xlimit 0 20k
alter v3 ac = 0
alter vin ac = 1
noise V(out) Vin oct 10 10 20K
setplot noise2
plot inoise_spectrum
*step oct param V 1m 1.44 2
.endc
*
.model IRFP240 VDMOS(nchan Rg=3 Vto=4 Rd=72m Rs=18m Rb=36m Kp=4.9 Lambda=.03 Cgdmax=1.34n Cgdmin=.1n Cgs=1.25n Cjo=1.25n Is=67p ksubthres=.1
+ shmod=1 RTH0=1.1k CTH0=1e-3 MU=1.27 texp0=1.5 texp1=0.3 tcvth=0.00625)
.model IRFP9240 VDMOS(pchan Rg=3 Vto=-4 Rd=200m Rs=50m Rb=100m Kp=8.2 Lambda=.10 Cgdmax=1.8n Cgdmin=.07n Cgs=.77n Cjo=.77n Is=76p ksubthres=.1
+ shmod=1 RTH0=1.1k CTH0=1e-3 MU=1.27 texp0=1.5 texp1=0.3 tcvth=0.00625)
*
.model MJE340 NPN(Is=1.03431e-13 BF=172.974 NF=.939811 VAF=27.3487 IKF=0.0260146 ISE=4.48447e-11 Ne=1.61605 Br=16.6725
+ Nr=0.796984 VAR=6.11596 IKR=0.10004 Isc=9.99914e-14 Nc=1.99995 RB=1.47761 IRB=0.2 RBM=1.47761 Re=0.0001 RC=1.42228
+ XTB=2.70726 XTI=1 Eg=1.206 CJE=1e-11 VJE=0.75 Mje=.33 TF=1e-09 XTF=1 VTF=10 ITF=0.01 CJC=1e-11 VJC=.75 MJC=0.33 XCJC=.9
+ Fc=0.5 CJS=0 VJS=0.75 MJS=0.5 TR=1e-07 PTF=0 KF=0 AF=1)
.model MJE350 PNP(Is=6.01619e-15 BF=157.387 NF=.910131 VAF=23.273 IKF=0.0564808 Ise=4.48479e-12 Ne=1.58557 BR=0.1
+ NR=1.03823 VAR=4.14543 IKR=.0999978 ISC=1.00199e-13 Nc=1.98851 RB=.1 IRB=0.202965 RBM=0.1 Re=.0710678 Rc=.355339
+ XTB=1.03638 XTI=3.8424 Eg=1.206 Cje=1e-11 Vje=0.75 Mje=0.33 TF=1e-09 XTF=1 VTF=10 ITF=0.01 Cjc=1e-11 Vjc=0.75
+ Mjc=0.33 XCJC=0.9 Fc=0.5 Cjs=0 Vjs=0.75 Mjs=0.5 TR=1e-07 PTF=0 KF=0 AF=1)
*
.subckt junction-ambient jct amb
rjc jct 1 0.4
ccs 1 0 5m
rcs 1 2 0.2
csa 2 0 30m
rsa 2 amb 1.3
.ends

.end

