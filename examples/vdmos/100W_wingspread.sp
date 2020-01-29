VDMOS wingspread plot example

M1 +V N004 N005 IRFP240
M2 -V N009 N007 IRFP9240
R1 OUT N007 .33
R2 N005 OUT .33
R4 N004 N003 470
R5 N009 N008 470
V1 +V 0 50
V2 0 -V 50
Q3 -V N011 N008 0 MJE350
R7 N003 N008 870
Q5 +V N002 N003 0 MJE340
Vin N006 0 0
RLoad OUT 0 r = 8
V3 N001 N006 4.8
V4 N006 N010 4.8
I1 +V N001 12m
I2 N010 -V 12m
R3 N002 N001 10
R8 N011 N010 10
*
.save all @r1[i] @r2[i] v(out) @rload[i]
.control

let gain=vector(2005)
reshape gain [5][401]
let irload=vector(2005)
reshape irload [5][401]

let offset = 0.05

foreach Rl 4 6 8

  setplot new
  set curplottitle = "wingspread $Rl Ohm"
  set plotname=$curplot

  alter Rload r = $Rl

  let index = 0

  foreach vbias 4.7 4.8 4.9 5.0 5.1
    alter v3 dc = $vbias + offset
    alter v4 dc = $vbias - offset
    op
    print v(out) @r1[i] @r2[i]
    dc vin -20 20 0.1
    set dcplotname = $curplot
    setplot $plotname
    let gain[index] = deriv({$dcplotname}.out)
    let irload[index] = {$dcplotname}.@rload[i]
    let index = index + 1
    destroy $dcplotname
  end

  settype current irload
  plot gain[0] gain[1] gain[2] gain[3] gain[4] vs irload[2]

end

.endc
*
.model IRFP240 VDMOS nchan
+ Vto=4 Kp=5.9 Lambda=.001 Theta=0.015 ksubthres=.27
+ Rd=61m Rs=18m Rg=3 Rds=1e7
+ Cgdmax=2.45n Cgdmin=10p a=0.3 Cgs=1.2n
+ Is=60p N=1.1 Rb=14m XTI=3
+ Cjo=1.5n Vj=0.8 m=0.5
+ tcvth=0.0065 MU=-1.27 texp0=1.5
*+ Rthjc=0.4 Cthj=5e-3
+ mtriode=0.8
.model IRFP9240 VDMOS pchan
+ Vto=-4 Kp=8.8 Lambda=.003 Theta=0.08 ksubthres=.35
+ Rd=180m Rs=50m Rg=3 Rds=1e7
+ Cgdmax=1.25n Cgdmin=50p a=0.23 Cgs=1.15n
+ Is=150p N=1.3 Rb=16m XTI=2
+ Cjo=1.3n Vj=0.8 m=0.5
+ tcvth=0.004 MU=-1.27 texp0=1.5
*+ Rthjc=0.4 Cthj=5e-3
+ mtriode=0.6
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
.end
