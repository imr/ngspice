** npn bipolar: table generator with q 2D (Vce, Ib)
* This file may be run by 'ngspice table-generator-q-2d.sp'
* It will generate a 2D data table by simulating the bipolar collector current 
* as function of collector voltage and base current. The simulation uses
* the ngspice bipolar model and model parameters of a clc409 transitor.
* This table is an input file for the XSPICE 2D table model.
* You may change the step sizes vcstep ibstep in CSPARAM
* to obtain the required resolution for the data.
* These tables will contain pure dc data. For transient simulation you may
* need to add some capacitors to the device model for a 'real world' simulation.

*NPN
.csparam vcstart=-0.2
.csparam vcstop=6.4
.csparam vcstep=0.05
.csparam ibstart=-0.1u
.csparam ibstop=200u
.csparam ibstep=0.1u


** Circuit Description **
Q3 2 1 3 QINN

ib 0 11 2u
Rb 11 1 1
vce 2 0 5
vee 3 0 0

.control
** output file **
set outfile = "$inputdir/qinn-clc409-2d-1.table"
dc vce -0.1 6 0.05 ib -0.1u 2u 0.1u
if not $?batchmode
plot i(vee)
plot v(1) ylimit -0.2 0.8
endif
if (1)
*goto next
echo *table for bipolar qinn CLC409 > $outfile

let xcount = floor((vcstop-vcstart)/vcstep) + 1
let ycount = floor((ibstop-ibstart)/ibstep) + 1
echo *x >> $outfile
echo $&xcount >> $outfile
echo *y >> $outfile
echo $&ycount >> $outfile
let xvec = vector(xcount)
let yvec = vector(ycount)

let loopx = vcstart
let lcx=0
while lcx < xcount
  let xvec[lcx] = loopx
  let loopx = loopx + vcstep
  let lcx = lcx + 1
end
echo *x row >> $outfile
echo $&xvec >> $outfile

let lcy=0
let loopy = ibstart
while lcy < ycount
  let yvec[lcy] = loopy
  let loopy = loopy + ibstep
  let lcy = lcy + 1
end
echo *y column >> $outfile
echo $&yvec >> $outfile

let lcy=0
let loopy = ibstart
while lcy < ycount
  alter ib loopy
  dc vce $&vcstart $&vcstop $&vcstep
*  let lcx=0
*  let loopx = vdstart
*  dowhile loopx le vdstop
*    alter vds loopx
*    op
*    let xvec[lcx] = i(vss)
*       destroy i(vss)
*    let loopx = loopx + vdstep
*    let lcx = lcx + 1
*  end
  let xvec = i(vee)
  echo $&xvec >> $outfile
  destroy dc2
  let loopy = loopy + ibstep
  let lcy = lcy + 1
end

label next
end

.endc

.MODEL QINN NPN
+ IS =0.166f    BF =3.239E+02 NF =1.000E+00 VAF=8.457E+01
+ IKF=2.462E-02 ISE=2.956E-17 NE =1.197E+00 BR =3.719E+01
+ NR =1.000E+00 VAR=1.696E+00 IKR=3.964E-02 ISC=1.835E-19
+ NC =1.700E+00 RB =118       IRB=0.000E+00 RBM=65.1
+ RC =2.645E+01 CJE=1.632E-13 VJE=7.973E-01
+ MJE=4.950E-01 TF =1.948E-11 XTF=1.873E+01 VTF=2.825E+00
+ ITF=5.955E-02 PTF=0.000E+00 CJC=1.720E-13 VJC=8.046E-01
+ MJC=4.931E-01 XCJC=589m     TR =4.212E-10 CJS=629f
+ MJS=0         KF =2.000E-12 AF =1.000E+00 FC =9.765E-01
*

.end
