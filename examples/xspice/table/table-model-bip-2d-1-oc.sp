** npn bipolar: table 2D (Vce, Ib) compared to q model
* bipolar transistor qinn from National Semi op-amp clc409
* please run the table generator table-generator-q-2d.sp in ngspice to
* create the table data file qinn-clc409-2d-1.table as required here

** Circuit Description **
Q3 2 1 3 QINN
ib 0 1 2u
vce 2 0 5
vee 3 0 0

xbip cc bb ee tbqnpn
ib2 0 bb 2u
vce2 cc 0 1
vee2 ee 0 0

* set a simulation temperature
.options temp=1

.subckt tbqnpn c b e
*** table model of npn bipolar transistor ***
* bip qinn from national op-amp CLC409
* table values extracted at nominal temperature of 27°C
* simple behavioral temperature model
.param fact = 0.05
.param tgain = 1. + (TEMPER / 27. - 1.) * {fact}
abip1 %vd(c e) %id(bint e) %id(c e) biptable1
.model biptable1 table2d (offset=0.0 gain={tgain} order=2 file="qinn-clc409-2d-1.table")
* CJE=1.632E-13
Cje b e 1.632E-13
* CJC=1.720E-13
Cjc b c 1.720E-13
* input diode
Dbe b bint DMOD
.model DMOD D (bv=5 is=1e-17 n=1.1)
.ends


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
