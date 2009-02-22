* I-V Characteristics of JFET 2N4221
*
*
j1 2 1 0 MODJ
VD 2 0 25
VG 1 0 -2
*
.model MODJ NJF LEVEL=1 VTO=-3.5 BETA=4.1E-4 LAMBDA=0.002 RD=200
*
.options noacct
.op
.dc VD 0 25 1 VG -3 0 1
.print DC I(VD)
.END
