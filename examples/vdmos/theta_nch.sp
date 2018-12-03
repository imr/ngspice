VDMOS Mobility Reduction
M1 D G 0 IRFP240
VG G 0 9V
VD D 0 1V
.control
foreach myTHETA 0.1 0.2 0.3 0.4 0.5
 altermod @IRFP240[THETA]=$myTHETA
 dc VG 2 9 0.1
end
plot abs(dc1.vd#branch) abs(dc2.vd#branch) abs(dc3.vd#branch) abs(dc4.vd#branch) abs(dc5.vd#branch)
.endc
.model IRFP240 VDMOS(Rg=3 Vto=4 Rd=72m Rs=18m Rb=36m Kp=4.9 Lambda=.03 Cgdmax=1.34n Cgdmin=.1n Cgs=1.25n Cjo=1.25n Is=67p ksubthres=.1 Vds=200 Ron=180m Qg=70n)
.end
