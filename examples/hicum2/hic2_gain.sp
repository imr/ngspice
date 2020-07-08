HICUM2v2.40 AC gain Test h21 = f(Ic) Vce=1V

vce 1 0 dc 1.0
vgain 1 c dc 0.0
f 0 2 vgain -2
l 2 b 1g
c 2 0 1g
ib 0 b dc 0.0 ac 1.0
ic 0 c 0.001
Q1 C B 0 hicumL2V2p40

.control
foreach myic 2e-03 4e-03 7e-03 9e-03 18e-03 33e-3
 alter ic = $myic
 op
 print all
 ac dec 10 1Meg 800g
end
plot abs(ac1.vgain#branch) abs(ac2.vgain#branch) abs(ac3.vgain#branch) abs(ac4.vgain#branch) abs(ac5.vgain#branch) abs(ac6.vgain#branch) ylimit 0.1 300 loglog
.endc

.include model-card-examples.lib

.end
