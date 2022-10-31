HICUM0 Output Test Ic=f(Vc,Ib)

IB 0 B 200n
VC C 0 2.0
VS S 0 0.0
X1 C B 0 S DT hicumL0V1p1_c_sbt

Rdt dt 0 1G

.control
pre_osdi test_osdi_win/HICUML0-2.osdi
dc vc 0.0 3.0 0.05 ib 10u 100u 10u
run
plot abs(i(vc))
plot v(dt)
.endc

.include Modelcards/model-card-hicumL0V1p11_mod.lib 

.end
