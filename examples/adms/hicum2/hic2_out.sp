HICUM2v2.2 Output Test Ic=f(Vc,Ib)

IB 0 B 200n
VC C 0 3.0
VS S 0 0.0
Q1 C B 0 S DT hicumL2V2p2_c_slh

.control
dc vc 0.0 2.0 0.05 ib 10u 50u 10u
run
plot abs(i(vc))
plot v(dt)
.endc

.include model-card-hicumL2V2p21.lib

.end
