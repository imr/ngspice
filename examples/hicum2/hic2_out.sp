HICUM2v2.40 Output Test Ic=f(Vc,Ib)

IB 0 B 1u
VC C 0 1.8
VS S 0 0.0

Q1 C B 0 S tj hicumL2V2p40

.control
dc vc 0.0 1.8 0.01 ib 10u 100u 10u
plot -i(vc)
reset
altermod @hicumL2V2p40[flsh]=1
dc vc 0.0 1.8 0.01 ib 1u 10u 1u
plot -i(vc)
plot v(tj)
.endc

.include model-card-examples.lib

.end
