VBIC Output Test Ic=f(Vc,Ib)

.include qnva.mod

IB 0 B 200n
VC C 0 2.0
VS S 0 0.0
XQ1 C B 0 S qnva

.control
dc vc 0.0 5.0 0.05 ib 1u 10u 1u
plot abs(i(vc))
.endc

.end
