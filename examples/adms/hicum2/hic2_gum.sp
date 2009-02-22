HICUM2v2.2 Gummel Test Ic,b,s=f(Vc,Ib)

VB B 0 0.5
VC C 0 1.0
VS S 0 0.0
Q1 C B 0 S DT hicumL2V2p2_c_sbt

.control
dc vb 0.2 1.4 0.01
run
plot abs(i(vc)) abs(i(vb)) abs(i(vs)) ylog xlimit 0.3 1.6 ylimit 1e-12 0.1
plot abs(i(vc))/abs(i(vb)) vs abs(-i(vc)) xlog xlimit 1e-09 10e-3 ylimit 0 120
.endc

.include model-card-hicumL2V2p21.lib

.end
