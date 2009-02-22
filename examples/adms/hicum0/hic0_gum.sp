HICUM0 Gummel Test Ic=f(Vc,Vb)

VB B 0 0.5
VC C 0 1.0
VS S 0 0.0
X1 C B 0 S DT hicumL0V1p1_c_sbt

.control
dc vb 0.2 1.4 0.01
run
plot abs(i(vc)) abs(i(vb)) abs(i(vs)) ylimit 0.1e-12 100e-3 ylog
plot abs(i(vc))/abs(i(vb)) vs abs(-i(vc)) xlog xlimit 1e-09 10e-3 ylimit 0 300
.endc

.include model-card-hicumL0V1p11.lib

.end
