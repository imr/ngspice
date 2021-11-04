HICUM0 Gummel Test Ic=f(Vc,Vb)

VB B 0 0.5
VC C 0 1.0
VS S 0 0.0
X1 C B 0 S DT hicumL0V1p1_c_sbt

.control
dc vb 0.2 1.4 0.01
plot abs(i(vc)) abs(i(vb)) abs(i(vs)) ylimit 0.1p 100m ylog
plot abs(i(vc))/abs(i(vb)) vs abs(i(vc)) xlog xlimit 100p 100m  ylimit 0 200 retraceplot
.endc

.include model-card-hicumL0V1p11.lib

.end
