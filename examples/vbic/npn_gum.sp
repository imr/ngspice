VBIC Gummel Test Ic=f(Vc,Vb)

.include qnva.mod

VB B 0 0.5
VC C 0 1.0
VS S 0 0.0
XQ1 C B 0 S qnva

.control
options gmin=1e-15
dc vb 0.2 1.2 0.01
plot abs(i(vc)) abs(i(vb)) abs(i(vs)) ylimit 0.1e-12 100e-3 ylog
plot abs(i(vc))/abs(i(vb)) vs abs(-i(vc)) xlog xlimit 10e-12 10e-3 ylimit 0 40
.endc

.end
