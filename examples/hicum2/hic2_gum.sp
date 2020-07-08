HICUM2v2.40 Gummel Test Ic,b,s=f(Vc,Ib) Vce=1V

VB B 0 1.2
VC C 0 1.0
VS S 0 0.0

Q1 C B 0 S hicumL2V2p40

.control
option gmin=1e-14
dc vb 0.2 1.2 0.01
*plot i(vc) i(vb) i(vs)
*gnuplot fgum i(vc) i(vb) i(vs) xlimit 0.2 1.2 ylog ylimit 1e-12 0.1
plot abs(i(vc)) abs(i(vb)) abs(i(vs)) xlimit 0.2 1.2 ylog ylimit 1e-14 0.1
plot abs(i(vc))/abs(i(vb)) vs abs(i(vc)) xlog xlimit 1e-09 100e-3; ylimit 0 500
.endc

.include model-card-examples.lib

.end
