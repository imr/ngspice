HICUM2v2.40 Gummel Test invers Ie,b,s=f(Ve,Ib) Vec=1V

VB B 0 1.2
VE E 0 1.0
VS S 0 0.0

Q1 0 B E S hicumL2V2p40

.control
dc vb 0.4 1.2 0.01
plot abs(i(ve)) abs(i(vb)) abs(i(vs)) xlimit 0.4 1.2 ylog ylimit 1e-12 0.1
plot abs(i(ve))/abs(i(vb)) vs abs(i(ve)) xlog xlimit 1e-06 100e-3 ylimit 0 40
.endc

.include model-card-examples.lib

.end
