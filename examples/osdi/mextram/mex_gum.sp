MEXTRAM Gummel Test Ic,b,s=f(Vc,Ib)

.include Modelcards/mex_model.lib

VB B 0 0.5
VC C 0 2.0
VS S 0 0.0
NQ1 C B 0 S dt BJTRF1

.control
* pre_osdi ../osdi_libs/bjt504t.osdi
dc vb 0.2 1.4 0.01
set xbrushwidth=2
plot abs(i(vc)) abs(i(vb)) abs(i(vs)) ylog xlimit 0.3 1.4 ylimit 1e-12 100e-3
plot abs(i(vc))/abs(i(vb)) vs abs(-i(vc)) xlog xlimit 1e-09 10e-3 ylimit 0 150
.endc


.end
