MEXTRAM Output Test Ic=f(Vc,Ib)

.include Modelcards/mex_model.lib

IB 0 b 1u
VC C 0 2.0
VS S 0 0.0
NQ1 C B 0 S T BJTRF1

.control
* pre_osdi ../osdi_libs/bjt504t.osdi
dc vc 0 6.0 0.05 ib 0 8u 1u
set xbrushwidth=2
plot abs(i(vc)) xlabel Vce title Output-Characteristic
.endc

.end
