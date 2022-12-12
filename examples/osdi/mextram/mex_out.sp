MEXTRAM Output Test Ic=f(Vc,Ib)

.include Modelcards/mex_model.lib

IB 0 b 1u
VC C 0 2.0
VS S 0 0.0
NQ1 C B 0 S T BJTRF1

.control
pre_osdi test_osdi_win/bjt504t.osdi
dc vc 0 6.0 0.05 ib 1u 8u 1u
plot abs(i(vc)) xlabel Vce title Output-Characteristic
.endc

.end
