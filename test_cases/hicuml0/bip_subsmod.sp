Bip model in subckt Gummel Test Ic=f(Vc,Vb)

VB B 0 0.5
VC C 0 1.0
VS S 0 0.0
X1 C B 0 S bip_default

.control
dc vb 0.2 1.4 0.01
plot abs(i(vc)) abs(i(vb)) abs(i(vs)) ylimit 0.1p 100m ylog
plot abs(i(vc))/abs(i(vb)) vs abs(i(vc)) xlog xlimit 100p 100m  ylimit 0 200 retraceplot
.endc

********************************************************************************
* Complete test transistor: default
********************************************************************************
.subckt bip_default c b e s
qhcm0 c b e s hic0_full
.model  hic0_full npn
.ends hicumL0V11_default
********************************************************************************

.end
