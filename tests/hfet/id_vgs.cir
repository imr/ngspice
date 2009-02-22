HFET Id versus Vgs characteristic

z1 1 2 0 hfet l=1u w=10u
vgs 2 0 dc 0.3
vds 1 0 dc 1.0

.options noacct
.model hfet nhfet level=5 rdi=0 rsi=0 m=2.57 lambda=0.17
+ vs=1.5e5 mu=0.385 vt0=0.13 eta=1.32 sigma0=0.04
+ vsigma=0.1 Vsigmat=0.3 js1s=0 js1d=0 nmax=6e15

.dc vds 0 1 0.01
.print DC i(vds)
.end
