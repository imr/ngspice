* subcircuit model file

.include modelcard.nmos
.include modelcard.pmos

.subckt NCH D G S B W=1 L=1
MN1 D G S B N1 W={W} L={L} AS={3*L*W} AD={3*L*W} PS={6*L+W} PD={6*L+W}
.ends


.subckt PCH D G S B W=1 L=1
MP1 D G S B P1 W={W} L={L} AS={3*L*W} AD={3*L*W} PS={6*L+W} PD={6*L+W}
.ends
