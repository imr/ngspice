*****  NMOS Transistor BSIM3 (Id-Vds) with Rd ***

M1 2 1 3 4 n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vgs 1 0 3.5 
vds 2 0 0.1 
vss 3 0 0
vbs 4 0 0

* drain series resistor
R2 2 22 1k
M2 22 1 32 4 n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vss2 32 0 0


.options Temp=27.0

* BSIM3v3.3.0 model with modified default parameters 0.18µm
.model n1 nmos level=49 version=3.3.0 tox=3.5n nch=2.4e17 nsub=5e16 vth0=0.15
.model p1 pmos level=49 version=3.3.0 tox=3.5n nch=2.5e17 nsub=5e16 vth0=-0.15

.control
* sim
dc vds 0 2 0.05 vgs 0 2 0.4

* plot
set xgridwidth=2
set xbrushwidth=3

set nolegend
plot vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'
unset nolegend
set color0=white
plot vss2#branch vs v(22) title 'Series resistor: Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'

*** postscript ***
set hcopydevtype = postscript
set hcopypscolor=0 ; background black
hardcopy plot_4.ps vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'
set hcopypscolor=1 ; background white
hardcopy plot_5.ps vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'
* for MS Windows only
if $oscompiled = 1 | $oscompiled = 8
  shell Start /B plot_4.ps
  shell Start /B plot_5.ps
* for CYGWIN
else
  shell xterm -e gs  plot_5.ps &
end

*** svg ***
set hcopydevtype = svg
*set color0=white
set color1=blue
hardcopy plot_4.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'
*set color0=black
set color1=orange
hardcopy plot_5.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'
* for MS Windows only
if $oscompiled = 1 | $oscompiled = 8
  shell Start /B plot_4.svg
  shell Start /B plot_5.svg
*  shell Start /B plot_6.svg
* for CYGWIN
else
  shell xterm -e gs  plot_5.svg &
end
.endc

.end





