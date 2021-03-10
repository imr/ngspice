*****  NMOS (Id-Vds) PostScript plot file ***

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

* BSIM3v3.3.0 model with modified default parameters 0.18 µm
.model n1 nmos level=49 version=3.3.0 tox=3.5n nch=2.4e17 nsub=5e16 vth0=0.15
.model p1 pmos level=49 version=3.3.0 tox=3.5n nch=2.5e17 nsub=5e16 vth0=-0.15

.control
* sim
dc vds 0 2 0.05 vgs 0 2 0.4

set nolegend
set nounits

set hcopydevtype=postscript

* allow color and set background color if set to value >= 0
set hcopypscolor=1
set hcopypstxcolor=7

set xgridwidth=2
set xbrushwidth=3
set hcopyfontsize=16

run

* Do not use line continuation (+ next line), because
* it will ower case all text after the +
* Use UTF-8 encoding (e.g. to get the µ)
hardcopy plot_1.ps vss#branch  100u + vss#branch title 'Drain Current' ylabel 'Drain Current / µA' xlabel 'Drain Voltage / V'
plot  vss#branch  100u + vss#branch title 'Drain current' ylabel 'Drain current / µA' xlabel 'Drain voltage / V'

* for MS Windows only
if $oscompiled = 1 | $oscompiled = 8
  shell Start /B plot_1.ps
else
* for CYGWIN, Linux
  shell gv plot_1.ps &
end
.endc

.end





