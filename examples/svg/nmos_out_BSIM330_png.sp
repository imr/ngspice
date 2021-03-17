*****  NMOS (Id-Vds) png and gnuplot file ***

M1 2 1 3 4 n1 W=1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vgs 1 0 3.5 
vds 2 0 0.1 
vss 3 0 0
vbs 4 0 0

* drain series resistor
R2 2 22 0.1
M2 22 1 32 4 n1 W=1.1u L=0.35u Pd=1.5u Ps=1.5u ad=1.5p as=1.5p
vss2 32 0 0

.options Temp=27.0

* BSIM3v3.3.0 model with modified default parameters 0.18µm
.model n1 nmos level=49 version=3.3.0 tox=3.5n nch=2.4e17 nsub=5e16 vth0=0.15
.model p1 pmos level=49 version=3.3.0 tox=3.5n nch=2.5e17 nsub=5e16 vth0=-0.15

.control
* sim
dc vds 0 2 0.05 vgs 0 2 0.4

set xbrushwidth=3
set xgridwidth=2

* no gnuplot window, only png file
set gnuplot_terminal=png/quit
gnuplot plot_1 vss#branch vss2#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage / V' ylabel 'Drain current / µA'
* plot vss#branch vss2#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage / V' ylabel 'Drain current / µA'

unset gnuplot_terminal
set nolegend
* only the gnuplot window, no gnuplot files
gnuplot temp vss#branch vss2#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage / V' ylabel 'Drain current / µA'

* MS Windows
if $oscompiled = 1 | $oscompiled = 8
  shell Start c:\"program files"\irfanview\i_view64.exe plot_1.png
else
  if $oscompiled = 7
* macOS (using feh from homebrew)
    shell feh --conversion-timeout 1  plot_1.png &
  else
* for CYGWIN, Linux
    shell feh --magick-timeout 1  plot_1.png &
  end
end
.endc

.end





