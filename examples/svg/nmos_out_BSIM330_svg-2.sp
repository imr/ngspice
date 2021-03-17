*****  NMOS (Id-Vds) SVG plot file 2 ***

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

set nolegend

* the default settings
* "svgwidth", "svgheight",  "svgfont-size", "svgfont-width", "svguse-color", "svgstroke-width", "svggrid-width",
set svg_intopts = ( 512 384 16 8 1 5 2 )
* "svgbackground", "svgfont-family", "svgfont"
setcs svg_stropts = ( yellow Arial Arial )



*** svg ***
set hcopydevtype = svg
*set color0=white
set color1=blue
set color2=green
set nounits

hardcopy plot_10.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage / V' ylabel 'Drain current / µA'

unset svg_intopts
set svg_intopts = ( 512 384 6 0 1 5 2 )
hardcopy plot_11.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage / V' ylabel 'Drain current / µA'


* for MS Windows only
if $oscompiled = 1 | $oscompiled = 8
  shell Start plot_10.svg
  shell Start plot_11.svg
else
  if $oscompiled = 7
* macOS (using feh from homebrew)
    shell feh --conversion-timeout 1  plot_10.svg &
    shell feh --conversion-timeout 1  plot_11.svg &
  else
* for CYGWIN, Linux
    shell feh --magick-timeout 1  plot_10.svg &
    shell feh --magick-timeout 1  plot_11.svg &
  end
end
.endc

.end





