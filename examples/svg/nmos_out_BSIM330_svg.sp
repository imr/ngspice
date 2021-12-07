*****  NMOS (Id-Vds) SVG plot file ***

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

* the default settings
* "svgwidth", "svgheight",  "svgfont-size", "svgfont-width", "svguse-color", "svgstroke-width", "svggrid-width",
set svg_intopts = ( 512 384 16 0 1 2 0 )
* "svgbackground", "svgfont-family", "svgfont"
setcs svg_stropts = ( blue Arial Arial )

*** svg ***
set hcopydevtype = svg

hardcopy plot_0.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'

set color0=white
set color1=blue
set color2=green
hardcopy plot_1.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'

set svg_intopts = ( 512, 384, 14, 0, 1, 2, 0 )

set color0=blue
set color1=white
set color2=red
hardcopy plot_2.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'

set svg_intopts = ( 512, 384, 12, 0, 0, 2, 2 )

set color0=black
set color1=yellow
set color2=white
hardcopy plot_3.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'

* reset the colors
set svg_intopts = ( 512, 384, 12, 0, 1, 2, 2 )

unset color0
unset color1
unset color2
hardcopy plot_4.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'

* choose backgroundfrom color list
set color0="#F0E68C"
set color1="#DDA0DD"
set color2"#EE82EE"
* set width and hight
set hcopywidth=1024
set hcopyheight=768
hardcopy plot_5.svg vss#branch title 'Drain current versus drain voltage' xlabel 'Drain voltage' ylabel 'Drain current'


* for MS Windows only
if $oscompiled = 1 | $oscompiled = 8
  shell Start plot_0.svg
  shell Start plot_1.svg
  shell Start plot_2.svg
  shell Start plot_3.svg
  shell Start plot_4.svg
  shell Start plot_5.svg
else
  if $oscompiled = 7
* macOS (using feh and ImageMagick from homebrew)
    shell feh --conversion-timeout 1  plot_0.svg &
    shell feh --conversion-timeout 1  plot_1.svg &
    shell feh --conversion-timeout 1  plot_2.svg &
    shell feh --conversion-timeout 1  plot_3.svg &
    shell feh --conversion-timeout 1  plot_4.svg &
    shell feh --conversion-timeout 1  plot_5.svg &
  else
* for CYGWIN, Linux
    shell feh --magick-timeout 1  plot_0.svg &
    shell feh --magick-timeout 1  plot_1.svg &
    shell feh --magick-timeout 1  plot_2.svg &
    shell feh --magick-timeout 1  plot_3.svg &
    shell feh --magick-timeout 1  plot_4.svg &
    shell feh --magick-timeout 1  plot_5.svg &
  end
end
.endc

.end





