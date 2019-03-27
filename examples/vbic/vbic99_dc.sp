***** VBIC99 level9 DC test *****
.OPTION gmin=1.0e-15
vbe bx 0 0
vcb cx bx 0
vib bx b 0
vic cx c 0
ve ex 0 0
vie ex e 0
vs sx 0 0
vis sx s 0
q1 c b e s dt vbic99_dc area=1 m=1
.include bjt_vbic.mod
.temp 27
.control
dc vbe 0.1 1.1 0.02
plot i(vib) i(vic) abs(i(vis)) ylog
plot v(dt)
.endc
.end
