** NMOSFET: table generator with BSIM4 2D (Vdrain, Vgate)
*NMOS
.csparam vdstart=-0.1
.csparam vdstop=1.8
.csparam vdstep=0.05
.csparam vgstart=-0.1
.csparam vgstop=1.8
.csparam vgstep=0.05
.csparam vbstart=-1.8
.csparam vbstop=0.4
.csparam vbstep=0.2

*PMOS
*.csparam vdstart=-1.8
*.csparam vdstop=0.1
*.csparam vdstep=0.05
*.csparam vgstart=-1.8
*.csparam vgstop=0.1
*.csparam vgstep=0.05
*.csparam vbstart=-0.4
*.csparam vbstop=1.8
*.csparam vbstep=0.2

** Circuit Description **
m1 2 1 3 4 n1 L=0.13u W=10.0u rgeoMod=1
*m1 2 1 3 4 p1 L=0.13u W=10.0u rgeoMod=1
vgs 1 0 1.8
vds 2 0 1.8
vss 3 0 0
vbs 4 0 0

.control
** output file **
set outfile = "table-3d-bsim4n.txt"

save i(vss)
echo * 3D table for nmos bsim 4 > $outfile

let xcount = floor((vdstop-vdstart)/vdstep) + 1
let ycount = floor((vgstop-vgstart)/vgstep) + 1
let zcount = floor((vbstop-vbstart)/vbstep) + 1
echo *x >> $outfile
echo $&xcount >> $outfile
echo *y >> $outfile
echo $&ycount >> $outfile
echo *z >> $outfile
echo $&zcount >> $outfile

let xvec = vector(xcount)
let yvec = vector(ycount)
let zvec = vector(zcount)

let loopx = vdstart
let lcx=0
while lcx < xcount
  let xvec[lcx] = loopx
  let loopx = loopx + vdstep
  let lcx = lcx + 1
end
echo *x row >> $outfile
echo $&xvec >> $outfile

let lcy=0
let loopy = vgstart
while lcy < ycount
  let yvec[lcy] = loopy
  let loopy = loopy + vgstep
  let lcy = lcy + 1
end
echo *y column >> $outfile
echo $&yvec >> $outfile

let lcz=0
let loopz = vbstart
while lcz < zcount
  let zvec[lcz] = loopz
  let loopz = loopz + vbstep
  let lcz = lcz + 1
end
echo *z tables >> $outfile
echo $&zvec >> $outfile

let lcz=0
let loopz = vbstart
while lcz < zcount
  echo *table $&loopz >> $outfile
  alter vbs loopz
  let lcy=0
  let loopy = vgstart
  while lcy < ycount
    alter vgs loopy
    dc vds $&vdstart $&vdstop $&vdstep
    let xvec = i(vss)
    echo $&xvec >> $outfile
    destroy dc1
    let loopy = loopy + vgstep
    let lcy = lcy + 1
  end
  let loopz = loopz + vbstep
  let lcz = lcz + 1
end
.endc


.include ./modelcards/modelcard.pmos
.include ./modelcards/modelcard.nmos
.end
