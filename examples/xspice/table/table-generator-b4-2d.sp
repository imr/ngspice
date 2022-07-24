** NMOSFET: table generator with BSIM4 2D (Vdrain, Vgate)
* This file may be run by 'ngspice table-generator-b4-2d.sp'
* It will generate a 2D data table by simulating the MOS drain current
* as function of drain and gate voltages. The simulation uses
* the ngspice BSIM4.6.1 MOS model and Berkeley model parameters.
* This table is an input file for the XSPICE 2D table model.
* You have to select NMOS or PMOS by manually editing parameter
* 'mostype'.
* In addition you may change the step sizes vdstep vgstep vbstep in CSPARAM
* to obtain the required resolution for the data.
* These tables will contain pure dc data. For transient simulation you may
* need to add some capacitors to the device model for a 'real world' simulation.

* setting the MOS type (NMOS or PMOS)
.param mostype = 2 ; NMOS, 2 for PMOS

.if (mostype == 1)
*NMOS
.csparam vdstart=-0.1
.csparam vdstop=1.8
.csparam vdstep=0.01
.csparam vgstart=-0.1
.csparam vgstop=1.8
.csparam vgstep=0.01
.csparam mtype=1
.elseif (mostype == 2)
*PMOS
.csparam vdstart=-1.8
.csparam vdstop=0.1
.csparam vdstep=0.01
.csparam vgstart=-1.8
.csparam vgstop=0.1
.csparam vgstep=0.01
.csparam mtype=2
.endif

** Circuit Description **
.if (mostype == 1)
m1 2 1 3 0 nbsim4 L=0.13u W=10.0u rgeoMod=1
.elseif (mostype == 2)
m1 2 1 3 0 pbsim4 L=0.13u W=10.0u rgeoMod=1
.endif
vgs 1 0 1.8
vds 2 0 1.8
vss 3 0 0

.control
** output file **
echo
if mtype = 1
  set outfile = "$inputdir/bsim4n-2d-1.table"
  echo nmos table generation , table is written to
  echo $outfile
else
  if mtype = 2
    set outfile = "$inputdir/bsim4p-2d-1.table"
    echo pmos table generation , table is written to
    echo $outfile
  end
end
echo

save i(vss)
if mtype = 1
  echo *table for nmos bsim 4 > $outfile
else
  echo *table for pmos bsim 4 > $outfile
end

let xcount = floor((vdstop-vdstart)/vdstep) + 1
let ycount = floor((vgstop-vgstart)/vgstep) + 1
echo *x >> $outfile
echo $&xcount >> $outfile
echo *y >> $outfile
echo $&ycount >> $outfile
let xvec = vector(xcount)
let yvec = vector(ycount)
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

let lcy=0
let loopy = vgstart
while lcy < ycount
  alter vgs loopy
  dc vds $&vdstart $&vdstop $&vdstep
*  let lcx=0
*  let loopx = vdstart
*  dowhile loopx le vdstop
*    alter vds loopx
*    op
*    let xvec[lcx] = i(vss)
*       destroy i(vss)
*    let loopx = loopx + vdstep
*    let lcx = lcx + 1
*  end
  let xvec = i(vss)
  echo $&xvec >> $outfile
  destroy dc1
  let loopy = loopy + vgstep
  let lcy = lcy + 1
end

.endc

.if (mostype == 1)
.include ./modelcards/modelcard.nmos
.elseif (mostype == 2)
.include ./modelcards/modelcard.pmos
.endif

.end
