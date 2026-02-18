psp103 CMOS NAND gate ring oscillator after HCI stress

* In .spiceinit: set ngbehavior=hsdea; de denotes: we want degradation simulation

* Upon loading the netlist: load the .agemodel data, add
* the degradation monitors, store the netlist internally

* First tran run: measure the degradation, store deg data for each instance.
* The degradation may be enhanced by raising the supply voltage.

* Second tran run, initiated by command 'plainsim': reload netlist from internal
* storage, remove degradation monitors, quasi a standard run without degradation

* Third tran run, initiated by command 'degsim': reload netlist from internal,
* storage, remove degradation monitors, add degradation data to device instances,
* simulate degraded circuit.


* stress temperature
.param CurTemp=21

* some intermediate parameters
.temp 'CurTemp'

* IHP Open Source PDK
.lib "$PDK_ROOT/$PDK/libs.tech/ngspice/models/cornerMOSlv.lib" mos_tt

.include "aging_par_ng.scs"

* Library name: sg13g2_stdcell
* Cell name: sg13g2_nand2_1
* View name: schematic
* Inherited view list: spectre cmos_sch cmos.sch schematic veriloga ahdl
* pspice dspf
.subckt sg13g2_nand2_1 A B VDD VSS Y
XP1 Y B VDD VDD sg13_lv_pmos w=1.12e-06 l=130.00n ng=1 ad=1p as=1p pd=1u ps=1u m=1
XP0 Y A VDD VDD sg13_lv_pmos w=1.12e-06 l=130.00n ng=1 ad=1p as=1p pd=1u ps=1u m=1
XN1 net1 B VSS VSS sg13_lv_nmos w=740.00n l=130.00n ng=1 ad=1p as=1p pd=1u ps=1u m=1
XN0 Y A net1 VSS sg13_lv_nmos w=740.00n l=130.00n ng=1 ad=1p as=1p pd=1u ps=1u m=1
.ends
* End of subcircuit definition.

* sg13g2_nand2_1 A B VDD VSS Y
Xu1 out5u out5u VDD VSSu out1u sg13g2_nand2_1
Xu2 out1u out1u VDD VSSu out2u sg13g2_nand2_1
Xu3 out2u out2u VDD VSSu out3u sg13g2_nand2_1
Xu4 out3u out3u VDD VSSu out4u sg13g2_nand2_1
Xu5 out4u out4u VDD VSSu out5u sg13g2_nand2_1

Vmeas2 VSSu 0 0

Vsupp VDD 0 1.2

.control
pre_osdi ../lib/ngspice/psp103_nqs.osdi
pre_osdi ../lib/ngspice/psp103.osdi

save out5u i(vmeas) i(vmeas2)

* create and measure degradation
* use higher stress voltage, 1.2V results in negligible degradation.
alter Vsupp = 1.8
tran 10p 200n
rusage

* simulate without degradation
plainsim
alter Vsupp = 1.2
tran 10p 200n
rusage

simulate with degradation
degsim
alter Vsupp = 1.2
tran 10p 200n
rusage


* output characteristics
*set color0=white
set xbrushwidth=2
let out5u_prev =  tran2.out5u
let out5u_prev_prev =  tran1.out5u
plot out5u_prev_prev out5u_prev out5u xlimit 86n 90n

linearize out5u_prev_prev out5u_prev out5u
fft out5u_prev_prev out5u_prev out5u
plot mag(out5u_prev_prev) mag(out5u_prev) mag(out5u) xlimit 1.5G 3.5G
plot mag(out5u_prev_prev) mag(out5u_prev) mag(out5u) xlimit 1.5G 2G ylimit 0 500m

.endc
.end
