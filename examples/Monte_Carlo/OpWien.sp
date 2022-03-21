OPWIEN.CIR - OPAMP WIEN-BRIDGE OSCILLATOR
* http://www.ecircuitcenter.com/Circuits/opwien/OPWIEN.CIR
* single simulation run
* 2 resistors and 2 capacitors of Wien bridge a varied statistically
* number of variations: varia

* Simulation time
.param ttime=12000m
.param varia=100
.param ttime10 = 'ttime/varia'

* nominal resistor and capacitor values
.param res = 10k
.param cn = 16NF

* CURRENT PULSE TO START OSCILLATIONS
IS	0	3	dc 0 PWL(0US 0MA   10US 0.1MA   40US 0.1MA   50US 0MA   10MS 0MA)
*
* RC TUNING
VR2 r2  0 dc 0 trrandom (2 'ttime10' 0 1)  ; Gauss controlling voltage
* 
*VR2 r2  0 dc 0 trrandom (1 'ttime10' 0 3) ; Uniform within -3 3
*
* If Gauss, factor 0.033 is 10% equivalent to 3 sigma
* if uniform, uniform between +/- 10%
R2 4 6  R = 'res + 0.033 * res*V(r2)'  ; behavioral resistor
*R2 4 6 'res' ; constant R

VC2 c2 0 dc 0  trrandom (2 'ttime10' 0 1)
*C2	6 	3'cn'  ; constant C
C2 6 3 C = 'cn + 0.033 * cn*V(c2)'  ; behavioral capacitor

VR1 r1  0 dc 0 trrandom (2 'ttime10' 0 1)
*VR1 r1  0 dc 0 trrandom (1 'ttime10' 0 3)
R1	3	0	R = 'res + 0.033 * res*V(r1)'
*R1	3 	0	'res'

VC1 c1 0 dc 0  trrandom (2 'ttime10' 0 1)
C1 3 0 C =  'cn + 0.033 * cn*V(c2)'
*C1	3 	0	'cn'

* NON-INVERTING OPAMP
R10	0	2	10K
R11	2	5	18K
XOP	3 2	4	OPAMP1
* AMPLITUDE STABILIZATION
R12	5	4	5K
D1	5	4	D1N914
D2	4	5	D1N914
*
.model	D1N914	D(Is=0.1p Rs=16 CJO=2p Tt=12n Bv=100 Ibv=0.4n)
*
* OPAMP MACRO MODEL, SINGLE-POLE 
* connections:      non-inverting input
*                   |   inverting input
*                   |   |   output
*                   |   |   |
.SUBCKT OPAMP1      1   2   6
* INPUT IMPEDANCE
RIN	1	2	10MEG
* DC GAIN (100K) AND POLE 1 (100HZ)
EGAIN	3 0	1 2	100K
RP1	3	4	1K
CP1	4	0	1.5915UF
* OUTPUT BUFFER AND RESISTANCE
EBUFFER	5 0	4 0	1
ROUT	5	6	10
.ENDS
*
* ANALYSIS
.TRAN 	0.05MS 'ttime'
*
* VIEW RESULTS
.control
option noinit
run
plot 5*V(r1) 5*V(r2) 5*V(c1) 5*V(c2) V(4)
linearize v(4)
fft v(4)
let v4mag =  mag(v(4))
plot v4mag
plot v4mag xlimit 500 1500
*wrdata histo v4mag
rusage
.endc

.END
