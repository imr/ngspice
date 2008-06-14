* PIN model
* line 2
* line 3
* -- Summary -------------------------------
* This is a simple spice model of a PIN diode.
*
* -- Description ---------------------------
* It is a three node device; one input node (relative to ground) and two
* output nodes (cathode and anode)
*

* -- Model ----------------------------------
.subckt	SIMPLE_PIN input cathode anode resp=0.5

* Input photocurrent is modled by a voltage
* This generates a current using a linear voltage-controlled current source
Gin    dk  da  input  0  {resp}
Rin	 input  0   1G
Cin  input  0   {resp}

* The pn-junction that generates this photocurrent in the real device is modelled 
* here by a simple diode
Dpn da dk  pndiode

* terminal resistances
Ra	anode   da  0.001ohm
Rk  cathode dk  0.001ohm

* subsircuit models:
.MODEL pndiode D IS=0.974p RS=0.1 N=1.986196 BV=7.1 IBV=0.1n 
+ CJO=99.2p VJ=0.455536 M=0.418717 TT=500n

.ends
