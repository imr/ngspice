* SEE generator model
.subckt seegen4 n1 n2 n3 n4
.param tochar = 2e-13
.param tfall = 500p trise=50p
.param Inull = 'tochar/(tfall-trise)'
* Eponential current source without control input
* only NMOS nodes with reference GND (substrate).
aseegen1 NULL mon [%i(n1) %i(n2) %i(n3) %i(n4)] seemod1
.model seemod1 seegen (tdelay = 0.62m tperiod=0.01m inull='Inull' perlim=FALSE)
.ends