* SEE generator model
.subckt seegen4 n1 n2 n3 n4
.param tochar = 1e-13
.param talpha = 500p tbeta=20p
.param Inull = 'tochar/(talpha-tbeta)'
* Eponential current source without control input
aseegen1 NULL [%i(n1) %i(n2) %i(n3) %i(n4)] seemod1
.model seemod1 seegen (tdelay = 0.62m tperiod=0.1m inull='Inull' perlim=FALSE)
.ends