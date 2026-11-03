* Simple test for xfer code model: comparison
*
* This circuit compares the results of an AC analysis of a filter (node out)
* with those from a behavioural model controlled by measured S-parameters
* of that circuit (node xout).  The AC analysis has more data points than
* that used to measure the S-parameters, to prevent the results from matching
* exactly.

* The use of S-parameters to create a behavioural simulation of a component
* was discussed here:
* https://sourceforge.net/p/ngspice/discussion/120973/thread/51228e0b01/

* Circuit from:
* Novarianti, Dini. (2019).
* Design and Implementation of Chebyshev Band Pass Filter with
* M-Derived Section in Frequency Band 88 - 108 MHz.
* Jurnal Jartel: Jurnal Jaringan Telekomunikasi. 8. 7-11.
* 10.33795/jartel.v8i1.147. 
*
* https://www.researchgate.net/publication/352822864_Design_and_Implementation_of_Chebyshev_Band_Pass_Filter_with_M-Derived_Section_in_Frequency_Band_88_-_108_MHz

* Set this parameter to 1 to generate a Touchstone file that can be used
* to generate the behavioural part of the circuit, filter.lib

.param do_sp=0
.csparam do_sp=do_sp

.if (do_sp)

.csparam Rbase=50 ; This is required by "wrs2p", below.
vgen 1 0 dc 0 ac 1 portnum 1

.else

vgen in 0 dc 0 ac 1
rs in 1 50

.endif

l1 1 2 0.058u
c2 2 0 40.84p
l3 2 3 0.128u
c4 3 0 47.91p
l5 3 4 0.128u
c6 4 0 40.48p
l7 4 5 0.058u

la 5 6 0.044u
lb 6 a 0.078u
cb a 0 17.61p
lc 6 b 0.151u
cc b 0 34.12p
c7 6 7 26.035p

l8 7 0 0.0653u
c8 7 8 20.8p
l9 8 0 0.055u
c9 8 9 20.8p
l10 9 0 0.653u

c10 9 out 45.64p

.if (do_sp)

vl out 0 dc 0 ac 0 portnum 2

.else

rl out 0 50

* Behavioural circuit, for comparison.

.inc filter.lib
R1 in port1 50
xsp port1 xout 0 filter
R2 xout 0 50

.endif

.control
if $&do_sp
   sp lin 40 10meg 200meg
   wrs2p filter.s2p
   plot S_1_1 S_2_2 polar
else
   ac lin 400 10meg 200meg
   plot db(mag(out)) 5*unwrap(ph(out)) db(mag(xout)) 5*unwrap(ph(xout))
end

.endc
.end
