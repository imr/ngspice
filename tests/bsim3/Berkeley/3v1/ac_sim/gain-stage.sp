*A Simple MOSFET Gain Stage. AC Analysis using BSIM3v3.1.

M1 3 2 0 0 nmos w=4u l=1u 
Rsource 1 2 100k
Rload 3 vdd 25k

Vdd vdd 0 5 
Vin 1 0 1.44 ac .1

.ac dec 10 100 1000Meg 
.print ac vdb(3)

.model nmos nmos level=49

.end

