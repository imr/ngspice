A paralled resistor array

Vin 1 0 DC 1 SIN (0 1 100MEG 1NS 0.0) AC 1

VR1 1 2 DC 0
R1  2 0 10K

VR2 1 3 DC 0
R2  3 0 10K ac=5K

VR3 1 4 DC 0
R3  4 0 rmodel1 L=11u W=2u ac=2.5k

VR4 1 5 DC 0
R4  5 0 10K ac=5k m=2
.model rmodel1 R  RSH = 1000 NARROW = 1u

VR5 1 6 DC 0
R5  6 0 10 ac=5 scale=1K

.options noacct
.OP
.TRAN 1ns 10ns
.AC DEC 100 1MEG 100MEG
.PRINT TRAN I(VR1), I(VR2), I(VR3), I(VR4),I(VR5)
.PRINT AC I(VR1), I(VR2), I(VR3), I(VR4),I(VR5)

.END
