HICUM2v2.40 Noise Voltage Test

*      _______
*    _|_     _|_
*    ///    / _ \
*           \/ \/ I1
*           /\_/\
*           \___/
*      _______|
*     |      _|
*     |____|'
*     B    |`->
*            _|_ E
*           /VIN\
*           \___/
*            _|_
*            ///
*
*

vin                 E  0  DC    0.0 ac 1.0u
I1                  0   B 25mA
q1                  B  B E    hicumL2V2p40


.include model-card-examples.lib

.control
setplot             new 

let               V1u    =     0*vector(81)

op
noise             v(B) vin dec 10 1 100Meg 1
destroy
*let               unknown1.V1u = sqrt(v(onoise_spectrum))
let               unknown1.V1u = v(onoise_spectrum)

set pensize =         2
plot unknown1.V1u vs frequency loglog title HICUM_NoiseVoltage

echo              "    ... done."
.endcontrol
.end
