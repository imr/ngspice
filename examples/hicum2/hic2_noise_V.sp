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
I1                  0   B 1uA
q1                  B  B E    hicumL2V2p40


.include model-card-examples.lib

.control
setplot             new 

let               V1u    =     0*vector(81)
let               V10u   =     0*vector(81)
let               V100u  =     0*vector(81)
let               V1000u =     0*vector(81)

op
noise             v(B) vin dec 10 1 100Meg 1
destroy
let               unknown1.V1u = sqrt(v(onoise_spectrum))

alter             I1      dc  = 10u
op
noise             v(B) vin dec 10 1 100Meg 1
destroy
let               unknown1.V10u = sqrt(v(onoise_spectrum))

alter             I1      dc  = 100u
op
noise             v(B) vin dec 10 1 100Meg 1
destroy
let               unknown1.V100u = sqrt(v(onoise_spectrum))

alter             I1      dc  = 1000u
op
noise             v(B) vin dec 10 1 100Meg 1
destroy
let               unknown1.V1000u = sqrt(v(onoise_spectrum))


set pensize =         2
plot unknown1.V1u unknown1.V10u unknown1.V100u unknown1.V1000u vs frequency loglog title HICUM_NoiseVoltage

echo              "    ... done."
.endcontrol
.end
