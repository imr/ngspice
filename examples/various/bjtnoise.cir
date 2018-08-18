BJT Noise Test

vcc 4 0 50
vin 1 0 ac 1

ccouple 1 2 1

ibias 0 2 100uA

rload 4 3 1k

q1 3 2 0 0 test

.model test npn kf=1e-20 af=1 bf=100 rb=10 cjc=4e-12
.noise v(3) vin dec 10 1k 100Meg 1

.control
run
setplot
setplot noise1
plot ally
plot inoise_spectrum onoise_spectrum
setplot noise2
print all
echo
print inoise_total onoise_total
.endc

.end
