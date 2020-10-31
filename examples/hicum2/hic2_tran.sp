HICUM2v2.40 Amplifier in Time Domain

vcc 2 0 2.5
*vin 1 0 ac 1 dc 0 sin 0 25m 1G
vin 1 0 ac 1 dc 0 pulse 0 50m 0 5p 5p 4n 8n
rs 1 in 50
c1 in b 1n
r1 2 c 180
r2 c b 5k
q1 c b 0 0 hicuml2v2p40
c2 c out 100p
r3 out 0 1k

.ic v(c)=0.9
.control
option method=gear
op
print all
ac dec 10 1Meg 9.9g
plot vdb(out)
tran 2p 200n
plot v(in) v(out)
.endc

.include model-card-examples.lib

.end
