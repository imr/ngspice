OpAmp Test

vddp vp 0 3
vddn vn 0 -3

Xopmap 0 ino outo vp vn CLC409
Rin in ino 1k
Rfb ino outo 3k


vin in 0 PULSE(-1 1 20NS 20NS 20NS 500NS 1000NS)

* circuit with standard bip model
*.include clc409.sub
* circuit with qinn bipolar table model
.include clc409mod.sub

.options savecurrents

.control
dc vin -1 1 0.1
plot v(outo)
tran 10n 1000n
plot v(outo)
.endc


.end

