* filesource Test

* two differential ports 1 0 and 3 0 are used, so your input file
* has to have three columns (time, port_value 1, portvalue 2)

AFILESRC %vd([1 0 3 0]) filesrc
.model filesrc filesource (file="sine.m" amploffset=[0 0] amplscale=[1 1] timerelative=false amplstep=false)

V2 2 0 0.0 SIN(0 1 1MEG 0 0 0.0)
V4 4 0 0.0 SIN(0 1 1MEG 0 0 90.0)

.tran 1n 1.0u

.control
run
*listing param
wrdata vspice V(1) V(2) V(3) V(4)


plot V(1) V(2) V(3) V(4)

* error between interpolation and sine source
* should be less than 1mV up to 1us
plot V(1,2) V(3,4)
.endc
.end
