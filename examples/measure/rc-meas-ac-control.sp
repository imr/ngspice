RC band pass example circuit
* This netlist demonstrates the following:
* global nodes (vdd, gnd)

* .measure statements for delay and an example ternary operator

* You can run the example circuit with this command:
*
* ngspice rc-meas-ac.sp


* global nodes
.global vdd gnd

* autostop -- stop simulation early if .measure statements done
*.option autostop

vin in gnd  dc 0 ac 1

R1 in mid1 1k
c1 mid1 gnd 1n
C2 mid1 out 500p
R2 out gnd 1k


.control
ac DEC 10 1k 10MEG
meas ac vout_at FIND v(out) AT=1MEG
meas ac vout_atr FIND vr(out) AT=1MEG
meas ac vout_ati FIND vi(out) AT=1MEG
meas ac vout_atm FIND vm(out) AT=1MEG
meas ac vout_atp FIND vp(out) AT=1MEG
meas ac vout_atd FIND vdb(out) AT=1MEG
meas ac vout_max max v(out) from=1k to=10MEG
meas ac freq_at when v(out)=0.1
meas ac vout_diff trig v(out)  val=0.1   rise=1 targ v(out) val=0.1   fall=1
meas ac fixed_diff trig AT = 10k targ v(out) val=0.1   rise=1
meas ac vout_avg  avg   v(out)  from=10k to=1MEG
meas ac vout_integ integ v(out) from=20k to=500k
meas ac freq_at2 when v(out)=0.1 fall=LAST
*meas ac bw_chk param='(vout_diff < 100k) ? 1 : 0'
if (vout_diff < 100k)
   let bw_chk = 1
else
   let bw_chk = 0
end  
echo bw_chk = "$&bw_chk" 
*meas ac bw_chk2 param='(vout_diff > 500k) ? 1 : 0'
if (vout_diff > 500k)
   let bw_chk2 = 1
else
   let bw_chk2 = 0
end 
echo bw_chk2 = "$&bw_chk2" 
meas ac vout_rms rms v(out) from=10 to=1G
*rusage all
plot v(out)
plot ph(v(out))
plot mag(v(out))
plot db(v(out))
.endc

.end

