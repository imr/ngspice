
** Stability analysis using the Tian method
** Thanks Frank Wiedmann!

*ng_script
.control
source lv_opamp_stb_tb.net
*show all

	let runs=2
	let run=0



			alter @Vprobe1[acmag]=1
			alter @iprobe1[acmag]=0
	
dowhile run < runs
		set run ="$&run"
			set temp = 27

save i(vprobe1) i(vprobe2) v(probe)

		ac dec 20 0.1 1G
			alter @Vprobe1[acmag]=0
			alter @iprobe1[acmag]=1

	let run = run + 1


end


let ip11 = ac1.i(vprobe1)
let ip12 = ac1.i(vprobe2)
let ip21 = ac2.i(vprobe1)
let ip22 = ac2.i(vprobe2)
let vprb1 = ac1.probe
let vprb2 = ac2.probe


let mb = 1/(vprb1+ip22)-1

let av = 1/(1/(2*(ip11*vprb2-vprb1*ip21)+vprb1+ip21)-1)


let phase 		= 180/PI*vp(av)
let pm_meas 		= phase+180
let phase_mb 		= 180/PI*vp(mb)
let opamp_gain 	= vdb(av)+6
let mgain = -1*opamp_gain
*plot vdb(av) vdb(mb)
*plot phase
*plot phase phase_mb
*plot vdb(mb) phase_mb
*plot phase_mb
*plot vdb(mb) vdb(av)
plot vdb(av) phase


echo "----"
echo "----"
			meas ac open_loop_gain find opamp_gain at=1
			meas ac loop_gain find vdb(av) at =1
			meas ac gain_margin find mgain when phase=10
			meas ac phase_margin_6db find pm_meas when vdb(av)=0
			meas ac phase_margin_ug find pm_meas when vdb(av)=-6
			meas ac loop_gain_3db_f when phase=-45
			meas ac zero_db_f when vdb(av)=0
			meas ac pole_2 when phase = -135
			meas ac gbwp when opamp_gain= 0

echo "----"
echo "----"




*echo "plots are: $plots"
*display all


*quit

