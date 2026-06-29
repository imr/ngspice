#!/bin/bash

# Copyright 2023 The ngspice team
# Authors: Jim Monte, Holger Vogt, Dietmar Warning
# License: New BSD

if test "$1" = "?"; then
    echo "ngspice \"paranoia\" test suite"
    echo "Format: $0 [ngspice executable]"
    echo "The ngspice executable must be independent of the current directory."
    exit 1
fi

SECONDS=0

NGSPICE="$1"
NGSPICE="${NGSPICE:-ngspice}"
VALGRIND="valgrind --leak-check=full --suppressions=$(pwd)/ignore_shared_libs.supp"

## The following three take much time. They are started in the background and may be skipped after being run once.
#cd examples/xspice/table
#$VALGRIND --log-file=../../../table-generator-q-2d.vlog  $NGSPICE -o ../../../table-generator-q-2d.log  table-generator-q-2d.sp &
#$VALGRIND --log-file=../../../table-generator-b4n-2d.vlog $NGSPICE -o ../../../table-generator-b4n-2d.log table-generator-b4n-2d.sp &
#$VALGRIND --log-file=../../../table-generator-b4p-2d.vlog $NGSPICE -o ../../../table-generator-b4p-2d.log table-generator-b4p-2d.sp &
#$VALGRIND --log-file=../../../table-generator-b4n-3d.vlog $NGSPICE -o ../../../table-generator-b4n-3d.log table-generator-b4n-3d.sp &
#$VALGRIND --log-file=../../../table-generator-b4p-3d.vlog $NGSPICE -o ../../../table-generator-b4p-3d.log table-generator-b4p-3d.sp &
#cd ../../..

cd examples/control_structs
$VALGRIND --log-file=../../s-param.vlog $NGSPICE -o ../../s-param.log s-param.cir
$VALGRIND --log-file=../../repeat3.vlog $NGSPICE -o ../../repeat3.log repeat3.sp &
$VALGRIND --log-file=../../foreach_bjt_ft.vlog $NGSPICE -o ../../foreach_bjt_ft.log foreach_bjt_ft.sp
cd ../..

cd examples/delta-sigma
$VALGRIND --log-file=../../mod1-ct-test.vlog $NGSPICE -o ../../mod1-ct-test.log mod1-ct-test.cir
cd ../..

cd examples/digital
$VALGRIND --log-file=../../adder_behav.vlog $NGSPICE -o ../../adder_behav.log adder_behav.cir
$VALGRIND --log-file=../../adder_bip.vlog $NGSPICE -o ../../adder_bip.log adder_bip.cir
$VALGRIND --log-file=../../adder_mos.vlog $NGSPICE -o ../../adder_mos.log adder_mos.cir
$VALGRIND --log-file=../../adder_cshunt.vlog $NGSPICE -o ../../adder_cshunt.log adder_cshunt.cir
$VALGRIND --log-file=../../adder_Xspice.vlog $NGSPICE -o ../../adder_Xspice.log adder_Xspice.cir
cd ../..

cd examples/inductive-systems
$VALGRIND --log-file=../../positive-definite-1.vlog $NGSPICE -o ../../positive-definite-1.log positive-definite-1.cir &
$VALGRIND --log-file=../../positive-definite-2.vlog $NGSPICE -o ../../positive-definite-2.log positive-definite-2.cir &
$VALGRIND --log-file=../../positive-definite-3.vlog $NGSPICE -o ../../positive-definite-3.log positive-definite-3.cir &
$VALGRIND --log-file=../../positive-definite-4.vlog $NGSPICE -o ../../positive-definite-4.log positive-definite-4.cir &
cd ../..

cd examples/measure
$VALGRIND --log-file=../../func_cap.vlog $NGSPICE -o ../../func_cap.log func_cap.sp &
$VALGRIND --log-file=../../mos-meas-dc.vlog $NGSPICE -o ../../mos-meas-dc.log mos-meas-dc.sp
$VALGRIND --log-file=../../inv-meas-tran.vlog $NGSPICE -o ../../inv-meas-tran.log inv-meas-tran.sp
$VALGRIND --log-file=../../simple-meas-tran.vlog $NGSPICE -o ../../simple-meas-tran.log simple-meas-tran.sp
$VALGRIND --log-file=../../rc-meas-ac.vlog $NGSPICE -o ../../rc-meas-ac.log rc-meas-ac.sp
cd ../..

cd examples/memristor
$VALGRIND --log-file=../../memristor.vlog $NGSPICE -o ../../memristor.log memristor.sp
$VALGRIND --log-file=../../memristor_x.vlog $NGSPICE -o ../../memristor_x.log memristor_x.sp
cd ../..

cd examples/Monte_Carlo
$VALGRIND --log-file=../../MC_2_control.vlog $NGSPICE -o ../../MC_2_control.log MC_2_control.sp
$VALGRIND --log-file=../../MC_ring.vlog $NGSPICE -o ../../MC_ring.log MC_ring.sp
$VALGRIND --log-file=../../OpWien.vlog $NGSPICE -o ../../OpWien.log OpWien.sp
$VALGRIND --log-file=../../MC_ring_ts.vlog $NGSPICE -o ../../MC_ring_ts.log MC_ring_ts.sp
cd ../..

cd examples/pll
$VALGRIND --log-file=../../pll-xspice.vlog $NGSPICE -o ../../pll-xspice.log pll-xspice.cir
cd ../..

cd examples/pton
$VALGRIND --log-file=../../555-timer-2.vlog $NGSPICE -o ../../555-timer-2.log 555-timer-2.cir
$VALGRIND --log-file=../../MOS1_out.vlog $NGSPICE -o ../../MOS1_out.log MOS1_out.cir
$VALGRIND --log-file=../../op-test-adi.vlog $NGSPICE -o ../../op-test-adi.log op-test-adi.cir
$VALGRIND --log-file=../../op-test.vlog $NGSPICE -o ../../op-test.log op-test.cir
$VALGRIND --log-file=../../Optimos_out.vlog $NGSPICE -o ../../Optimos_out.log Optimos_out.cir
$VALGRIND --log-file=../../OP_MCP6041.vlog $NGSPICE -o ../../OP_MCP6041.log OP_MCP6041.cir
$VALGRIND --log-file=../../relax_osc_st.vlog $NGSPICE -o ../../relax_osc_st.log relax_osc_st.cir
$VALGRIND --log-file=../../remcirc-test.vlog $NGSPICE -o ../../remcirc-test.log remcirc-test.cir
$VALGRIND --log-file=../../switch-oscillators.vlog $NGSPICE -o ../../switch-oscillators.log switch-oscillators.cir
$VALGRIND --log-file=../../switch-oscillators_inc.vlog $NGSPICE -o ../../switch-oscillators_inc.log switch-oscillators_inc.cir
cd ../..

cd examples/transient-noise
$VALGRIND --log-file=../../noi-ring51-demo.vlog $NGSPICE -o ../../noi-ring51-demo.log noi-ring51-demo.cir
$VALGRIND --log-file=../../noi-sc-tr.vlog $NGSPICE -o ../../noi-sc-tr.log noi-sc-tr.cir
$VALGRIND --log-file=../../rts-1.vlog $NGSPICE -o ../../rts-1.log rts-1.cir
$VALGRIND --log-file=../../noise_vnoi.vlog $NGSPICE -o ../../noise_vnoi.log noise_vnoi.cir
$VALGRIND --log-file=../../shot_ng.vlog $NGSPICE -o ../../shot_ng.log shot_ng.cir
cd ../..

cd examples/optran
$VALGRIND --log-file=../../TLV9002-test.vlog $NGSPICE -o ../../TLV9002-test.log TLV9002-test.cir
$VALGRIND --log-file=../../TLV6001-test.vlog $NGSPICE -o ../../TLV6001-test.log TLV6001-test.cir
$VALGRIND --log-file=../../F5TurboV2thermal-optran.vlog $NGSPICE -o ../../F5TurboV2thermal-optran.log F5TurboV2thermal-optran.cir
cd ../..

cd examples/TransImpedanceAmp
$VALGRIND --log-file=../../output.vlog $NGSPICE -o ../../output.log output.net
cd ../..

cd examples/various
$VALGRIND --log-file=../../FFT_Leakage.vlog $NGSPICE -o ../../FFT_Leakage.log FFT_Leakage.cir
$VALGRIND --log-file=../../ro_17_4.vlog $NGSPICE -o ../../ro_17_4.log ro_17_4.cir
$VALGRIND --log-file=../../3d_loop.vlog $NGSPICE -o ../../3d_loop.log 3d_loop.sp
$VALGRIND --log-file=../../FFT_tests.vlog $NGSPICE -o ../../FFT_tests.log FFT_tests.cir
$VALGRIND --log-file=../../agauss_test.vlog $NGSPICE -o ../../agauss_test.log agauss_test.cir
$VALGRIND --log-file=../../bjtnoise.vlog $NGSPICE -o ../../bjtnoise.log bjtnoise.cir
$VALGRIND --log-file=../../probe-i-dev.vlog $NGSPICE -o ../../probe-i-dev.log probe-i-dev.cir
#$VALGRIND --log-file=../../Tschebyschef-LP.cir.vlog $NGSPICE -o ../../Tschebyschef-LP.cir.log Tschebyschef-LP.cir
cd ../..

cd examples/vbic
$VALGRIND --log-file=../../DFF_Y_ECL.vlog $NGSPICE -o ../../DFF_Y_ECL.log DFF_Y_ECL.sp
$VALGRIND --log-file=../../npn_ft.vlog $NGSPICE -o ../../npn_ft.log npn_ft.sp
$VALGRIND --log-file=../../npn_gum.vlog $NGSPICE -o ../../npn_gum.log npn_gum.sp
$VALGRIND --log-file=../../npn_out.vlog $NGSPICE -o ../../npn_out.log npn_out.sp
$VALGRIND --log-file=../../self-heat.vlog $NGSPICE -o ../../self-heat.log self-heat.sp
$VALGRIND --log-file=../../vbic99_dc.vlog $NGSPICE -o ../../vbic99_dc.log vbic99_dc.sp
$VALGRIND --log-file=../../vbic99_tran.vlog $NGSPICE -o ../../vbic99_tran.log vbic99_tran.sp
$VALGRIND --log-file=../../vbic_ac_par.vlog $NGSPICE -o ../../vbic_ac_par.log vbic_ac_par.sp
cd ../..

cd examples/vdmos
$VALGRIND --log-file=../../100W.vlog $NGSPICE -o ../../100W.log 100W.sp
$VALGRIND --log-file=../../100W_wingspread.vlog $NGSPICE -o ../../100W_wingspread.log 100W_wingspread.sp
$VALGRIND --log-file=../../crss_coss_ciss.vlog $NGSPICE -o ../../crss_coss_ciss.log crss_coss_ciss.sp
$VALGRIND --log-file=../../dcdc.vlog $NGSPICE -o ../../dcdc.log dcdc.sp
$VALGRIND --log-file=../../inv_vdmos.vlog $NGSPICE -o ../../inv_vdmos.log inv_vdmos.sp
$VALGRIND --log-file=../../IXTH80N20L-IXTH48P20P-quasisat.vlog $NGSPICE -o ../../IXTH80N20L-IXTH48P20P-quasisat.log IXTH80N20L-IXTH48P20P-quasisat.sp
$VALGRIND --log-file=../../IXTP6N100D2-cap.vlog $NGSPICE -o ../../IXTP6N100D2-cap.log IXTP6N100D2-cap.sp
$VALGRIND --log-file=../../IXTP6N100D2-n-weak-inv.vlog $NGSPICE -o ../../IXTP6N100D2-n-weak-inv.log IXTP6N100D2-n-weak-inv.sp
$VALGRIND --log-file=../../mtriode_nch.vlog $NGSPICE -o ../../mtriode_nch.log mtriode_nch.sp
$VALGRIND --log-file=../../ro_11_vdmos.vlog $NGSPICE -o ../../ro_11_vdmos.log ro_11_vdmos.sp
$VALGRIND --log-file=../../self-heating.vlog $NGSPICE -o ../../self-heating.log self-heating.sp
$VALGRIND --log-file=../../soa_chk.vlog $NGSPICE -o ../../soa_chk.log soa_chk.sp
$VALGRIND --log-file=../../theta_nch.vlog $NGSPICE -o ../../theta_nch.log theta_nch.sp
$VALGRIND --log-file=../../VDMOS-DIO.vlog $NGSPICE -o ../../VDMOS-DIO.log VDMOS-DIO.sp
$VALGRIND --log-file=../../vdmos-out.vlog $NGSPICE -o ../../vdmos-out.log vdmos-out.sp
$VALGRIND --log-file=../../vdmos-out_ir_mtr.vlog $NGSPICE -o ../../vdmos-out_ir_mtr.log vdmos-out_ir_mtr.sp
$VALGRIND --log-file=../../vdmosp-out.vlog $NGSPICE -o ../../vdmosp-out.log vdmosp-out.sp
$VALGRIND --log-file=../../vdmosp-out-mtr.vlog $NGSPICE -o ../../vdmosp-out-mtr.log vdmosp-out-mtr.sp
cd ../..

cd examples/xspice
$VALGRIND --log-file=../../xspice_c1.vlog $NGSPICE -o ../../xspice_c1.log xspice_c1.cir
$VALGRIND --log-file=../../xspice_c2.vlog $NGSPICE -o ../../xspice_c2.log xspice_c2.cir
$VALGRIND --log-file=../../xspice_c3.vlog $NGSPICE -o ../../xspice_c3.log xspice_c3.cir
$VALGRIND --log-file=../../bug-378-test-sequence.vlog $NGSPICE -o ../../bug-378-test-sequence.log bug-378-test-sequence.cir
cd ../..

cd examples/xspice/d_source
$VALGRIND --log-file=../../../d_source.vlog $NGSPICE -o ../../../d_source.log d_source.cir &
cd ../../..
cd examples/xspice/filesource
$VALGRIND --log-file=../../../simple-filesource.vlog $NGSPICE -o ../../../simple-filesource.log simple-filesource.cir
cd ../../..
cd examples/xspice/state
$VALGRIND --log-file=../../../state-machine.vlog $NGSPICE -o ../../../state-machine.log state-machine.cir
cd ../../..

# waiting for the table generator files
#wait
cd examples/xspice/table
$VALGRIND --log-file=../../../table-model-bip-2d-1.vlog $NGSPICE -o ../../../table-model-bip-2d-1.log table-model-bip-2d-1.sp
$VALGRIND --log-file=../../../table-model-bip-2d-2.vlog $NGSPICE -o ../../../table-model-bip-2d-2.log table-model-bip-2d-2.sp
$VALGRIND --log-file=../../../table-model-mos-2d-2.vlog $NGSPICE -o ../../../table-model-mos-2d-2.log table-model-mos-2d-2.sp
$VALGRIND --log-file=../../../table-model-mos-2d-3.vlog $NGSPICE -o ../../../table-model-mos-2d-3.log table-model-mos-2d-3.sp
$VALGRIND --log-file=../../../table-model-mos-2d-4.vlog $NGSPICE -o ../../../table-model-mos-2d-4.log table-model-mos-2d-4.sp
$VALGRIND --log-file=../../../table-model-mos-3d-2.vlog $NGSPICE -o ../../../table-model-mos-3d-2.log table-model-mos-3d-2.sp
$VALGRIND --log-file=../../../table-model-mos-3d-3.vlog $NGSPICE -o ../../../table-model-mos-3d-3.log table-model-mos-3d-3.sp
$VALGRIND --log-file=../../../table-model-mos-3d-4.vlog $NGSPICE -o ../../../table-model-mos-3d-4.log table-model-mos-3d-4.sp
$VALGRIND --log-file=../../../table-model-mos-3d-5.vlog $NGSPICE -o ../../../table-model-mos-3d-5.log table-model-mos-3d-5.sp
$VALGRIND --log-file=../../../table-model-man-3d-1.vlog $NGSPICE -o ../../../table-model-man-3d-1.log table-model-man-3d-1.sp
$VALGRIND --log-file=../../../table-model-man-2d-1.vlog $NGSPICE -o ../../../table-model-man-2d-1.log table-model-man-2d-1.sp
$VALGRIND --log-file=../../../combi_script.vlog $NGSPICE -o ../../../combi_script.log combi_script.cir
cd ../../..

cd examples/digital_devices
$VALGRIND --log-file=../../dig-behav-568.vlog $NGSPICE -o ../../dig-behav-568.log behav-568.cir
$VALGRIND --log-file=../../dig-counter.vlog $NGSPICE -o ../../dig-counter.log counter.cir
$VALGRIND --log-file=../../dig-ex4.vlog $NGSPICE -o ../../dig-ex4.log ex4.cir
cd ../..

cd examples/cider
$VALGRIND --log-file=../../cider-cmosinv.vlog $NGSPICE -o ../../cider-cmosinv.log cmosinv.cir
$VALGRIND --log-file=../../cider-diode.vlog $NGSPICE -o ../../cider-diode.log diode.cir
$VALGRIND --log-file=../../cider-pullup.vlog $NGSPICE -o ../../cider-pullup.log pullup.cir
$VALGRIND --log-file=../../cider-recovery.vlog $NGSPICE -o ../../cider-recovery.log recovery.cir
$VALGRIND --log-file=../../cider-rtlinv.vlog $NGSPICE -o ../../cider-rtlinv.log rtlinv.cir
cd ../..

# Check the results
# Find correct response: ngspice-<version> done
NGSPICE_OK="`$NGSPICE -v | awk '/level/ {print $2;}'` done"

echo "*******************************************"
echo "vlog files with errors found by valgrind:"
grep -L "ERROR SUMMARY: 0 errors from 0 context" ./*.vlog
echo "*******************************************"
echo "log files with ngspice errors:"
grep -L "$NGSPICE_OK" ./*.log
echo "*******************************************"
echo "log files with convergence issues:"
grep -l "Too many iterations without convergence" ./*.log
echo "*******************************************"
echo "log files with messages containing 'error':"
grep -i -l "error" ./*.log
echo "*******************************************"

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo
echo $ELAPSED

