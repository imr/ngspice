// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015-2020, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

//
// Operation point (output) variables
//

// The external currents and current gain
`OPM(ic,        "A", "External DC collector current")
`OPM(ib,        "A", "External DC base current")
`OPP(betadc,     "", "External DC current gain Ic/Ib")

// begin added in MXT 504.9:
`OPM(ie,        "A", "External DC emitter current")

// The external biases
`OPP(vbe,     "V", "External base-emitter bias")
`OPP(vce,     "V", "External collector-emitter bias")
`OPP(vbc,     "V", "External base-collector bias")

`ifdef SUBSTRATE
    `OPM(isx,     "A", "External DC substrate current")
    `OPP(vse,     "V", "External substrate-emitter bias")
    `OPP(vbs,     "V", "External base-substrate bias")
    `OPP(vsc,     "V", "External substrate-collector bias")
`endif

// end added in MXT 504.9
// The internal biases
`OPP(vb2e1,     "V", "Internal base-emitter bias")
`OPP(vb2c2,     "V", "Internal base-collector bias")
`OPP(vb2c1,     "V", "Internal base-collector bias including epilayer")
`OPP(vb1c1,     "V", "External base-collector bias without contact resistances")
`OPP(vc4c1,     "V", "Bias over intrinsic buried layer")
`OPP(vc3c4,     "V", "Bias over extrinsic buried layer")
`OPP(ve1e,      "V", "Bias over emitter resistance")

// The actual currents
`OPM(in,        "A", "Main current")
`OPM(ic1c2,     "A", "Epilayer current")
`OPM(ib1b2,     "A", "Pinched-base current")
`OPM(ib1,       "A", "Ideal forward base current")
`OPM(ib1s,      "A", "Ideal side-wall base current")
`OPM(ib2s,      "A", "Non-ideal side-wall base current")
`OPM(ibrel,     "A", "Additional non-ideal base current for reliability simulation")
//
// 504.8, RvdT, TU-Delft April. 2009, Zener tunneling current:
//
`OPM(izteb,     "A", "Zener tunneling current in the emitter base junction")
//
`OPM(iztcb,     "A", "Zener tunneling current in the collector base junction")
`OPM(ib2,       "A", "Non-ideal forward base current")
`OPM(ib3,       "A", "Non-ideal reverse base current")
`OPM(iavl,      "A", "Avalanche current")
`OPM(iex,       "A", "Extrinsic reverse base current")

`OPM(xiex,      "A", "Extrinsic reverse base current")
`ifdef SUBSTRATE
    `OPM(isub,      "A", "Substrate current")
    `OPM(xisub,     "A", "Substrate current")
    `OPM(isf,       "A", "Substrate failure current")
`endif
`OPM(ire,       "A", "Current through emitter resistance")
`OPM(irbc,      "A", "Current through constant base resistance")
`OPM(ircblx,    "A", "Current through extrinsic buried layer resistance")
`OPM(ircbli,    "A", "Current through intrinsic buried layer resistance")
`OPM(ircc,      "A", "Current through collector contact resistance")

//The actual charges
`OPM(qe,        "coul", "Emitter charge or emitter neutral charge")
`OPM(qte,       "coul", "Base-emitter depletion charge")
`OPM(sqte,      "coul", "Sidewall base-emitter depletion charge")
`OPM(qbe,       "coul", "Base-emitter diffusion charge")
`OPM(qbc,       "coul", "Base_collector diffusion charge")
`OPM(qtc,       "coul", "Base-collector depletion charge")
`OPM(qepi,      "coul", "Epilayer diffusion charge")
`OPM(qb1b2,     "coul", "ac current crowding charge")
`OPM(qtex,      "coul", "Extrinsic base-collector depletion charge")
`OPM(xqtex,     "coul", "Extrinsic base-collector depletion charge")
`OPM(qex,       "coul", "Extrinsic base-collector diffusion charge")
`OPM(xqex,      "coul", "Extrinsic base-collector diffusion charge")
`ifdef SUBSTRATE
    `OPM(qts,       "coul", "Collector-substrate depletion charge")
`endif

//Small signal equivalent circuit conductances and resistances
`OPM(gx,        "S", "Forward transconductance")
`OPM(gy,        "S", "Reverse transconductance")
`OPM(gz,        "S", "Reverse transconductance")
`OPM(sgpi,      "S", "Conductance sidewall b-e junction")
`OPM(gpix,      "S", "Conductance floor b-e junction")
`OPM(gpiy,      "S", "Early effect on recombination base current")
`OPM(gpiz,      "S", "Early effect on recombination base current")
`OPM(gmux,      "S", "Early effect on avalanche current limiting")
`OPM(gmuy,      "S", "Conductance of avalanche current")
`OPM(gmuz,      "S", "Conductance of avalanche current")
`OPM(gmuex,     "S", "Conductance of extrinsic b-c junction")
`OPM(xgmuex,    "S", "Conductance of extrinsic b-c junction")
`OPM(grcvy,     "S", "Conductance of epilayer current")
`OPM(grcvz,     "S", "Conductance of epilayer current")
`OPD(rb_v,    "Ohm", "Variable base resistance")
`OPM(grbvx,     "S", "Early effect on variable base resistance")
`OPM(grbvy,     "S", "Early effect on variable base resistance")
`OPM(grbvz,     "S", "Early effect on variable base resistance")
`OPD(r_e,     "Ohm", "Emitter resistance")
`OPD(rb_c,    "Ohm", "Constant base resistance")
`OPD(rc_c,    "Ohm", "Collector contact resistance")
`OPD(rc_blx,  "Ohm", "Extrinsic buried layer resistance")
`OPD(rc_bli,  "Ohm", "Intrinsic buried layer resistance")
`ifdef SUBSTRATE
    `OPM(gs,        "S", "Conductance parasitic PNP transistor")
    `OPM(xgs,       "S", "Conductance parasitic PNP transistor")
    `OPM(gsf,       "S", "Conductance substrate failure current")
`endif
//Small signal equivalent circuit capacitances
`OPM(scbe,      "F", "Capacitance sidewall b-e junction")
`OPM(cbex,      "F", "Capacitance floor b-e junction")
`OPM(cbey,      "F", "Early effect on b-e diffusion charge")
`OPM(cbez,      "F", "Early effect on b-e diffusion charge")
`OPM(cbcx,      "F", "Early effect on b-c diffusion charge")
`OPM(cbcy,      "F", "Capacitance floor b-c junction")
`OPM(cbcz,      "F", "Capacitance floor b-c junction")
`OPM(cbcex,     "F", "Capacitance extrinsic b-c junction")
`OPM(xcbcex,    "F", "Capacitance extrinsic b-c junction")
`OPM(cb1b2,     "F", "Capacitance AC current crowding")
`OPM(cb1b2x,    "F", "Cross-capacitance AC current crowding")
`OPM(cb1b2y,    "F", "Cross-capacitance AC current crowding")
`OPM(cb1b2z,    "F", "Cross-capacitance AC current crowding")
`ifdef SUBSTRATE
    `OPM(cts,       "F", "Capacitance s-c junction")
`endif
//Approximate small signal equivalent circuit
`OPM(gm,        "S", "transconductance")
`OPP(beta,       "", "Current amplification")
`OPM(gout,      "S", "Output conductance")
`OPM(gmu,       "S", "Feedback transconductance")
`OPD(rb,      "Ohm", "Base resistance")
`OPD(rc,      "Ohm", "Collector resistance")
`OPM(cbe,       "F", "Base-emitter capacitance")
`OPM(cbc,       "F", "Base-collector capacitance")

//quantities to describe internal state of the model
`OPP(ft,       "Hz", "Good approximation for cut-off frequency")
`OPM(iqs,       "A", "Current at onset of quasi-saturation")
`OPP(xiwepi,     "", "Thickness of injection layer normalized to epi layer width")
`OPP(vb2c2star, "V", "Physical value of internal base-collector bias")

//self-heating
`ifdef SELFHEATING
    `OPM(pdiss,     "W", "Dissipation")
`endif
`OPP(tk,        "K", "Actual temperature")

//help variables
real dydx, dydz, gpi;
real gammax, gammay, gammaz, gbfx, gbfy, gbfz, alpha_ft;
real rx, ry, rz, rb1b2, rex, xrex, taut;
