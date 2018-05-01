// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

//
// Operation point (output) variables
//

// The external currents and current gain
`OPP(OP_ic,        A, External DC collector current)
`OPP(OP_ib,        A, External DC base current)
`OPP(OP_betadc,     , External DC current gain Ic/Ib)

// begin added in MXT 504.9:
`OPP(OP_ie,        A, External DC emitter current)

// The external biases
`OPP(OP_vbe,     V, External base-emitter bias)
`OPP(OP_vce,     V, External collector-emitter bias)
`OPP(OP_vbc,     V, External base-collector bias)

`ifdef SUBSTRATE
`OPP(OP_is,        A, External DC substrate current)
`OPP(OP_vse,     V, External substrate-emitter bias)
`OPP(OP_vbs,     V, External base-substrate bias)
`OPP(OP_vsc,     V, External substrate-collector bias)
`endif

// end added in MXT 504.9
// The internal biases
`OPP(OP_vb2e1,     V, Internal base-emitter bias)
`OPP(OP_vb2c2,     V, Internal base-collector bias)
`OPP(OP_vb2c1,     V, Internal base-collector bias including epilayer)
`OPP(OP_vb1c1,     V, External base-collector bias without contact resistances)
`OPP(OP_vc4c1,     V, Bias over intrinsic buried layer)
`OPP(OP_vc3c4,     V, Bias over extrinsic buried layer)
`OPP(OP_ve1e,      V, Bias over emitter resistance)

// The actual currents
`OPP(OP_in,        A,   Main current)
`OPP(OP_ic1c2,     A,   Epilayer current)
`OPP(OP_ib1b2,     A,   Pinched-base current)
`OPP(OP_ib1,       A,   Ideal forward base current)
`OPP(OP_sib1,      A,   Ideal side-wall base current)
//
// 504.8, RvdT, TU-Delft April. 2009, Zener tunneling current:
//
`OPP(OP_izteb,      A,   Zener tunneling current in the emitter base junction)
//
`OPP(OP_ib2,       A,   Non-ideal forward base current)
`OPP(OP_ib3,       A,   Non-ideal reverse base current)
`OPP(OP_iavl,      A,   Avalanche current)
`OPP(OP_iex,       A,   Extrinsic reverse base current)

`OPP(OP_xiex,      A,   Extrinsic reverse base current)
`ifdef SUBSTRATE
`OPP(OP_isub,      A,   Substrate current)
`OPP(OP_xisub,     A,   Substrate current)
`OPP(OP_isf,       A,   Substrate failure current)
`endif
`OPP(OP_ire,       A,   Current through emitter resistance)
`OPP(OP_irbc,      A,   Current through constant base resistance)
`OPP(OP_ircblx,    A,   Current through extrinsic buried layer resistance)
`OPP(OP_ircbli,    A,   Current through intrinsic buried layer resistance)
`OPP(OP_ircc,      A,   Current through collector contact resistance)

//The actual charges
`OPP(OP_qe,        C,   Emitter charge or emitter neutral charge)
`OPP(OP_qte,       C,   Base-emitter depletion charge)
`OPP(OP_sqte,      C,   Sidewall base-emitter depletion charge)
`OPP(OP_qbe,       C,   Base-emitter diffusion charge)
`OPP(OP_qbc,       C,   Base_collector diffusion charge)
`OPP(OP_qtc,       C,   Base-collector depletion charge)
`OPP(OP_qepi,      C,   Epilayer diffusion charge)
`OPP(OP_qb1b2,     C,   AC current crowding charge)
`OPP(OP_qtex,      C,   Extrinsic base-collector depletion charge)
`OPP(OP_xqtex,     C,   Extrinsic base-collector depletion charge)
`OPP(OP_qex,       C,   Extrinsic base-collector diffusion charge)
`OPP(OP_xqex,      C,   Extrinsic base-collector diffusion charge)
`ifdef SUBSTRATE
`OPP(OP_qts,       C,   Collector-substrate depletion charge)
`endif

//Small signal equivalent circuit conductances and resistances
`OPP(OP_gx,        S,   Forward transconductance)
`OPP(OP_gy,        S,   Reverse transconductance)
`OPP(OP_gz,        S,   Reverse transconductance)
`OPP(OP_sgpi,      S,   Conductance sidewall b-e junction)
`OPP(OP_gpix,      S,   Conductance floor b-e junction)
`OPP(OP_gpiy,      S,   Early effect on recombination base current)
`OPP(OP_gpiz,      S,   Early effect on recombination base current)
`OPP(OP_gmux,      S,   Early effect on avalanche current limiting)
`OPP(OP_gmuy,      S,   Conductance of avalanche current)
`OPP(OP_gmuz,      S,   Conductance of avalanche current)
`OPP(OP_gmuex,     S,   Conductance of extrinsic b-c junction)
`OPP(OP_xgmuex,    S,   Conductance of extrinsic b-c junction)
`OPP(OP_grcvy,     S,   Conductance of epilayer current)
`OPP(OP_grcvz,     S,   Conductance of epilayer current)
`OPP(OP_rbv,       Ohm, Base resistance)
`OPP(OP_grbvx,     S,   Early effect on base resistance)
`OPP(OP_grbvy,     S,   Early effect on base resistance)
`OPP(OP_grbvz,     S,   Early effect on base resistance)
`OPP(OP_re,        Ohm, Emitter resistance)
`OPP(OP_rbc,       Ohm, Constant base resistance)
`OPP(OP_rcc,       Ohm, Collector contact resistance)
`OPP(OP_rcblx,       Ohm, Extrinsic buried layer resistance)
`OPP(OP_rcbli,       Ohm, Intrinsic buried layer resistance)
`ifdef SUBSTRATE
`OPP(OP_gs,        S,   Conductance parasitic PNP transistor)
`OPP(OP_xgs,       S,   Conductance parasitic PNP transistor)
`OPP(OP_gsf,       S,   Conductance substrate failure current)
`endif
//Small signal equivalent circuit capacitances
`OPP(OP_scbe,      F,   Capacitance sidewall b-e junction)
`OPP(OP_cbex,      F,   Capacitance floor b-e junction)
`OPP(OP_cbey,      F,   Early effect on b-e diffusion charge)
`OPP(OP_cbez,      F,   Early effect on b-e diffusion charge)
`OPP(OP_cbcx,      F,   Early effect on b-c diffusion charge)
`OPP(OP_cbcy,      F,   Capacitance floor b-c junction)
`OPP(OP_cbcz,      F,   Capacitance floor b-c junction)
`OPP(OP_cbcex,     F,   Capacitance extrinsic b-c junction)
`OPP(OP_xcbcex,    F,   Capacitance extrinsic b-c junction)
`OPP(OP_cb1b2,     F,   Capacitance AC current crowding)
`OPP(OP_cb1b2x,    F,   Cross-capacitance AC current crowding)
`OPP(OP_cb1b2y,    F,   Cross-capacitance AC current crowding)
`OPP(OP_cb1b2z,    F,   Cross-capacitance AC current crowding)
`ifdef SUBSTRATE
`OPP(OP_cts,       F,   Capacitance s-c junction)
`endif
//Approximate small signal equivalent circuit
`OPP(OP_gm,        S,transconductance)
`OPP(OP_beta,       ,   Current amplification)
`OPP(OP_gout,      S,   Output conductance)
`OPP(OP_gmu,       S,   Feedback transconductance)
`OPP(OP_rb,        Ohm, Base resistance)
`OPP(OP_rc,        Ohm, Collector resistance)
`OPP(OP_cbe,       C,   Base-emitter capacitance)
`OPP(OP_cbc,       C,   Base-collector capacitance)

//quantities to describe internal state of the model
`OPP(OP_ft,         ,   Good approximation for cut-off frequency)
`OPP(OP_iqs,       A,   Current at onset of quasi-saturation)
`OPP(OP_xiwepi,    m,   Thickness of injection layer)
`OPP(OP_vb2c2star, V, Physical value of internal base-collector bias)

//self-heating
`ifdef SELFHEATING
`OPP(OP_pdiss,     W, Dissipation)
`endif
`OPP(OP_tk,        K, Actual temperature)

//help variables
real dydx, dydz, gpi;
real gammax, gammay, gammaz, gbfx, gbfy, gbfz, alpha_ft;
real rx, ry, rz, rb1b2, rex, xrex, taut;
