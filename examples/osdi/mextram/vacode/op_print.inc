// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015-2020, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// print the operating point (output) variables

// The external voltage differences
$strobe("vbe : ", vbe);  // External base-emitter bias
$strobe("vce : ", vce);  // External collector-emitter bias
$strobe("vbc : ", vbc);  // External base-collector bias

`ifdef SUBSTRATE
    $strobe("vse : ", vse);  // External substrate-emitter bias
    $strobe("vbs : ", vbs);  // External base-substrate bias
    $strobe("vsc : ", vsc);  // External substrate-collector bias
`endif

// The external currents and the current gain
$strobe("ic : ", ic);        // External DC collector current
$strobe("ib : ", ib);        // External DC base Current

// begin added in MXT 504.9:
$strobe("ie : ", ie);        // External DC emitter current
`ifdef SUBSTRATE
    $strobe("isx : ", isx);     // External DC substrate current
`endif
// end added in MXT 504.9
$strobe("betadc :", betadc);   // External DC Current gain

// The internal voltage differences
$strobe("vb2e1 : ", vb2e1);  // Internal base-emitter bias
$strobe("vb2c2 : ", vb2c2);  // Internal base-emitter bias 
$strobe("vb2c1 : ", vb2c1);  // Internal base-collector bias including epilayer

$strobe("vb1c1 : " ,vb1c1);  // External base-collector bias without contact resistances

$strobe("vc4c1 : ", vc4c1);  // Bias over intrinsic buried layer
$strobe("vc3c4 : ", vc3c4);  // Bias over extrinsic buried layer

$strobe("ve1e  : ", ve1e);   // Bias over emitter resistance

// The branch currents
$strobe("in    : ", in);            // Main current
$strobe("ic1c2 : ", ic1c2);         // Epilayer current
$strobe("ib1b2 : ", ib1b2);         // Pinched-base current
$strobe("ib1   : ", ib1);           // Ideal forward base current
$strobe("ib1s  : ", ib1s);          // Ideal side-wall base current
$strobe("izteb  : ", izteb);         // Zener tunneling current in the emitter base junction
$strobe("ib2   : ", ib2);         // Non-ideal forward base current
$strobe("ib3   : ", ib3);         // Non-ideal reverse base current
$strobe("iavl  : ", iavl);         // Avalanche current
$strobe("iex   : ", iex);         // Extrinsic reverse base current
$strobe("xiex  : ", xiex);         // Extrinsic reverse base current
`ifdef SUBSTRATE
    $strobe("isub  : ", isub);          // Substrate current
    $strobe("xisub : ", xisub);         // Substrate current
    $strobe("isf   : ", isf);           // Substrate-collector current
`endif
$strobe("ire   : ", ire);  // Current through emitter resistance
$strobe("irbc  : ", irbc); // Current through constant base resistance

$strobe("ircc  : ", ircc); // Current through collector contact resistance
$strobe("ircblx: ", ircblx); // Current through extrinsic buried layer resistance
$strobe("ircbli: ", ircbli); // Current through extrinsic buried layer resistance

// The branch charges
$strobe("qe    : ", qe); // Emitter charge or emitter neutral charge
$strobe("qte   : ", qte); // Base-emitter depletion charge
$strobe("sqte  : ", sqte); // Sidewall base-emitter depletion charge
$strobe("qbe   : ", qbe); // Base-emitter diffusion charge
$strobe("qbc   : ", qbc); // Base-collector diffusion charge
$strobe("qtc   : ", qtc); // Base-collector depletion charge
$strobe("qepi  : ", qepi); // Epilayer diffusion charge 
$strobe("qb1b2 : ", qb1b2); // AC current crowding charge
$strobe("qtex  : ", qtex); // Extrinsic base-collector depletion charge
$strobe("xqtex : ", xqtex); // Extrinsic base-collector depletion charge
$strobe("qex   : ", qex); // Extrinsic base-collector diffusion charge
$strobe("xqex  : ", xqex); // Extrinsic base-collector diffusion charge
`ifdef SUBSTRATE
    $strobe("qts   : ", qts); // Collector substrate depletion charge
`endif



// Small signal equivalent circuit conductances and resistances

$strobe("gx     : ",  gx); // Forward transconductance

$strobe("gy     : ",gy); // Reverse transconductance
$strobe("gz     : ",gz); // Reverse transconductance
$strobe("sgpi   : ",sgpi); // Conductance sidewall b-e junction
$strobe("gpix   : ",gpix); // Conductance floor b-e junction

$strobe("gpiy   : ",gpiy); // Early effect on recombination base current
$strobe("gpiz   : ",gpiz); // Early effect on recombination base current

$strobe("gmux   : ",gmux); // Early effect on avalanche current limiting
$strobe("gmuy   : ",gmuy); // Conductance of avalanche current
$strobe("gmuz   : ",gmuz); // Conductance of avalanche current
$strobe("gmuex  : ",gmuex); // Conductance extrinsic b-c current
$strobe("xgmuex : ",xgmuex); // Conductance extrinsic b-c current

$strobe("grcvy  : ",grcvy); // Conductance of epilayer current
$strobe("grcvz  : ",grcvz); // Conductance of epilayer current

$strobe("rb_v   : ",rb_v); // Variable base resistance

$strobe("grbvx  : ",grbvx); // Early effect on variable base resistance
$strobe("grbvy  : ",grbvy); // Early effect on variable base resistance
$strobe("grbvz  : ",grbvz); // Early effect on variable base resistance
$strobe("r_e    : ",r_e); // Emitter resistance
$strobe("rb_c   : ",rb_c); // Constant base resistance
$strobe("rc_c   : ",rc_c); // Collector Contact resistance
$strobe("rc_blx : ",rc_blx); // Extrinsic buried layer resistance
$strobe("rc_bli : ",rc_bli); // Extrinsic buried layer resistance


`ifdef SUBSTRATE
    $strobe("gs     : ", gs); // Conductance parasitic PNP transistor
    $strobe("xgs    : ", xgs); // Conductance parasitic PNP transistor
    $strobe("gsf    : ", gsf); // Conductance substrate failure current
`endif

// Small signal equivalent circuit capacitances
$strobe("scbe   : ", scbe); // Capacitance sidewall b-e junction
$strobe("cbex   : ", cbex); // Capacitance floor b-e junction
$strobe("cbey   : ", cbey); // Early effect on b-e diffusion junction
$strobe("cbez   : ", cbez); // Early effect on b-e diffusion junction
$strobe("cbcx   : ", cbcx); // Early effect on b-c diffusion junction
$strobe("cbcy   : ",  cbcy); // Capacitance floor b-c junction
$strobe("cbcz   : ", cbcz); // Capacitance floor b-c junction
$strobe("cbcex  : ", cbcex); // Capacitance extrinsic b-c junction 
$strobe("xcbcex : ", xcbcex); // Capacitance extrinsic b-c junction
$strobe("cb1b2  : ", cb1b2); // Capacitance AC current crowding
$strobe("cb1b2x : ", cb1b2x); // Cross-capacitance AC current crowding
$strobe("cb1b2y : ", cb1b2y); // Cross-capacitance AC current crowding
$strobe("cb1b2z : ", cb1b2z); // Cross-capacitance AC current crowding
`ifdef SUBSTRATE
    $strobe("cts    : ", cts); // Capacitance s-c junction
`endif
$strobe("gm   : ",  gm); // Transconductance
$strobe("beta : ", beta); // Current amplification
$strobe("gout : ", gout); // Output conductance

$strobe("gmu  : ",  gmu); // Feedback transconductance
$strobe("rb   : ",  rb); // Base resistance
$strobe("rc   : ",  rc); // Collector resistance
$strobe("cbe  : ",   cbe); // Base-emitter capacitance

$strobe("cbc       : ", cbc); // Base-collector capacitance
$strobe("ft        : ", ft); // Good approximation for cut-off frequency

$strobe("iqs       : ", iqs); // Current at onset of quasi-saturation
$strobe("xiwepi    : ", xiwepi); // Thickness of injection layer
$strobe("vb2c2star : ", vb2c2star); // Physical value of internal base-collector bias

// self-heating
`ifdef SELFHEATING
    $strobe("pdiss : ", pdiss); // Dissipation
`endif

$strobe("tk : ", tk); // Actual temperature
