// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Evaluate the operating point (output) variables
begin 

`ifdef __VAMS_COMPACT_MODELING__

   
// The external currents and the current gain
OP_ic     = I(<c>);        // External DC collector current
OP_ib     = I(<b>);        // External DC base Current

if (OP_ib == 0)
  begin
   OP_betadc = 0.0 ;
  end
else
  begin
    OP_betadc = OP_ic / OP_ib; // External DC Current gain
  end

// begin added in MXT 504.9:
OP_ie     = I(<e>);        // External DC emitter current
OP_vbe    = V(b, e);   // External base-emitter bias
OP_vce    = V(c, e);   // External collector-emitter bias
OP_vbc    = V(b, c);   // External base-collector bias

`ifdef SUBSTRATE
OP_is     = I(<s>);        // External DC emitter current
OP_vse    = V(s, e);   // External substrate-emitter bias
OP_vbs    = V(b, s);   // External base-substrate bias
OP_vsc    = V(s, c);   // External substrate-collector bias
`endif

// end added in MXT 504.9:

// The internal voltage differences
OP_vb2e1 = Vb2e1;  // Internal base-emiter bias
OP_vb2c2 = Vb2c2;  // Internal base-emiter bias 
OP_vb2c1 = Vb2c1;  // Internal base-collector bias including epilayer

OP_vb1c1 = Vb1b2 + Vb2c1;  // External base-collector bias without contact resistances

OP_vc4c1 = Vc4c1;  // Bias over intrinsic buried layer
OP_vc3c4 = Vc3c4;  // Bias over extrinsic buried layer

OP_ve1e  = - Vee1; // Bias over emiter resistance

// The branch currents
OP_in    = In;            // Main current
OP_ic1c2 = Ic1c2;         // Epilayer current
OP_ib1b2 = Ib1b2;         // Pinched-base current
OP_ib1   = Ib1;           // Ideal forward base current
OP_sib1  = Ib1_s;         // Ideal side-wall base current
//
// 504.8, RvdT, TU-Delft April. 2009: 
//
OP_izteb  = Izteb ;        // Zener tunneling current
//
OP_ib2   = Ib2;           // Non-ideal forward base current
OP_ib3   = Ib3;           // Non-ideal reverse base current
OP_iavl  = Iavl;          // Avalanche current
OP_iex   = Iex;           // Extrinsic reverse base current
OP_xiex  = XIex;          // Extrinsic reverse base current
`ifdef SUBSTRATE
OP_isub  = Isub;          // Substrate current
OP_xisub = XIsub;         // Substrate current
OP_isf   = Isf;           // Substrate-collector current
`endif
OP_ire   = - Vee1 / RE_TM;  // Current through emiter resistance
OP_irbc  = Vbb1 / RBC_TM; // Current through constant base resistance

OP_ircc  = Vcc3 * GCCxx_TM; // Current through collector contact resistance
OP_ircblx = Vc3c4 * GCCex_TM; // Current through extrinsic buried layer resistance
OP_ircbli = Vc4c1 * GCCin_TM; // Current through extrinsic buried layer resistance

// The branch charges
OP_qe    = Qe;    // Emitter charge or emitter neutral charge
OP_qte   = Qte;   // Base-emiter depletion charge
OP_sqte  = Qte_s; // Sidewall base-emiter depletion charge
OP_qbe   = Qbe;   // Base-emiter diffusion charge
OP_qbc   = Qbc;   // Base-collector diffusion charge
OP_qtc   = Qtc;   // Base-colector depletion charge
OP_qepi  = Qepi;  // Epilayer diffusion charge 
OP_qb1b2 = Qb1b2; // AC current crowding charge
OP_qtex  = Qtex;  // Extrinsic base-collector depletion charge
OP_xqtex = XQtex; // Extrinsic base-collector depletion charge
OP_qex   = Qex;   // Extrinsic base-collector diffusion charge
OP_xqex  = XQex;  // Extrinsic base-collector diffusion charge
`ifdef SUBSTRATE
OP_qts   = Qts;   // Collector substrate depletion charge
`endif

// Small signal equivalent circuit conductances and resistances

OP_gx     = - ddx(In, V(e1));      // Forward transconductance
OP_gy     = - ddx(In, V(c2));      // Reverse transconductance

OP_gz     = - ddx(In, V(c1));      // Reverse transconductance

OP_sgpi   = - ddx(Ib1_s, V(e))
            - ddx(Ib1_s, V(e1));   // Conductance sidewal b-e junction
OP_gpix   = - ddx(Ib1+Ib2, V(e1)); // Conductance floor b-e junction

OP_gpiy   = - ddx(Ib1, V(c2));  // Early effect on recombination base current
OP_gpiz   = - ddx(Ib1, V(c1));  // Early effect on recombination base current

OP_gmux   =  ddx( Iavl, V(e1)); // Early effect on avalanche current limitting
OP_gmuy   =  ddx( Iavl, V(c2)); // Conductance of avalanche current
OP_gmuz   =  - ddx(- Iavl, V(c1));   // Conductance of avalanche current

// Conductance extrinsic b-c current :
OP_gmuex  =   ddx(Iex+Ib3, V(e))
            + ddx(Iex+Ib3, V(b1))
            + ddx(Iex+Ib3, V(b2))
            + ddx(Iex+Ib3, V(e1))
            + ddx(Iex+Ib3, V(c2));

OP_xgmuex = ddx(XIex, V(b)) ; // Conductance extrinsic b-c current

OP_grcvy  =  - ddx(Ic1c2, V(c2)); // Conductance of epilayer current
OP_grcvz  =  - ddx(Ic1c2, V(c1));  // Conductance of epilayer current

OP_rbv    = 1.0 / (- ddx(Ib1b2, V(b2)) - ddx(Ib1b2, V(c2)));  // Base resistance

OP_grbvx  = - ddx(Ib1b2, V(e)) - ddx(Ib1b2, V(e1)); // Early effect on base resistance
OP_grbvy  = - ddx(Ib1b2, V(c2)); // Early effect on base resistance

OP_grbvz  = - ddx(Ib1b2, V(c1)); // Early effect on base resistance

OP_re     = RE_TM;  // Emiter resistance
OP_rbc    = RBC_TM; // Constant base resistance
OP_rcc    = RCCxx_TM; // Collector Contact resistance
OP_rcblx  = RCCex_TM; // Extrinsic buried layer resistance
OP_rcbli  = RCCin_TM; // Extrinsic buried layer resistance


`ifdef SUBSTRATE
OP_gs     = ddx(Isub, V(b)) + ddx(Isub, V(b1));  // Conductance parasitic PNP transitor
OP_xgs    = ddx(XIsub, V(b)) ; // Conductance parasitic PNP transistor
OP_gsf    = ddx(Isf, V(s)) ;   // Conductance substrate-collector current
`endif



// Small signal equivalent circuit capacitances
OP_scbe   = - ddx(Qte_s, V(e)) - ddx(Qte_s, V(e1)); // Capacitance sidewall b-e junction

OP_cbex   = - ddx(Qte + Qbe + Qe, V(e1)) ;           // Capacitance floor b-e junction

OP_cbey   = - ddx(Qbe, V(c2)); // Early effect on b-e diffusion junction

OP_cbez   = - ddx(Qbe, V(c1)); // Early effect on b-e diffusion junction

OP_cbcx   = - ddx(Qbc, V(e)) - ddx(Qbc, V(e1)); // Early effect on b-c diffusion junction


OP_cbcy   =  - ddx(Qtc + Qbc + Qepi, V(c2)); // Capacitance floor b-c junction
OP_cbcz   =  - ddx(Qtc + Qbc + Qepi, V(c1)); // Capacitance floor b-c junction

// Capacitance extrinsic b-c junction :
OP_cbcex  =   ddx(Qtex + Qex,V(e))    
            + ddx(Qtex + Qex,V(b1 ))    
            + ddx(Qtex + Qex,V(b2))    
            + ddx(Qtex + Qex,V(e1))    
            + ddx(Qtex + Qex,V(c2)) ;
 
// Capacitance extrinsic b-c junction :
OP_xcbcex =  ddx(XQtex + XQex, V(b)) ;

OP_cb1b2  = - ddx(Qb1b2, V(b2)) - ddx(Qb1b2, V(c2));  // Capacitance AC current crowding

OP_cb1b2x =  - ddx(Qb1b2, V(e)) - ddx(Qb1b2, V(e1));  // Cross-capacitance AC current crowding
OP_cb1b2y =  - ddx(Qb1b2, V(c2));  // Cross-capacitance AC current crowding
OP_cb1b2z =  - ddx(Qb1b2, V(c1)) ;  // Cross-capacitance AC current crowding

`ifdef SUBSTRATE
OP_cts    = ddx(Qts, V(s)) ; // Capacitance s-c junction
`endif

// Approximate small signal equivalent circuit
dydx = (OP_gx - OP_gmux)            / (OP_grcvy + OP_gmuy - OP_gy);
dydz = (OP_gz - OP_grcvz - OP_gmuz) / (OP_grcvy + OP_gmuy - OP_gy);
gpi  = OP_sgpi + OP_gpix + OP_gmux + OP_gpiz + OP_gmuz +    
       (OP_gpiy + OP_gmuy) * (dydx + dydz);
OP_gm   = (OP_grcvy * (OP_gx - OP_gmux +                   // Transconductance
           OP_gz - OP_gmuz) - OP_grcvz * 
          (OP_gy - OP_gmuy)) / (OP_grcvy + OP_gmuy - OP_gy);
OP_beta = OP_gm / gpi;                                     // Current amplification
OP_gout = ((OP_gy - OP_gmuy) * OP_grcvz -                  // Output conductance
          (OP_gz - OP_gmuz) * OP_grcvy) / 
          (OP_grcvy + OP_gmuy - OP_gy);
OP_gmu  = OP_gpiz + OP_gmuz + (OP_gpiy + OP_gmuy) * dydz + // Feedback transconductance
          OP_gmuex + OP_xgmuex;
OP_rb   = RBC_TM + OP_rbv;                                 // Base resistance
OP_rc   = OP_rcc + OP_rcblx + OP_rcbli;                    // Collector resistance
OP_cbe  = OP_cbex + OP_scbe + OP_cbcx +                    // Base-emitter capacitance
          (OP_cbey + OP_cbcy) * dydx + CBEO_M;
OP_cbc  = (OP_cbey + OP_cbcy) * dydz + OP_cbcz +           // Base-collector capacitance
           OP_cbcex + OP_xcbcex + CBCO_M;


// Quantities to describe internal state of the model
gammax = (OP_gpix + OP_gmux - OP_grbvx) * OP_rbv;
gammay = (OP_gpiy + OP_gmuy - OP_grbvy) * OP_rbv;
gammaz = (OP_gpiz + OP_gmuz - OP_grbvz) * OP_rbv;
gbfx   =  OP_gpix + OP_sgpi * (1.0 + gammax);
gbfy   =  OP_gpiy + OP_sgpi * gammay;
gbfz   =  OP_gpiz + OP_sgpi * gammaz;

// RvdT March 2008:
alpha_ft  = (1.0 + (OP_grcvy * dydx * OP_rc) +
            (OP_gx + gbfx + (OP_gy + gbfy) * dydx) * RE_TM)/
            (1.0 - (OP_grcvz + OP_grcvy * dydz) * OP_rc -
            (OP_gz + gbfz + (OP_gy + gbfy) * dydz) * RE_TM);

rx = pow((OP_grcvy * dydx + alpha_ft * (OP_grcvz + OP_grcvy * dydz)), -1);
rz = alpha_ft * rx;
ry = (1.0 - OP_grcvz * rz) / OP_grcvy;
rb1b2 = gammax * rx + gammay * ry + gammaz * rz;
rex = rz + rb1b2 - OP_rcbli;
xrex = rz + rb1b2 + RBC_TM * ((gbfx + OP_gmux) * rx + (gbfy + OP_gmuy) * ry +
       (gbfz + OP_gmuz) * rz) - OP_rcbli - OP_rcblx;

taut = OP_scbe * (rx + rb1b2) + (OP_cbex + OP_cbcx) * rx + (OP_cbey + OP_cbcy) *
          ry + (OP_cbez + OP_cbcz) * rz + OP_cbcex * rex + OP_xcbcex * xrex +
          (CBEO_M + CBCO_M) * (xrex - RCCxx_TM);

OP_ft = 1.0 / (2.0 * `PI * taut); // Good approximation for cut-off frequency
OP_iqs = Iqs;                     // Current at onset of quasi-saturation
OP_xiwepi = xi_w;                 // Thickness of injection layer
OP_vb2c2star = Vb2c2star;         // Physical value of internal base-collector bias

//self-heating
`ifdef SELFHEATING
OP_pdiss = power;                 // Dissipation
`endif

OP_tk = Tk;                       // Actual temperature

`endif
end
