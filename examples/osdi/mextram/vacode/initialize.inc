// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015-2020, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Initialize model constants

// Impact ionization constants (NPN - PNP)

if (type == 1) begin
    An = 7.03e7;
    Bn = 1.23e8;
end else begin
    An = 1.58e8;
    Bn = 2.04e8;
end

Xext1 = 1.0 - xext;

// Reference Temperature expressed in Kelvin:
Trk = tref + `C2K;
// Ambient Temperature expressed in Kelvin:
Tamb = $temperature + dta;
// Minimum conductance:
GMIN = $simparam("gmin", 0.0);
// icss flag
`ifdef SUBSTRATE
    if (icss > 0) begin
        GMIN_cs = GMIN;
    end else begin
        GMIN_cs = 0.0;
    end
`endif

// nff_t and nfr_t limiting width
eps_nf = 1e-3;
// Minimum bavl_t:
eps_bavl_t = 1e-3;

// begin: RvdT, November 2008; Zener tunneling current model

pow2_2m_pe = pow(2.0, 2.0 - pe);
pow2_pe_m2 = 1.0 / pow2_2m_pe;

// begin: RvdT, November 2008; Zener tunneling current model
//
// Comment added March 2009: this assumes vgzebok as a model parameter.
//
// Bandgap for Zener tunnel current model at reference temperature in eV:
//   vgzeb_tr = vgzebok - avgeb*Trk*Trk / (Trk + tvgeb);
//    `max_logexp(vgzeb_tr, vgzebok - avgeb*Trk*Trk / (Trk + tvgeb), 0.05, 0.1)
// end: RvdT, November 2008

// begin: RvdT March 2009:
// to decrease parameter interdependency,
// use vgzeb as a parameter, instead of vgzebok:
//   vgzeb   : bandgap for Zener tunneling at T = Tref,
//   vgzebok : bandgap for Zener tunneling at T = 0 K.
`max_logexp(vgzebok, vgzeb + avgeb * Trk * Trk / (Trk + tvgeb), 0.05, 0.1)
vgzeb_tr = vgzeb;
// end: RvdT March 2009: use vgzeb as a parameter, instead of vgzebok:
// end: RvdT, November 2008; Zener tunneling current model
inv_vgzeb_tr = 1.0 / vgzeb_tr;
inv_vde = 1.0 / vde;

// CB Zener tunneling
Vdc_zener = vdcctc;
Pc_zener = pc;
pow2_2m_pc = pow(2.0, 2.0 - Pc_zener);
pow2_pc_m2 = 1.0 / pow2_2m_pc;
`max_logexp(vgzcbok, vgzcb + avgcb * Trk * Trk / (Trk + tvgcb), 0.05, 0.1)
vgzcb_tr = vgzcb;
inv_vgzcb_tr = 1.0 / vgzcb_tr;
inv_vdc_zener = 1.0 / Vdc_zener;
Vfmax_z = -vzmin;

// Alpha factor of CB junction avalanche model
alpha_brcb = 1.0 - 1.0 /frevcb;
