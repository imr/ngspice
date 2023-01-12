// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015-2020, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Temperature scaling of parameters

// The excess transistor temperature due to the self-heating
`ifdef SELFHEATING
    Tki = Temp(dt);
    // *** Convergence related smoothing ***
    if (Tki < 0.0) begin
        Tki = - ln(1.0 - Tki);
    end
    `linLog(Vdt, Tki, 200.0)
    //   `min_logexp(Vdt, Tki, 200.0, 10.0)
`else
    Vdt = 0.0;
`endif

// Temperature variables
Tk  = Tamb + Vdt;

tN  = Tk / Trk;
Vt  = `KBdivQQ * Tk;
Vtr = `KBdivQQ * Trk;
VtINV = 1.0 / Vt;
VtrINV = 1.0 / Vtr;
VdtINV = VtINV - VtrINV;
dT = Tk - Trk;

lntN = ln(tN);

// RvdT, November 2008, EB Zener tunneling model
//        vgzeb_t = vgzebok - avgeb*Tk*Tk / (Tk + tvgeb);
`max_logexp(vgzeb_t, vgzebok - avgeb*Tk*Tk / (Tk + tvgeb), 0.05, 0.1)

// 2016. CB Zener tunneling model
`max_logexp(vgzcb_t, vgzcbok - avgcb*Tk*Tk / (Tk + tvgcb), 0.05, 0.1)

// Diffusion voltages for depletion capacitances, quasi-saturation, and Zener tunneling
// EB vde - shared by CV and Zener
UdeT = -3.0 * Vt * ln(tN) + vde * tN + (1.0 - tN) * vgb;
`max_logexp(vde_t, `VDLOW, UdeT, Vt)

// CB vdc dedicated to quasi-saturation
UdcT = -3.0 * Vt * ln(tN) + vdc * tN + (1.0 - tN) * vgc;
`max_logexp(vdc_t, `VDLOW, UdcT, Vt)

// Qex 2019
UdcexT = -3.0 * Vt * ln(tN) + vdcex * tN + (1.0 - tN) * vgc;
`max_logexp(vdcex_t, `VDLOW, UdcexT, Vt)

// CB vdc dedicated to CV
UdcT_ctc = -3.0 * Vt * ln(tN) + vdcctc * tN + (1.0 - tN) * vgc;
`max_logexp(vdc_ctc_t, `VDLOW, UdcT_ctc, Vt)

// CB vdc dedicated to Zener tunneling
UdcT_zener = -3.0 * Vt * ln(tN) + Vdc_zener * tN + (1.0 - tN) * vgc;
`max_logexp(vdc_zener_t, `VDLOW, UdcT_zener, Vt)

// CS
`ifdef SUBSTRATE
    UdsT = -3.0 * Vt * ln(tN) + vds * tN + (1.0 - tN) * vgs;
    `max_logexp(vds_t, `VDLOW, UdsT, Vt)
`endif

inv_vde_t = 1.0 / vde_t;
inv_vdc_zener_t = 1.0 / vdc_zener_t;
cje_t_div_cje = pow(vde * inv_vde_t, pe);
cjc_t_div_cjc_zener = pow(Vdc_zener * inv_vdc_zener_t, Pc_zener);
cje_t = cje * cje_t_div_cje;

`ifdef SUBSTRATE
    cjs_t = cjs * pow(vds / vds_t, ps);
`endif

cjc_scale = ((1.0 - xp) * pow(vdcctc / vdc_ctc_t, pc) + xp);
cjc_scale_inv = 1.0 / cjc_scale;

cjc_t = cjc * cjc_scale;
xp_t = xp * cjc_scale_inv;

// Resistances

// RvdT, November 2008:
// Instead of the following definition
//   re_t  = re  * pow(tN, ae);
// we use, here, and in all following powers of tN,
// the following computationally cheaper implementation:
re_t  = re  * exp(lntN * ae);
// This is based on the observation that exp() is faster than pow().
// Acknowledgement due to Geoffrey Coram.

rbv_t = rbv * exp(lntN * (ab - aqbo));
rbc_t = rbc * exp(lntN * aex);

// RvdT, 30-11-2007: new collector resistances rcc_xx_t, rcc_ex_t, rcc_in_t
rcc_xx_t = rcc * exp(lntN * ac);
rcc_ex_t = rcblx * exp(lntN * acbl);
rcc_in_t = rcbli * exp(lntN * acbl);

rcv_t = rcv * exp(lntN * aepi);

// Currents and voltages
// main current non-ideality factor - forward
if (tnff != 0.0) begin
    nff_t_tmp = nff * (1.0 + dT * tnff);
    `max_logexp(nff_t_tmp, nff_t_tmp, 1.0, eps_nf)
    nff_t = nff_t_tmp - eps_nf * `LN2;
end else begin
    nff_t = nff;
end
// main current non-ideality factor - reverse
if (tnfr != 0.0) begin
    nfr_t_tmp = nfr * (1.0 + dT * tnfr);
    `max_logexp(nfr_t_tmp, nfr_t_tmp, 1.0, eps_nf)
    nfr_t = nfr_t_tmp - eps_nf * `LN2;
end else begin
    nfr_t = nfr;
end

// avalanche coefficient bavl
bavl_t_tmp = bavl * (1.0 + tbavl * dT);
`max_hyp0(bavl_t, bavl_t_tmp, eps_bavl_t)

// Saturation currents for main current and diode currents, knee currents
is_t = is * exp(lntN * (4.0 - ab - aqbo + dais) / nff_t) * exp(-vgb * VdtINV / nff_t);
ik_t = ik * exp(lntN * (1.0 - ab));
ikbx_t = ikbx * exp(lntN * (1.0 - acx));
ibf_t = ibf * exp(lntN * (6.0 - 2.0 * mlf)) * exp(-vgj * VdtINV / mlf);
ibr_t = ibr * exp(lntN * (6.0 - 2.0 * mlr)) * exp(-vgc * VdtINV / mlr);
ibi_t = ibi * exp(lntN * (4.0 - ae + dais) / nbi) * exp(-vge * VdtINV / nbi);
ibx_t = ibx * exp(lntN * (4.0 - acx + dais)) * exp(-vgcx * VdtINV);
ibis_t = ibis * exp(lntN * (4.0 - ae + dais) / nbis) * exp(-vge * VdtINV / nbis);
ibfs_t = ibfs * exp(lntN * (6.0 - 2.0 * mlfs)) * exp(-vgj * VdtINV / mlfs);
isibrel_t = isibrel * exp(lntN * (4.0 / nfibrel)) * exp(-vgj * VdtINV / nfibrel);
// begin  RvdT, November 2008, MXT504.8_alpha
// T-scaling BE tunneling:
//
x = pow(vgzeb_t * inv_vgzeb_tr, -0.5);
//   y = pow(vde_t * inv_vde, pe);
// more efficient, because we need both y and 1.0 / y:
y = 1.0 / cje_t_div_cje;
// definition:
// nzeb_t = nzeb* pow(vgzeb_t/vgzeb_tr, 1.5) * pow(vde_t / vde, pe-1);
// more efficient implementation:
//   nzeb_t = nzeb* vgzeb_t * vgzeb_t * x * y * vde /(vde_t*vgzeb_tr*vgzeb_tr);
nzeb_t = nzeb* vgzeb_t * vgzeb_t * x * y * vde * inv_vde_t * inv_vgzeb_tr * inv_vgzeb_tr;

// definition:
// izeb_t = izeb* pow(vgzeb_t/vgzeb_tr, -0.5) * pow(vde_t / vde, 2-pe) * exp(nzeb-nzeb_t);
// more efficient implementation:
izeb_t = izeb * x * vde_t * vde_t * inv_vde * inv_vde * cje_t_div_cje * exp(nzeb - nzeb_t);
//
// end  RvdT, November 2008, MXT504.8_alpha

// T-scaling CB tunneling:
inv_vdc_zener_t = 1.0 / vdc_zener_t;
xx = pow(vgzcb_t * inv_vgzcb_tr, -0.5);
yy = 1.0 / cjc_t_div_cjc_zener;
nzcb_t = nzcb * vgzcb_t * vgzcb_t  * xx * yy * Vdc_zener * inv_vdc_zener_t * inv_vgzcb_tr * inv_vgzcb_tr;
izcb_t = izcb * xx * vdc_zener_t  * vdc_zener_t * inv_vdc_zener * inv_vdc_zener * cjc_t_div_cjc_zener * exp(nzcb - nzcb_t);

x = exp(lntN * aqbo);
vef_t = vef * x * cjc_scale_inv;
//   ver_t = ver * x * pow(vde / vde_t, -pe);
ver_t = ver * x * y;

`ifdef SUBSTRATE
    iss_t = iss * exp(lntN * (4.0 - as)) * exp(-vgs * VdtINV);
    // New 504.9:
    icss_t = icss * exp(lntN * (3.5 - 0.5 * asub)) * exp(-vgs * VdtINV);
    // End New 504.9.

    iks_t = iks * exp(lntN * (1.0 - as));
    ikcs_t = ikcs * exp(lntN * (1.0 - asub));
`endif

// Transit times

taue_t = taue * exp(lntN * (ab - 2.0)) * exp(-dvgte * VdtINV);
taub_t = taub * exp(lntN * (aqbo + ab - 1.0));
tepi_t = tepi * exp(lntN * (aepi - 1.0));
taur_t = taur * (taub_t + tepi_t) / (taub + tepi);
tauex_t = tauex * exp(lntN * (aepiex - 1.0));

// Avalanche constant

Tk300 = Tk - 300.0;
// RvdT, 15-02-2008: prevent division by zero and overflow at high temperatures:
if (Tk < 525.0) begin
    Bnt = Bn * (1.0 + 7.2e-4 * Tk300 - 1.6e-6 * Tk300 * Tk300);
end else begin
    Bnt = Bn * 1.081;
end

// Heterojunction features

deg_t = deg * exp(lntN * aqbo);

`ifdef SELFHEATING
    // Temperature scaling of the thermal resistance
    rth_tamb = rth * pow(Tamb / Trk, ath);
`endif

// mult - scaling
// RvdT: November 2008, EB Zener tunneling parameters
// 2016, CB tunneling izcb

// RvdT, 30-01-2007: new collector resistances:

// RvdT, 03-12-2007: new collector conductances
if (rcc > 0.0) begin
    gcc_xx_t = 1.0 / rcc_xx_t;
end else begin
    gcc_xx_t = 0.0;
end

if (rcblx > 0.0) begin
    gcc_ex_t = 1.0 / rcc_ex_t;
end else begin
    gcc_ex_t = 0.0;
end

if (rcbli > 0.0) begin
    gcc_in_t = 1.0 / rcc_in_t;
end else begin
    gcc_in_t = 0.0;
end
