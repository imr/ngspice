// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Temperature scaling of parameters

// The excess transistor temperature due to the self-heating
`ifdef SELFHEATING
    Tki = V(dt);
    // *** Convergence related smoothing ***
    if (Tki < 0.0) begin
        Tki = - ln(1.0 - Tki);
    end
    `linLog(Vdt, Tki, 200.0);
    //   `min_logexp(Vdt, Tki, 200.0, 10.0);
`else
    Vdt = 0.0;
`endif

// Temperature variables
Tk   = Tamb + Vdt;

tN  = Tk / Trk;
Vt  = `KBdivQQ * Tk;
Vtr = `KBdivQQ * Trk;
VtINV = 1.0 / Vt;
VtrINV = 1.0 / Vtr;
VdtINV = VtINV - VtrINV;

lntN = ln(tN) ;

// begin: RvdT, November 2008, "Zener tunneling model"
//      VGZEB_T = VGZEBOK - AVGEB*Tk*Tk / (Tk + TVGEB) ;
`max_logexp(VGZEB_T, VGZEBOK - AVGEB*Tk*Tk / (Tk + TVGEB), 0.05, 0.1) ;

// end: RvdT, November 2008, "Zener tunneling model"

// Depletion capacitances

UdeT = -3.0 * Vt * ln(tN) + VDE * tN + (1.0 - tN) * VGB;
`max_logexp(VDE_T, `VDLOW, UdeT, Vt);

UdcT = -3.0 * Vt * ln(tN) + VDC * tN + (1.0 - tN) * VGC;
`max_logexp(VDC_T, `VDLOW, UdcT, Vt);

`ifdef SUBSTRATE
    UdsT = -3.0 * Vt * ln(tN) + VDS * tN + (1.0 - tN) * VGS;
    `max_logexp(VDS_T, `VDLOW, UdsT, Vt);
`endif
inv_VDE_T = 1.0 / VDE_T ;
CJE_T_div_CJE = pow(VDE * inv_VDE_T, PE);
CJE_T = CJE * CJE_T_div_CJE ;

`ifdef SUBSTRATE
    CJS_T = CJS * pow(VDS / VDS_T, PS);
`endif

CJCscale = ((1.0 - XP) * pow(VDC / VDC_T, PC) + XP);
CJCscaleINV = 1.0 / CJCscale;

CJC_T = CJC * CJCscale;
XP_T  = XP  * CJCscaleINV;

// Resistances

// RvdT, November 2008:
// Instead of the following definition
//   RE_T  = RE  * pow(tN, AE);
// we use, here, and in all following powers of tN,
// the following computationally cheaper implementation:
RE_T  = RE  * exp(lntN * AE);
// This is based on the observation that exp() is faster than pow().
// Acknowledgement due to Geoffrey Coram.

RBV_T = RBV * exp(lntN * (AB - AQBO));
RBC_T = RBC * exp(lntN * AEX);

// RvdT, 30-11-2007: new collector resistances RCCxx_T, RCCex_T, RCCin_T
RCCxx_T = RCC * exp(lntN * AC);
RCCex_T = RCBLX * exp(lntN * ACBL);
RCCin_T = RCBLI * exp(lntN * ACBL);

RCV_T = RCV * exp(lntN * AEPI);

// Current gains

BF_T  = BF  * exp(lntN * (AE - AB - AQBO)) * exp(-DVGBF * VdtINV);
BRI_T = BRI * exp(-DVGBR * VdtINV);

// Currents and voltages

IS_T  = IS  * exp(lntN * (4.0 - AB - AQBO + DAIS)) * exp(-VGB * VdtINV);
IK_T  = IK  * exp(lntN * (1.0 - AB));
IBF_T = IBF * exp(lntN * (6.0 - 2.0 * MLF)) * exp(-VGJ * VdtINV / MLF);
IBR_T = IBR * tN * tN * exp(-VGC * VdtINV / 2.0);

// begin  RvdT, November 2008, MXT504.8_alpha
// T-scaling BE tunneling:
//
x = pow(VGZEB_T * inv_VGZEB_Tr, -0.5) ;
//   y = pow(VDE_T * inv_VDE, PE) ;
// more efficient, because we need both y and 1.0 / y:
y = 1.0 / CJE_T_div_CJE ;
// definition:
// nZEB_T = NZEB* pow(VGZEB_T/VGZEB_Tr, 1.5) * pow(VDE_T / VDE, PE-1) ;
// more efficient implementation:
//   nZEB_T = NZEB* VGZEB_T * VGZEB_T  * x * y * VDE /(VDE_T*VGZEB_Tr*VGZEB_Tr) ;
nZEB_T = NZEB* VGZEB_T * VGZEB_T  * x * y * VDE * inv_VDE_T*inv_VGZEB_Tr*inv_VGZEB_Tr ;

// definition:
// IZEB_T = IZEB* pow(VGZEB_T/VGZEB_Tr, -0.5) * pow(VDE_T / VDE, 2-PE) * exp(NZEB-nZEB_T);
// more efficient implementation:
IZEB_T = IZEB* x * VDE_T  * VDE_T * inv_VDE * inv_VDE * CJE_T_div_CJE * exp(NZEB-nZEB_T) ;
//
// end  RvdT, November 2008, MXT504.8_alpha

x = exp(lntN * AQBO) ;
VEF_T = VEF * x * CJCscaleINV;
//   VER_T = VER * x * pow(VDE / VDE_T, -PE);
VER_T = VER * x * y;

`ifdef SUBSTRATE
    ISS_T = ISS * exp(lntN * (4.0 - AS)) * exp(-VGS * VdtINV);
    // New 504.9:
    ICSS_T = ICSS * exp(lntN * (3.5 - 0.5 * ASUB)) * exp(-VGS * VdtINV);
    // End New 504.9.

    if ((ISS_T > 0.0))
        IKS_T = IKS * exp(lntN * (1.0 - AS)) * (IS_T / IS) * (ISS / ISS_T);
    else
        IKS_T = IKS * exp(lntN * (1.0 - AS));
`endif

// Transit times

TAUE_T = TAUE * exp(lntN * (AB - 2.0)) * exp(-DVGTE * VdtINV);
TAUB_T = TAUB * exp(lntN * (AQBO + AB - 1.0));
TEPI_T = TEPI * exp(lntN * (AEPI - 1.0));
TAUR_T = TAUR * (TAUB_T + TEPI_T) / (TAUB + TEPI);

// Avalanche constant

Tk300 = Tk - 300.0;
// RvdT, 15-02-2008: prevent division by zero and overflow at high temperatures:
if (Tk < 525.0)
begin
    BnT = Bn * (1.0 + 7.2e-4 * Tk300 - 1.6e-6 * Tk300 * Tk300) ;
end
else
begin
    BnT = Bn * 1.081 ;
end

// Heterojunction features

DEG_T = DEG * exp(lntN * AQBO);

`ifdef SELFHEATING
    // Temperature scaling of the thermal resistance

    RTH_Tamb = RTH * pow(Tamb / Trk, ATH);
`endif

// MULT - scaling

IS_TM  = IS_T  * MULT;
IK_TM  = IK_T  * MULT;
IBF_TM = IBF_T * MULT;
IBR_TM = IBR_T * MULT;
// RvdT: November 2008, Zener tunneling parameters
IZEB_TM = IZEB_T * MULT ;

// end Zener tunneling parameters


IHC_M  = IHC   * MULT;
`ifdef SUBSTRATE
    ISS_TM = ISS_T * MULT;
    // New: 504.9
    ICSS_TM = ICSS_T * MULT;
    IKS_TM = IKS_T * MULT;
`endif
CJE_TM = CJE_T * MULT;
CJC_TM = CJC_T * MULT;

// begin  RvdT, 28-10-2008, MXT504.8_alpha
// Base-emitter tunneling current Mult scaling:
//   BTJE_TM = BTJE_T * MULT;
// end  RvdT, 28-10-2008, MXT504.8_alpha


`ifdef SUBSTRATE
    CJS_TM = CJS_T * MULT;
`endif

RE_TM      = RE_T     * invMULT;
RBC_TM     = RBC_T    * invMULT;
RBV_TM     = RBV_T    * invMULT;
// RvdT, 30-01-2007: new collector resistances:
RCCxx_TM   = RCCxx_T  * invMULT;
RCCex_TM   = RCCex_T  * invMULT;
RCCin_TM   = RCCin_T  * invMULT;
RCV_TM     = RCV_T    * invMULT;


// RvdT, 03-12-2007: new collector conductances
if (RCC > 0.0)
begin
    GCCxx_TM = 1.0 / RCCxx_TM ;
end
else
begin
    GCCxx_TM = 0 ;
end

if (RCBLX > 0.0)
begin
    GCCex_TM = 1.0 / RCCex_TM ;
end
else
begin
    GCCex_TM = 0 ;
end

if (RCBLI > 0.0)
begin
    GCCin_TM = 1.0 / RCCin_TM ;
end
else
begin
    GCCin_TM = 0 ;
end

`ifdef SELFHEATING
    RTH_Tamb_M = RTH_Tamb * invMULT;
`endif
