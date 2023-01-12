// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015-2020, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

// Evaluate the operating point (output) variables

// The external currents and the current gain
ic = I(<c>);        // External DC collector current
ib = I(<b>);        // External DC base Current

if (ib == 0.0) begin
    betadc = 0.0;
end else begin
    betadc = ic / ib; // External DC Current gain
end

// begin added in MXT 504.9:
ie     = I(<e>);        // External DC emitter current
vbe    = V(b, e);   // External base-emitter bias
vce    = V(c, e);   // External collector-emitter bias
vbc    = V(b, c);   // External base-collector bias

`ifdef SUBSTRATE
    isx    = I(<s>);        // External DC substrate current
    vse    = V(s, e);   // External substrate-emitter bias
    vbs    = V(b, s);   // External base-substrate bias
    vsc    = V(s, c);   // External substrate-collector bias
`endif

// end added in MXT 504.9:

// The internal voltage differences
vb2e1 = Vb2e1;  // Internal base-emitter bias
vb2c2 = Vb2c2;  // Internal base-emitter bias
vb2c1 = Vb2c1;  // Internal base-collector bias including epilayer

vb1c1 = Vb1b2 + Vb2c1;  // External base-collector bias without contact resistances

vc4c1 = Vc4c1;  // Bias over intrinsic buried layer
vc3c4 = Vc3c4;  // Bias over extrinsic buried layer

ve1e  = - Vee1; // Bias over emitter resistance

// The branch currents
in    = In;            // Main current
ic1c2 = Ic1c2;         // Epilayer current
ib1b2 = Ib1b2;         // Pinched-base current
ib1   = Ib1;           // Ideal forward base current
ib1s  = Ib1_s;         // Ideal side-wall base current
ib2s  = Ib2_s;         // Non-ideal side-wall base current
ibrel = Ibrel;         // Additional non-ideal base current for reliability simulation
//
// 504.8, RvdT, TU-Delft April. 2009:
//
izteb  = Izteb;        // Zener emitter-base tunneling current
//
iztcb = Iztcb;         // Zener collector-base tunneling current
ib2   = Ib2;           // Non-ideal forward base current
ib3   = Ib3;           // Non-ideal reverse base current
iavl  = Iavl;          // Avalanche current
iex   = Iex;           // Extrinsic reverse base current
xiex  = XIex;          // Extrinsic reverse base current
`ifdef SUBSTRATE
    isub  = Isub;          // Substrate current
    xisub = XIsub;         // Substrate current
    isf   = Isf;           // Substrate-collector current
`endif
ire   = - Vee1 / re_t;  // Current through emitter resistance
irbc  = Vbb1 / rbc_t; // Current through constant base resistance

ircc  = Vcc3 * gcc_xx_t; // Current through collector contact resistance
ircblx = Vc3c4 * gcc_ex_t; // Current through extrinsic buried layer resistance
ircbli = Vc4c1 * gcc_in_t; // Current through extrinsic buried layer resistance

// The branch charges
qe    = Qe;    // Emitter charge or emitter neutral charge
qte   = Qte;   // Base-emitter depletion charge
sqte  = Qte_s; // Sidewall base-emitter depletion charge
qbe   = Qbe;   // Base-emitter diffusion charge
qbc   = Qbc;   // Base-collector diffusion charge
qtc   = Qtc;   // Base-collector depletion charge
qepi  = Qepi;  // Epilayer diffusion charge
qb1b2 = Qb1b2; // AC current crowding charge
qtex  = Qtex;  // Extrinsic base-collector depletion charge
xqtex = XQtex; // Extrinsic base-collector depletion charge
qex   = Qex;   // Extrinsic base-collector diffusion charge
xqex  = XQex;  // Extrinsic base-collector diffusion charge
`ifdef SUBSTRATE
    qts   = Qts;   // Collector substrate depletion charge
`endif

// Small signal equivalent circuit conductances and resistances

gx     = - ddx(In, V(e1));      // Forward transconductance
gy     = - ddx(In, V(c2));      // Reverse transconductance

gz     = - ddx(In, V(c1));      // Reverse transconductance

sgpi   = - ddx(Ib1_s, V(e1));   // Conductance sidewall b-e junction
gpix   = - ddx(Ib1+Ib2, V(e1)); // Conductance floor b-e junction

gpiy   = - ddx(Ib1, V(c2));  // Early effect on recombination base current
gpiz   = - ddx(Ib1, V(c1));  // Early effect on recombination base current

gmux   =  ddx( Iavl, V(e1)); // Early effect on avalanche current limiting
gmuy   =  ddx( Iavl, V(c2)); // Conductance of avalanche current
gmuz   =  - ddx(- Iavl, V(c1));   // Conductance of avalanche current

// Conductance extrinsic b-c current :
gmuex  =   ddx(Iex+Ib3, V(b1))
              + ddx(Iex+Ib3, V(b2))
              + ddx(Iex+Ib3, V(c2));

xgmuex = ddx(XIex, V(b)); // Conductance extrinsic b-c current

grcvy  =  - ddx(Ic1c2, V(c2)); // Conductance of epilayer current
grcvz  =  - ddx(Ic1c2, V(c1));  // Conductance of epilayer current

rb_v   = 1.0 / (- ddx(Ib1b2, V(b2)) - ddx(Ib1b2, V(c2)));  // Variable base resistance

grbvx  = - ddx(Ib1b2, V(e1)); // Early effect on variable base resistance
grbvy  = - ddx(Ib1b2, V(c2)); // Early effect on variable base resistance

grbvz  = - ddx(Ib1b2, V(c1)); // Early effect on variable base resistance

r_e    = re_t;  // Emitter resistance
rb_c   = rbc_t; // Constant base resistance
rc_c   = rcc_xx_t; // Collector Contact resistance
rc_blx = rcc_ex_t; // Extrinsic buried layer resistance
rc_bli = rcc_in_t; // Extrinsic buried layer resistance


`ifdef SUBSTRATE
    gs     = ddx(Isub, V(b1));  // Conductance parasitic PNP transistor
    xgs    = ddx(XIsub, V(b)); // Conductance parasitic PNP transistor
    gsf    = ddx(Isf, V(s));   // Conductance substrate-collector current
`endif



// Small signal equivalent circuit capacitances
scbe   = - ddx(Qte_s, V(e1)); // Capacitance sidewall b-e junction

cbex   = - ddx(Qte + Qbe + Qe, V(e1));           // Capacitance floor b-e junction

cbey   = - ddx(Qbe, V(c2)); // Early effect on b-e diffusion junction

cbez   = - ddx(Qbe, V(c1)); // Early effect on b-e diffusion junction

cbcx   = - ddx(Qbc, V(e1)); // Early effect on b-c diffusion junction


cbcy   =  - ddx(Qtc + Qbc + Qepi, V(c2)); // Capacitance floor b-c junction
cbcz   =  - ddx(Qtc + Qbc + Qepi, V(c1)); // Capacitance floor b-c junction

// Capacitance extrinsic b-c junction :
cbcex  =   ddx(Qtex + Qex,V(b1))
              + ddx(Qtex + Qex,V(b2))
              + ddx(Qtex + Qex,V(c2));

// Capacitance extrinsic b-c junction :
xcbcex =  ddx(XQtex + XQex, V(b));

cb1b2  = - ddx(Qb1b2, V(b2)) - ddx(Qb1b2, V(c2));  // Capacitance AC current crowding

cb1b2x =  - ddx(Qb1b2, V(e1));  // Cross-capacitance AC current crowding
cb1b2y =  - ddx(Qb1b2, V(c2));  // Cross-capacitance AC current crowding
cb1b2z =  - ddx(Qb1b2, V(c1));  // Cross-capacitance AC current crowding

`ifdef SUBSTRATE
    cts    = ddx(Qts, V(s)); // Capacitance s-c junction
`endif

// Approximate small signal equivalent circuit
dydx = (gx - gmux) / (grcvy + gmuy - gy);
dydz = (gz - grcvz - gmuz) / (grcvy + gmuy - gy);
gpi  = sgpi + gpix + gmux + gpiz + gmuz +
         (gpiy + gmuy) * (dydx + dydz);
gm   = (grcvy * (gx - gmux +                   // Transconductance
           gz - gmuz) - grcvz *
           (gy - gmuy)) / (grcvy + gmuy - gy);
beta = gm / gpi;                                     // Current amplification
gout = ((gy - gmuy) * grcvz -                  // Output conductance
           (gz - gmuz) * grcvy) /
           (grcvy + gmuy - gy);
gmu  = gpiz + gmuz + (gpiy + gmuy) * dydz + // Feedback transconductance
          gmuex + xgmuex;
rb   = rbc_t + rb_v;                        // Base resistance
rc   = rc_c + rc_blx + rc_bli;           // Collector resistance
cbe  = cbex + scbe + cbcx +                    // Base-emitter capacitance
          (cbey + cbcy) * dydx + cbeo;
cbc  = (cbey + cbcy) * dydz + cbcz +           // Base-collector capacitance
           cbcex + xcbcex + cbco;


// Quantities to describe internal state of the model
gammax = (gpix + gmux - grbvx) * rb_v;
gammay = (gpiy + gmuy - grbvy) * rb_v;
gammaz = (gpiz + gmuz - grbvz) * rb_v;
gbfx   =  gpix + sgpi * (1.0 + gammax);
gbfy   =  gpiy + sgpi * gammay;
gbfz   =  gpiz + sgpi * gammaz;

// RvdT March 2008:
alpha_ft  = (1.0 + (grcvy * dydx * rc) +
             (gx + gbfx + (gy + gbfy) * dydx) * re_t)/
            (1.0 - (grcvz + grcvy * dydz) * rc -
             (gz + gbfz + (gy + gbfy) * dydz) * re_t);

rx = pow((grcvy * dydx + alpha_ft * (grcvz + grcvy * dydz)), -1);
rz = alpha_ft * rx;
ry = (1.0 - grcvz * rz) / grcvy;
rb1b2 = gammax * rx + gammay * ry + gammaz * rz;
rex = rz + rb1b2 - rc_bli;
xrex = rz + rb1b2 + rbc_t * ((gbfx + gmux) * rx + (gbfy + gmuy) * ry +
       (gbfz + gmuz) * rz) - rc_bli - rc_blx;

taut = scbe * (rx + rb1b2) + (cbex + cbcx) * rx + (cbey + cbcy) *
       ry + (cbez + cbcz) * rz + cbcex * rex + xcbcex * xrex +
       (cbeo + cbco) * (xrex - rcc_xx_t);

ft = 1.0 / (2.0 * `PI * taut); // Good approximation for cut-off frequency
iqs = Iqs;                     // Current at onset of quasi-saturation
xiwepi = xi_w;                 // Thickness of injection layer
vb2c2star = Vb2c2star;         // Physical value of internal base-collector bias

//self-heating
`ifdef SELFHEATING
    pdiss = power;                 // Dissipation
`endif

tk = Tk;                       // Actual temperature

if (mult != 1.0) begin
    ic = ic * mult;
    ib = ib * mult;
    ie = ie * mult;
    `ifdef SUBSTRATE
        isx = isx * mult;
    `endif
    in = in * mult;
    ic1c2 = ic1c2 * mult;
    ib1b2 = ib1b2 * mult;
    ib1 = ib1 * mult;
    ib1s = ib1s * mult;
    ib2s = ib2s * mult;
    ibrel = ibrel * mult;
    izteb = izteb * mult;
    iztcb = iztcb * mult;
    ib2 = ib2 * mult;
    ib3 = ib3 * mult;
    iavl = iavl * mult;
    iex = iex * mult;
    xiex = xiex * mult;
    `ifdef SUBSTRATE
        isub = isub * mult;
        xisub = xisub * mult;
        isf = isf * mult;
    `endif
    ire = ire * mult;
    irbc = irbc * mult;
    ircblx = ircblx * mult;
    ircbli = ircbli * mult;
    ircc = ircc * mult;
    qe = qe * mult;
    qte = qte * mult;
    sqte = sqte * mult;
    qbe = qbe * mult;
    qbc = qbc * mult;
    qtc = qtc * mult;
    qepi = qepi * mult;
    qb1b2 = qb1b2 * mult;
    qtex = qtex * mult;
    xqtex = xqtex * mult;
    qex = qex * mult;
    xqex = xqex * mult;
    `ifdef SUBSTRATE
        qts = qts * mult;
    `endif
    gx = gx * mult;
    gy = gy * mult;
    gz = gz * mult;
    sgpi = sgpi * mult;
    gpix = gpix * mult;
    gpiy = gpiy * mult;
    gpiz = gpiz * mult;
    gmux = gmux * mult;
    gmuy = gmuy * mult;
    gmuz = gmuz * mult;
    gmuex = gmuex * mult;
    xgmuex = xgmuex * mult;
    grcvy = grcvy * mult;
    grcvz = grcvz * mult;
    rb_v = rb_v / mult;
    grbvx = grbvx * mult;
    grbvy = grbvy * mult;
    grbvz = grbvz * mult;
    r_e = r_e / mult;
    rb_c = rb_c / mult;
    rc_c = rc_c / mult;
    rc_blx = rc_blx / mult;
    rc_bli = rc_bli / mult;
    `ifdef SUBSTRATE
        gs = gs * mult;
        xgs = xgs * mult;
        gsf = gsf * mult;
    `endif
    scbe = scbe * mult;
    cbex = cbex * mult;
    cbey = cbey * mult;
    cbez = cbez * mult;
    cbcx = cbcx * mult;
    cbcy = cbcy * mult;
    cbcz = cbcz * mult;
    cbcex = cbcex * mult;
    xcbcex = xcbcex * mult;
    cb1b2 = cb1b2 * mult;
    cb1b2x = cb1b2x * mult;
    cb1b2y = cb1b2y * mult;
    cb1b2z = cb1b2z * mult;
    `ifdef SUBSTRATE
        cts = cts * mult;
    `endif
    gm = gm * mult;
    gout = gout * mult;
    gmu = gmu * mult;
    rb = rb / mult;
    rc = rc / mult;
    cbe = cbe * mult;
    cbc = cbc * mult;
    iqs = iqs * mult;
    `ifdef SELFHEATING
        pdiss = pdiss * mult;
    `endif
end
