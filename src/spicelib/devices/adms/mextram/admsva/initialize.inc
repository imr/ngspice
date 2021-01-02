// Copyright (c) 2000-2007, NXP Semiconductor
// Copyright (c) 2007-2014, Delft University of Technology
// Copyright (c) 2015, Auburn University
// All rights reserved, see IP_NOTICE_DISCLAIMER_LICENSE for further information.

if (`PGIVEN(GMIN))
    my_gmin = GMIN;
else
    my_gmin = $simparam("gmin");

// Initialize model constants
`INITIAL_INSTANCE
begin
    // Impact ionization constants (NPN - PNP)

    if (TYPE == 1) begin

        An = 7.03e7;
        Bn = 1.23e8;

    end else begin

        An = 1.58e8;
        Bn = 2.04e8;

    end

    Xext1 = 1.0 - XEXT;

    // Temperature independent  MULT scaling

`ifdef SELFHEATING
    CTH_M  = CTH   * MULT;
`endif

    CBEO_M = CBEO  * MULT;
    CBCO_M = CBCO  * MULT;

    invMULT    = 1.0 / MULT;
    SCRCV_M    = SCRCV    * invMULT;

    KF_M     = KF  * pow(MULT, 1.0 - AF);
    KFN_M    = KFN * pow(MULT, 1.0 - (2.0 * (MLF - 1.0) + AF * (2.0 - MLF)));

    // begin: RvdT, November 2008; Zener tunneling current model

    pow2_2mPE = pow(2.0, 2.0 - PE);
    pow2_PEm2 = 1.0 / pow2_2mPE;

    // Reference Temperature expressed in Kelvin:
    Trk = TREF + `C2K;
    // Ambient Temperature expressed in Kelvin:
    Tamb = $temperature + DTA;


    // begin: RvdT, November 2008; Zener tunneling current model
    //
    // Comment added March 2009: this assumes VGZEBOK as a model parameter.
    //
    // Bandgap for Zener tunnel current model at reference temperature in eV:
    //   VGZEB_Tr = VGZEBOK - AVGEB*Trk*Trk / (Trk + TVGEB);
    //    `max_logexp(VGZEB_Tr, VGZEBOK - AVGEB*Trk*Trk / (Trk + TVGEB), 0.05, 0.1);
    // end: RvdT, November 2008

    // begin: RvdT March 2009:
    // to decrease parameter interdependency,
    // use VGZEB as a parameter, instead of VGZEBOK:
    //   VGZEB   : bandgap for Zener tunneling at T = Tref,
    //   VGZEBOK : bandgap for Zener tunneling at T = 0 K.
    //`max_logexp(VGZEBOK,  VGZEB   + AVGEB*Trk*Trk / (Trk + TVGEB), 0.05, 0.1);
    //dw admsXml can't expand the macro `max_logexp here - using the code
    _x = VGZEB + AVGEB*Trk*Trk / (Trk + TVGEB);
    _x0 = 0.05;
    _a = 0.1;
    _dxa = (_x - _x0) / (_a);
    if (_x < _x0)
        VGZEBOK = _x0 + _a * ln(1.0 + exp(_dxa));
    else
        VGZEBOK = _x  + _a * ln(1.0 + exp(-_dxa));

    VGZEB_Tr = VGZEB;
    // end: RvdT March 2009: use VGZEB as a parameter, instead of VGZEBOK:

    inv_VGZEB_Tr = 1.0 / VGZEB_Tr;

    inv_VDE = 1.0 / VDE;

    // end: RvdT, November 2008; Zener tunneling current model
end
