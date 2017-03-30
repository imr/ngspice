
#ifndef PARAMR
#define PARAMR(code,id,flag,ref,default,descrip)
#endif
#ifndef PARAMA
#define PARAMA(code,id,flag,ref,default,descrip) PARAM(code,id,flag,ref,default,descrip)
#endif
#ifndef JFET2_MOD_VTO
#define JFET2_MOD_VTO 141
#define JFET2_MOD_NJF 102
#define JFET2_MOD_PJF 103
#define JFET2_MOD_TNOM 104
#define JFET2_MOD_PB 131
#endif
    PARAM("acgam", 107,         JFET2acgamGiven, JFET2acgam,    0, "")
    PARAM("af",    108,         JFET2fNexpGiven, JFET2fNexp,    1, "Flicker Noise Exponent")
    PARAM("beta",  109,          JFET2betaGiven, JFET2beta,  1e-4, "Transconductance parameter")
   PARAMA("cds",   146,         JFET2capDSGiven, JFET2capds,    0, "D-S junction capacitance")
   PARAMA("cgd",   110,         JFET2capGDGiven, JFET2capgd,    0, "G-D junction capacitance")
   PARAMA("cgs",   111,         JFET2capGSGiven, JFET2capgs,    0, "G-S junction capacitance")
    PARAM("delta", 113,         JFET2deltaGiven, JFET2delta,    0, "coef of thermal current reduction")
    PARAM("hfeta", 114,         JFET2hfetaGiven, JFET2hfeta,    0, "drain feedback modulation")
    PARAM("hfe1",  115,          JFET2hfe1Given, JFET2hfe1,     0, "")
    PARAM("hfe2",  116,          JFET2hfe2Given, JFET2hfe2,     0, "")
    PARAM("hfg1",  117,          JFET2hfg1Given, JFET2hfg1,     0, "")
    PARAM("hfg2",  118,          JFET2hfg2Given, JFET2hfg2,     0, "")
    PARAM("mvst",  119,          JFET2mvstGiven, JFET2mvst,     0, "modulation index for subtreshold current")
    PARAM("mxi",   120,           JFET2mxiGiven, JFET2mxi,      0, "saturation potential modulation parameter")
    PARAM("fc",    121,            JFET2fcGiven, JFET2fc,     0.5, "Forward bias junction fit parm.")
    PARAM("ibd",   122,           JFET2ibdGiven, JFET2ibd,      0, "Breakdown current of diode jnc")
    PARAM("is",    123,            JFET2isGiven, JFET2is,   1e-14, "Gate junction saturation current")
    PARAM("kf",    124,            JFET2kfGiven, JFET2fNcoef,   0, "Flicker Noise Coefficient")
    PARAM("lambda",125,           JFET2lamGiven, JFET2lambda,   0, "Channel length modulation param.")
    PARAM("lfgam", 126,         JFET2lfgamGiven, JFET2lfgam,    0, "drain feedback parameter")
    PARAM("lfg1",  127,          JFET2lfg1Given, JFET2lfg1,     0, "")
    PARAM("lfg2",  128,          JFET2lfg2Given, JFET2lfg2,     0, "")
    PARAM("n",     129,             JFET2nGiven, JFET2n,        1, "gate junction ideality factor")
    PARAM("p",     130,             JFET2pGiven, JFET2p,        2, "Power law (triode region)")
    PARAM("vbi",   JFET2_MOD_PB,   JFET2phiGiven, JFET2phi,     1, "Gate junction potential")
    PARAMR("pb",   JFET2_MOD_PB,   JFET2phiGiven, JFET2phi,     1, "Gate junction potential")
    PARAM("q",     132,             JFET2qGiven, JFET2q,        2, "Power Law (Saturated region)")
    PARAM("rd",    133,            JFET2rdGiven, JFET2rd,       0, "Drain ohmic resistance")
    PARAM("rs",    134,            JFET2rsGiven, JFET2rs,       0, "Source ohmic resistance")
    PARAM("taud",  135,          JFET2taudGiven, JFET2taud,     0, "Thermal relaxation time")
    PARAM("taug",  136,          JFET2taugGiven, JFET2taug,     0, "Drain feedback relaxation time")
    PARAM("vbd",   137,           JFET2vbdGiven, JFET2vbd,      1, "Breakdown potential of diode jnc")
    PARAM("ver",   139,           JFET2verGiven, JFET2ver,      0, "version number of PS model")
    PARAM("vst",   140,           JFET2vstGiven, JFET2vst,      0, "Crit Poten subthreshold conductn")
    PARAM("vt0",   JFET2_MOD_VTO,  JFET2vtoGiven, JFET2vto,    -2, "Threshold voltage")
    PARAMR("vto",  JFET2_MOD_VTO,  JFET2vtoGiven, JFET2vto,    -2, "Threshold voltage")
    PARAM("xc",    142,            JFET2xcGiven, JFET2xc,       0, "amount of cap. red at pinch-off")
    PARAM("xi",    143,            JFET2xiGiven, JFET2xi,    1000, "velocity saturation index")
    PARAM("z",     144,             JFET2zGiven, JFET2z,        1, "rate of velocity saturation")
    PARAM("hfgam", 145,           JFET2hfgGiven, JFET2hfgam, model->JFET2lfgam, "high freq drain feedback parm")
#undef   PARAM
#undef  PARAMR
#undef  PARAMA

