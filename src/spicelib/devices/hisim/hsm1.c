/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1.c of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "devdefs.h"
#include "hsm1def.h"
#include "suffix.h"

IFparm HSM1pTable[] = { /* parameters */
 IOP( "l",   HSM1_L,      IF_REAL   , "Length"),
 IOP( "w",   HSM1_W,      IF_REAL   , "Width"),
 IOP( "ad",  HSM1_AD,     IF_REAL   , "Drain area"),
 IOP( "as",  HSM1_AS,     IF_REAL   , "Source area"),
 IOP( "pd",  HSM1_PD,     IF_REAL   , "Drain perimeter"),
 IOP( "ps",  HSM1_PS,     IF_REAL   , "Source perimeter"),
 IOP( "nrd", HSM1_NRD,    IF_REAL   , "Number of squares in drain"),
 IOP( "nrs", HSM1_NRS,    IF_REAL   , "Number of squares in source"),
 IOP( "temp", HSM1_TEMP,  IF_REAL   , "Lattice temperature"),
 IOP( "dtemp", HSM1_DTEMP,IF_REAL   , ""),
 IOP( "off", HSM1_OFF,    IF_FLAG   , "Device is initially off"), 
 IP ( "ic",  HSM1_IC,     IF_REALVEC , "Vector of DS,GS,BS initial voltages"),
 IOP( "m",   HSM1_M,      IF_REAL   , "Parallel multiplier")
};

IFparm HSM1mPTable[] = { /* model parameters */
  IP("nmos", HSM1_MOD_NMOS, IF_FLAG, ""),
  IP("pmos", HSM1_MOD_PMOS, IF_FLAG, ""),
  IOP("level", HSM1_MOD_LEVEL, IF_INTEGER, ""),
  IOP("info", HSM1_MOD_INFO, IF_INTEGER, "information level (for debug, etc.)"),
  IOP("noise", HSM1_MOD_NOISE, IF_INTEGER, "noise model selector"),
  IOP("version", HSM1_MOD_VERSION, IF_INTEGER, "model version 102 or 112 or 120"),
  IOP("show", HSM1_MOD_SHOW, IF_INTEGER, "show physical value"),
  IOP("corsrd", HSM1_MOD_CORSRD, IF_INTEGER, "solve equations accounting Rs and Rd."),
  IOP("coiprv", HSM1_MOD_COIPRV, IF_INTEGER, "use ids_prv as initial guess of Ids"),
  IOP("copprv", HSM1_MOD_COPPRV, IF_INTEGER, "use ps{0/l}_prv as initial guess of Ps{0/l}"),
  IOP("cocgso", HSM1_MOD_COCGSO, IF_INTEGER, "calculate cgso"),
  IOP("cocgdo", HSM1_MOD_COCGDO, IF_INTEGER, "calculate cgdo"),
  IOP("cocgbo", HSM1_MOD_COCGBO, IF_INTEGER, "calculate cgbo"),
  IOP("coadov", HSM1_MOD_COADOV, IF_INTEGER, "add overlap to intrisic"),
  IOP("coxx08", HSM1_MOD_COXX08, IF_INTEGER, "spare"),
  IOP("coxx09", HSM1_MOD_COXX09, IF_INTEGER, "spare"),
  IOP("coisub", HSM1_MOD_COISUB, IF_INTEGER, "calculate isub"),
  IOP("coiigs", HSM1_MOD_COIIGS, IF_INTEGER, "calculate igate"),
  IOP("cogidl", HSM1_MOD_COGIDL, IF_INTEGER, "calculate igidl"),
  IOP("cogisl", HSM1_MOD_COGISL, IF_INTEGER, "calculate igisl"),
  IOP("coovlp", HSM1_MOD_COOVLP, IF_INTEGER, "calculate overlap charge"),
  IOP("conois", HSM1_MOD_CONOIS, IF_INTEGER, "calculate 1/f noise"),
  IOP("coisti", HSM1_MOD_COISTI, IF_INTEGER, "calculate STI HiSIM1.1"),
  IOP("cosmbi", HSM1_MOD_COSMBI, IF_INTEGER, "biases smoothing in dvth HiSIM1.2"),
  IOP("vmax", HSM1_MOD_VMAX, IF_REAL, "saturation velocity [cm/s"),
  IOP("bgtmp1", HSM1_MOD_BGTMP1, IF_REAL, "first order temp. coeff. for band gap [V/K]"),
  IOP("bgtmp2", HSM1_MOD_BGTMP2, IF_REAL, "second order temp. coeff. for band gap [V/K^2]"),
  IOP("tox", HSM1_MOD_TOX, IF_REAL, "oxide thickness [m]"),
  IOP("xld", HSM1_MOD_XLD, IF_REAL, "lateral diffusion of S/D under the gate [m]"),
  IOP("xwd", HSM1_MOD_XWD, IF_REAL, "lateral diffusion along the width dir. [m]"),
  IOP("xj", HSM1_MOD_XJ, IF_REAL, "HiSIM1.0.z [m]"),
  IOP("xqy", HSM1_MOD_XQY, IF_REAL, "HiSIM1.1.z or later [m]"),
  IOP("rs", HSM1_MOD_RS, IF_REAL, "source contact resistance [ohm m]"),
  IOP("rd", HSM1_MOD_RD, IF_REAL, "drain contact resistance  [ohm m]"),
  IOP("vfbc", HSM1_MOD_VFBC, IF_REAL, "constant part of Vfb [V]"),
  IOP("nsubc", HSM1_MOD_NSUBC, IF_REAL, "constant part of Nsub [1/cm^3]"),
  IOP("parl1", HSM1_MOD_PARL1, IF_REAL, "factor for L dependency of dVthSC [-]"),
  IOP("parl2", HSM1_MOD_PARL2, IF_REAL, "under diffusion [m]"),
  IOP("lp", HSM1_MOD_LP, IF_REAL, "length of pocket potential [m]"),
  IOP("nsubp", HSM1_MOD_NSUBP, IF_REAL, "[1/cm^3]"),
  IOP("scp1", HSM1_MOD_SCP1, IF_REAL, "parameter for pocket [-]"),
  IOP("scp2", HSM1_MOD_SCP2, IF_REAL, "parameter for pocket [1/V]"),
  IOP("scp3", HSM1_MOD_SCP3, IF_REAL, "parameter for pocket [m/V]"),
  IOP("sc1", HSM1_MOD_SC1, IF_REAL, "parameter for SCE [-]"),
  IOP("sc2", HSM1_MOD_SC2, IF_REAL, "parameter for SCE [1/V]"),
  IOP("sc3", HSM1_MOD_SC3, IF_REAL, "parameter for SCE [m/V]"),
  IOP("pgd1", HSM1_MOD_PGD1, IF_REAL, "parameter for gate-poly depletion [V]"),
  IOP("pgd2", HSM1_MOD_PGD2, IF_REAL, "parameter for gate-poly depletion [V]"),
  IOP("pgd3", HSM1_MOD_PGD3, IF_REAL, "parameter for gate-poly depletion [-]"),
  IOP("ndep", HSM1_MOD_NDEP, IF_REAL, "coeff. of Qbm for Eeff [-]"),
  IOP("ninv", HSM1_MOD_NINV, IF_REAL, "coeff. of Qnm for Eeff [-]"),
  IOP("ninvd", HSM1_MOD_NINVD, IF_REAL, "parameter for universal mobility [1/V]"),
  IOP("muecb0", HSM1_MOD_MUECB0, IF_REAL, "const. part of coulomb scattering [cm^2/Vs]"),
  IOP("muecb1", HSM1_MOD_MUECB1, IF_REAL, "coeff. for coulomb scattering [cm^2/Vs]"),
  IOP("mueph0", HSM1_MOD_MUEPH0, IF_REAL, "power of Eeff for phonon scattering [-]"),
  IOP("mueph1", HSM1_MOD_MUEPH1, IF_REAL, ""),
  IOP("mueph2", HSM1_MOD_MUEPH2, IF_REAL, ""),
  IOP("w0", HSM1_MOD_W0, IF_REAL, ""),
  IOP("muesr0", HSM1_MOD_MUESR0, IF_REAL, "power of Eeff for S.R. scattering [-]"),
  IOP("muesr1", HSM1_MOD_MUESR1, IF_REAL, "coeff. for S.R. scattering [-]"),
  IOP("muetmp", HSM1_MOD_MUETMP, IF_REAL, "parameter for mobility [-]"),
  IOP("bb", HSM1_MOD_BB, IF_REAL, "empirical mobility model coefficient [-]"),
  IOP("sub1", HSM1_MOD_SUB1, IF_REAL, "parameter for Isub [1/V]"),
  IOP("sub2", HSM1_MOD_SUB2, IF_REAL, "parameter for Isub [V]"),
  IOP("sub3", HSM1_MOD_SUB3, IF_REAL, "parameter for Isub [-]"),
  IOP("wvthsc", HSM1_MOD_WVTHSC, IF_REAL, "parameter for STI [-] HiSIM1.1"),
  IOP("nsti", HSM1_MOD_NSTI, IF_REAL, "parameter for STI [1/cm^3] HiSIM1.1"),
  IOP("wsti", HSM1_MOD_WSTI, IF_REAL, "parameter for STI [m] HiSIM1.1"),
  IOP("cgso", HSM1_MOD_CGSO, IF_REAL, "G-S overlap capacitance per unit W [F/m]"),
  IOP("cgdo", HSM1_MOD_CGDO, IF_REAL, "G-D overlap capacitance per unit W [F/m]"),
  IOP("cgbo", HSM1_MOD_CGBO, IF_REAL, "G-B overlap capacitance per unit L [F/m]"),
  IOP("tpoly", HSM1_MOD_TPOLY, IF_REAL, "hight of poly gate [m]"),
  IOP("js0", HSM1_MOD_JS0, IF_REAL, "Saturation current density [A/m^2]"),
  IOP("js0sw", HSM1_MOD_JS0SW, IF_REAL, "Side wall saturation current density [A/m]"),
  IOP("nj", HSM1_MOD_NJ, IF_REAL, "Emission coefficient"),
  IOP("njsw", HSM1_MOD_NJSW, IF_REAL, "Sidewall emission coefficient"),
  IOP("xti", HSM1_MOD_XTI, IF_REAL, "Junction current temparature exponent coefficient"),
  IOP("cj", HSM1_MOD_CJ, IF_REAL, "Bottom junction capacitance per unit area at zero bias [F/m^2]"),
  IOP("cjsw", HSM1_MOD_CJSW, IF_REAL, "Source/drain sidewall junction capacitance grading coefficient per unit length at zero bias [F/m]"),
  IOP("cjswg", HSM1_MOD_CJSWG, IF_REAL, "Source/drain gate sidewall junction capacitance per unit length at zero bias [F/m]"),
  IOP("mj", HSM1_MOD_MJ, IF_REAL, "Bottom junction capacitance grading coefficient"),
  IOP("mjsw", HSM1_MOD_MJSW, IF_REAL, "Source/drain sidewall junction capacitance grading coefficient"),
  IOP("mjswg", HSM1_MOD_MJSWG, IF_REAL, "Source/drain gate sidewall junction capacitance grading coefficient"),
  IOP("pb", HSM1_MOD_PB, IF_REAL, "Bottom junction build-in potential  [V]"),
  IOP("pbsw", HSM1_MOD_PBSW, IF_REAL, "Source/drain sidewall junction build-in potential [V]"),
  IOP("pbswg", HSM1_MOD_PBSWG, IF_REAL, "Source/drain gate sidewall junction build-in potential [V]"),
  IOP("xpolyd", HSM1_MOD_XPOLYD, IF_REAL, "parameter for Cov [m]"),
  IOP("clm1", HSM1_MOD_CLM1, IF_REAL, "parameter for CLM [-]"),
  IOP("clm2", HSM1_MOD_CLM2, IF_REAL, "parameter for CLM [1/m]"),
  IOP("clm3", HSM1_MOD_CLM3, IF_REAL, "parameter for CLM [-]"),
  IOP("rpock1", HSM1_MOD_RPOCK1, IF_REAL, "parameter for Ids [V]"),
  IOP("rpock2", HSM1_MOD_RPOCK2, IF_REAL, "parameter for Ids [V^2 sqrt(m)/A]"),
  IOP("rpocp1", HSM1_MOD_RPOCP1, IF_REAL, "parameter for Ids [-] HiSIM1.1"),
  IOP("rpocp2", HSM1_MOD_RPOCP2, IF_REAL, "parameter for Ids [-] HiSIM1.1"),
  IOP("vover", HSM1_MOD_VOVER, IF_REAL, "parameter for overshoot [m^{voverp}]"),
  IOP("voverp", HSM1_MOD_VOVERP, IF_REAL, "parameter for overshoot [-]"),
  IOP("wfc", HSM1_MOD_WFC, IF_REAL, "parameter for narrow channel effect [m*F/(cm^2)]"),
  IOP("qme1", HSM1_MOD_QME1, IF_REAL, "parameter for quantum effect [mV]"),
  IOP("qme2", HSM1_MOD_QME2, IF_REAL, "parameter for quantum effect [V]"),
  IOP("qme3", HSM1_MOD_QME3, IF_REAL, "parameter for quantum effect [m]"),
  IOP("gidl1", HSM1_MOD_GIDL1, IF_REAL, "parameter for GIDL [?]"),
  IOP("gidl2", HSM1_MOD_GIDL2, IF_REAL, "parameter for GIDL [?]"),
  IOP("gidl3", HSM1_MOD_GIDL3, IF_REAL, "parameter for GIDL [?]"),
  IOP("gleak1", HSM1_MOD_GLEAK1, IF_REAL, "parameter for gate current [?]"),
  IOP("gleak2", HSM1_MOD_GLEAK2, IF_REAL, "parameter for gate current [?]"),
  IOP("gleak3", HSM1_MOD_GLEAK3, IF_REAL, "parameter for gate current [?]"),
  IOP("vzadd0", HSM1_MOD_VZADD0, IF_REAL, "Vzadd at Vds=0  [V]"),
  IOP("pzadd0", HSM1_MOD_PZADD0, IF_REAL, "Pzadd at Vds=0  [V]"),
  IOP("nftrp", HSM1_MOD_NFTRP, IF_REAL, ""),
  IOP("nfalp", HSM1_MOD_NFALP, IF_REAL, ""),
  IOP("cit", HSM1_MOD_CIT, IF_REAL, ""),
  IOP("glpart1", HSM1_MOD_GLPART1, IF_REAL, "partitoning of gate current HiSIM1.2"),
  IOP("glpart2", HSM1_MOD_GLPART2, IF_REAL, "partitoning of gate current HiSIM1.2"),
  IOP("kappa", HSM1_MOD_KAPPA, IF_REAL, "HiSIM1.2 dielectric constant for high-k stacked gate"),
  IOP("xdiffd", HSM1_MOD_XDIFFD, IF_REAL, "HiSIM1.2 parameter for W_design [m]"),
  IOP("pthrou", HSM1_MOD_PTHROU, IF_REAL, "HiSIM1.2 modify subthreshold sloop [-]"),
  IOP("vdiffj", HSM1_MOD_VDIFFJ, IF_REAL, "HiSIM1.2 threshold voltage for S/D junction diode [V]"),
  IOP( "ef", HSM1_MOD_EF, IF_REAL, "flicker noise frequency exponent"),
  IOP( "af", HSM1_MOD_AF, IF_REAL, "flicker noise exponent"),
  IOP( "kf", HSM1_MOD_KF, IF_REAL, "flicker noise coefficient")
};

char *HSM1names[] = {
    "Drain",
    "Gate",
    "Source",
    "Bulk"
};

int	HSM1nSize = NUMELEMS(HSM1names);
int	HSM1pTSize = NUMELEMS(HSM1pTable);
int	HSM1mPTSize = NUMELEMS(HSM1mPTable);
int	HSM1iSize = sizeof(HSM1instance);
int	HSM1mSize = sizeof(HSM1model);

