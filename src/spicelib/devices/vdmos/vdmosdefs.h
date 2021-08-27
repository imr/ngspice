/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
VDMOS: 2018 Holger Vogt, 2020 Dietmar Warning
**********/

#ifndef VDMOS
#define VDMOS

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* declarations for VDMOSFETs */

/* indices to the array of MOSFET(1) noise sources */

enum {
    VDMOSRDNOIZ = 0,
    VDMOSRSNOIZ,
    VDMOSIDNOIZ,
    VDMOSFLNOIZ,
    VDMOSTOTNOIZ,
    /* finally, the number of noise sources */
    VDMOSNSRCS
};

/* information needed for each instance */

typedef struct sVDMOSinstance {

    struct GENinstance gen;

#define VDMOSmodPtr(inst) ((struct sVDMOSmodel *)((inst)->gen.GENmodPtr))
#define VDMOSnextInstance(inst) ((struct sVDMOSinstance *)((inst)->gen.GENnextInstance))
#define VDMOSname gen.GENname
#define VDMOSstates gen.GENstate

    const int VDMOSdNode;  /* number of the drain node of the mosfet */
    const int VDMOSgNode;  /* number of the gate node of the mosfet */
    const int VDMOSsNode;  /* number of the source node of the mosfet */
    int VDMOStempNode;  /* number of the temperature node of the mosfet */
    int VDMOStcaseNode;  /* number of the 2nd temperature node of the mosfet */
    int VDMOSdNodePrime; /* number of the internal drain node of the mosfet */
    int VDMOSsNodePrime; /* number of the internal source node of the mosfet */
    int VDMOSgNodePrime; /* number of the internal gate node of the mosfet */
    int VDMOStNodePrime; /* number of the internal temp node between voltage source and Rthca */
    int VDIOposPrimeNode; /* number of the internal node of the body diode */

    int VDMOSvcktTbranch; /* equation number of branch equation added for cktTemp source */ 

    double VDMOSm;   /* parallel device multiplier */

    double VDMOSsourceConductance;   /*conductance of source(or 0):set in setup*/
    double VDMOSdrainConductance;    /*conductance of drain(or 0):set in setup*/
    double VDMOSdrainResistance;    /*resistance of drain(or 0): set in temp*/
    double VDMOSqsResistance;    /*resistance of drain: set in temp*/
    double VDMOSgateConductance;    /*conductance of gate(or 0):set in setup*/
    double VDMOSdsConductance;    /*conductance of drain to source:set in setup*/
    double VDMOStemp;    /* operating temperature of this instance */
    double VDMOSdtemp;   /* operating temperature of the instance relative to circuit temperature*/
    int    VDMOSthermal;  /* flag indicate self heating on */

    double VDMOStTransconductance;   /* temperature corrected transconductance*/
    double VDMOStPhi;                /* temperature corrected Phi */
    double VDMOStVth;                /* temperature corrected Vth */
    double VDMOStksubthres;          /* temperature weak inversion slope */

    double VDMOSicVDS;   /* initial condition D-S voltage */
    double VDMOSicVGS;   /* initial condition G-S voltage */
    double VDMOSvon;
    double VDMOSvdsat;
    double VDMOScd;
    double VDMOSgm;
    double VDMOSgds;

    double VDIOcap;
    double VDIOtSatCur;     /* temperature corrected saturation Cur. density */
    double VDIOtSatCur_dT;
    double VDIOinitCond;
    double VDIOtVcrit;
    double VDIOconductance;
    double VDIOtConductance;
    double VDIOtConductance_dT;
    double VDIOtBrkdwnV;
    double VDIOtJctCap;
    double VDIOtDepCap;     /* temperature adjusted transition point in */
                            /* the curve matching Fc * Vj */
    double VDIOtJctPot;     /* temperature corrected junction potential */
    double VDIOtGradingCoeff;

    double VDIOtTransitTime;
    double VDIOtF1;
    double VDIOtF2;
    double VDIOtF3;

    double VDMOSTempSH;      /* for portability of SH temp to noise analysis */

    double VDMOSgmT;
    double VDMOSgtempg;
    double VDMOSgtempd;
    double VDMOSgtempT;
    double VDMOScgT;
    double VDMOScdT;
    double VDMOScth;         /* current alias power */

/*
 * naming convention:
 * x = vgs
 * y = vbs
 * z = vds
 * cdr = cdrain
 */

#define VDMOSNDCOEFFS   11

#ifndef NODISTO
    double VDMOSdCoeffs[VDMOSNDCOEFFS];
#else /* NODISTO */
    double *VDMOSdCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define cdr_x2      VDMOSdCoeffs[0]
#define cdr_y2      VDMOSdCoeffs[1]
#define cdr_xy      VDMOSdCoeffs[2]
#define cdr_x3      VDMOSdCoeffs[3]
#define cdr_y3      VDMOSdCoeffs[4]
#define cdr_x2y     VDMOSdCoeffs[5]
#define cdr_xy2     VDMOSdCoeffs[6]
#define capgs2      VDMOSdCoeffs[7]
#define capgs3      VDMOSdCoeffs[8]
#define capgd2      VDMOSdCoeffs[9]
#define capgd3      VDMOSdCoeffs[10]

#endif

#ifndef NONOISE
    double VDMOSnVar[NSTATVARS][VDMOSNSRCS];
#else /* NONOISE */
    double **VDMOSnVar;
#endif /* NONOISE */

    int VDMOSmode;       /* device mode : 1 = normal, -1 = inverse */

    unsigned VDMOSoff:1;  /* non-zero to indicate device is off for dc analysis*/
    unsigned VDMOStempGiven :1;  /* instance temperature specified */
    unsigned VDMOSdtempGiven :1;  /* instance delta temperature specified */
    unsigned VDMOSmGiven :1;
    unsigned VDMOSdNodePrimeSet :1;
    unsigned VDMOSsNodePrimeSet :1;
    unsigned VDMOSicVDSGiven :1;
    unsigned VDMOSicVGSGiven :1;
    unsigned VDMOSvonGiven : 1;
    unsigned VDMOSvdsatGiven :1;
    unsigned VDMOSmodeGiven :1;

    double *VDMOSDdPtr;      /* pointer to sparse matrix element at
                                       (Drain node,drain node) */
    double *VDMOSGgPtr;      /* pointer to sparse matrix element at
                                       (gate node,gate node) */
    double *VDMOSSsPtr;      /* pointer to sparse matrix element at
                                       (source node,source node) */
    double *VDMOSDPdpPtr;    /* pointer to sparse matrix element at
                                       (drain prime node,drain prime node) */
    double *VDMOSSPspPtr;    /* pointer to sparse matrix element at
                                       (source prime node,source prime node) */
    double *VDMOSDdpPtr;     /* pointer to sparse matrix element at
                                       (drain node,drain prime node) */
    double *VDMOSGdpPtr;     /* pointer to sparse matrix element at
                                       (gate node,drain prime node) */
    double *VDMOSGspPtr;     /* pointer to sparse matrix element at
                                       (gate node,source prime node) */
    double *VDMOSSspPtr;     /* pointer to sparse matrix element at
                                       (source node,source prime node) */
    double *VDMOSDPspPtr;    /* pointer to sparse matrix element at
                                       (drain prime node,source prime node) */
    double *VDMOSDPdPtr;     /* pointer to sparse matrix element at
                                       (drain prime node,drain node) */
    double *VDMOSDPgPtr;     /* pointer to sparse matrix element at
                                       (drain prime node,gate node) */

    double *VDMOSSPgPtr;     /* pointer to sparse matrix element at
                                       (source prime node,gate node) */
    double *VDMOSSPsPtr;     /* pointer to sparse matrix element at
                                       (source prime node,source node) */
    double *VDMOSSPdpPtr;    /* pointer to sparse matrix element at
                                       (source prime node,drain prime node) */
    /* added for VDMOS */
    double *VDMOSGPgpPtr;    /* pointer to sparse matrix element at
                               (gate prime node, gate prime node) */
    double *VDMOSGPdpPtr;    /* pointer to sparse matrix element at
                               (gate prime node, drain prime node) */
    double *VDMOSGPspPtr;    /* pointer to sparse matrix element at
                               (gate prime node, source prime node) */
    double *VDMOSDPgpPtr;    /* pointer to sparse matrix element at
                               (drain prime node, gate prime node) */
    double *VDMOSSPgpPtr;    /* pointer to sparse matrix element at
                               (source prime node, gate prime node) */
    double *VDMOSGgpPtr;    /* pointer to sparse matrix element at
                               (gate node, gate prime node) */
    double *VDMOSGPgPtr;    /* pointer to sparse matrix element at
                               (gate prime node, gate node) */
    double *VDMOSDsPtr;    /* pointer to sparse matrix element at
                               (source node, drain node) */
    double *VDMOSSdPtr;    /* pointer to sparse matrix element at
                               (drain node, source node) */
    /* body diode */
    double *VDIORPdPtr;    /* pointer to sparse matrix element at
                               (diode prime node, drain node) */
    double *VDIODrpPtr;    /* pointer to sparse matrix element at
                               (drain node, diode prime node) */
    double *VDIORPrpPtr;    /* pointer to sparse matrix element at
                               (diode prime node, diode prime node) */
    double *VDIOSrpPtr;    /* pointer to sparse matrix element at
                              (source node, diode prime node) */
    double *VDIORPsPtr;    /* pointer to sparse matrix element at
                              (diode prime node, source node) */
    /* self heating */
    double *VDMOSTemptempPtr;   /* Transistor thermal contribution */
    double *VDMOSTempdpPtr;
    double *VDMOSTempspPtr;
    double *VDMOSTempgpPtr;
    double *VDMOSGPtempPtr;
    double *VDMOSDPtempPtr;
    double *VDMOSSPtempPtr;

    double *VDIOTempposPrimePtr; /* Diode thermal contribution */
    double *VDMOSTempdPtr;
    double *VDIOPosPrimetempPtr;
    double *VDMOSDtempPtr;
    double *VDMOStempSPtr;
    double *VDMOSSTempPtr;

    double *VDMOSTcasetcasePtr; /* for Rthjc */
    double *VDMOSTcasetempPtr;
    double *VDMOSTemptcasePtr;
    double *VDMOSTptpPtr;       /* for Rthca */
    double *VDMOSTptcasePtr;
    double *VDMOSTcasetpPtr;
    double *VDMOSCktTcktTPtr;   /* for VcktTemp */
    double *VDMOSCktTtpPtr;
    double *VDMOSTpcktTPtr;

} VDMOSinstance ;

#define VDMOSvgs VDMOSstates+ 0   /* gate-source voltage */
#define VDMOSvds VDMOSstates+ 1   /* drain-source voltage */
#define VDMOSdelTemp VDMOSstates+ 2 /* thermal voltage over rth0 */

#define VDMOScapgs VDMOSstates+3  /* gate-source capacitor value */
#define VDMOSqgs VDMOSstates+ 4   /* gate-source capacitor charge */
#define VDMOScqgs VDMOSstates+ 5  /* gate-source capacitor current */

#define VDMOScapgd VDMOSstates+ 6 /* gate-drain capacitor value */
#define VDMOSqgd VDMOSstates+ 7   /* gate-drain capacitor charge */
#define VDMOScqgd VDMOSstates+ 8  /* gate-drain capacitor current */

#define VDIOvoltage VDMOSstates+ 9
#define VDIOcurrent VDMOSstates+ 10
#define VDIOconduct VDMOSstates+ 11

#define VDIOcapCharge VDMOSstates+ 12
#define VDIOcapCurrent VDMOSstates+ 13

#define VDMOScapth VDMOSstates+ 14 /* thermal capacitor value */
#define VDMOSqth VDMOSstates+ 15   /* thermal capacitor charge */
#define VDMOScqth VDMOSstates+ 16  /* thermal capacitor current */

#define VDIOdIdio_dT VDMOSstates+ 17

#define VDMOSnumStates 18


/* per model data */

    /* NOTE:  parameters marked 'input - use xxxx' are parameters for
     * which a temperature correction is applied in VDMOStemp, thus
     * the VDMOSxxxx value in the per-instance structure should be used
     * instead in all calculations 
     */


typedef struct sVDMOSmodel {       /* model structure for a resistor */

    struct GENmodel gen;

#define VDMOSmodType gen.GENmodType
#define VDMOSnextModel(inst) ((struct sVDMOSmodel *)((inst)->gen.GENnextModel))
#define VDMOSinstances(inst) ((VDMOSinstance *)((inst)->gen.GENinstances))
#define VDMOSmodName gen.GENmodName

    int VDMOStype;       /* device type : 1 = nmos,  -1 = pmos */
    double VDMOStnom;        /* temperature at which parameters measured */
    double VDMOSdrainResistance;
    double VDMOSsourceResistance;
    double VDMOSgateResistance;
    double VDMOSqsResistance;
    double VDMOSqsVoltage;
    double VDMOStransconductance;    /* input - use tTransconductance */
    double VDMOSoxideCapFactor;
    double VDMOSvth0; /* input - use tVth */
    double VDMOSphi; /* input - use tPhi */
    double VDMOSlambda;
    double VDMOStheta;
    double VDMOSfNcoef;
    double VDMOSfNexp;

    double VDMOScgdmin;
    double VDMOScgdmax;
    double VDMOSa;
    double VDMOScgs;
    double VDMOSsubshift;
    double VDMOSksubthres;
    double VDMOSmtr;
    double VDMOSrds;

    /* body diode */
    double VDIOjunctionCap;   /* input - use tCj */
    double VDIOjunctionPot;    /* input - use tJctPot */
    double VDIOdepletionCapCoeff;
    double VDIOjctSatCur;   /* input - use tSatCur */
    double VDIObv;
    double VDIOibv;
    double VDIObrkdEmissionCoeff;
    double VDIOresistance;
    double VDIOn;
    double VDIOtransitTime;
    double VDIOtranTimeTemp1;
    double VDIOtranTimeTemp2;
    double VDIOeg;
    double VDIOxti;
    double VDIOgradCoeff;
    double VDIOgradCoeffTemp1;
    double VDIOgradCoeffTemp2;
    double VDIOtrb1;
    double VDIOtrb2;

    double VDMOStcvth;
    double VDMOSrthjc;
    double VDMOSrthca;
    double VDMOScthj;
    double VDMOSmu;
    double VDMOStexp0;
    double VDMOStexp1;
    double VDMOStrd1;
    double VDMOStrd2;
    double VDMOStrg1;
    double VDMOStrg2;
    double VDMOStrs1;
    double VDMOStrs2;
    double VDMOStksubthres1;
    double VDMOStksubthres2;

    double VDMOSvgsMax;
    double VDMOSvgdMax;
    double VDMOSvdsMax;
    double VDMOSvgsrMax;
    double VDMOSvgdrMax;
    double VDMOSid_max;
    double VDMOSidr_max;
    double VDMOSpd_max;
    double VDMOSrth_ext;
    double VDMOSte_max;
    double VDMOSderating;

    unsigned VDMOStypeGiven  :1;
    unsigned VDMOSdrainResistanceGiven   :1;
    unsigned VDMOSsourceResistanceGiven  :1;
    unsigned VDMOSgateResistanceGiven    :1;
    unsigned VDMOSqsResistanceGiven    :1;
    unsigned VDMOSqsVoltageGiven    :1;
    unsigned VDMOSqsGiven    :1;
    unsigned VDMOStransconductanceGiven  :1;
    unsigned VDMOSvth0Given   :1;
    unsigned VDMOSphiGiven   :1;
    unsigned VDMOSlambdaGiven    :1;
    unsigned VDMOSthetaGiven    :1;
    unsigned VDMOStnomGiven  :1;
    unsigned VDMOSfNcoefGiven  :1;
    unsigned VDMOSfNexpGiven   :1;

    unsigned VDMOScgdminGiven   :1;
    unsigned VDMOScgdmaxGiven   :1;
    unsigned VDMOScgsGiven   :1;
    unsigned VDMOSaGiven   :1;
    unsigned VDMOSsubshiftGiven   :1;
    unsigned VDMOSksubthresGiven :1;
    unsigned VDMOSmtrGiven   :1;
    unsigned VDMOSrdsGiven   :1;

    unsigned VDIOjctSatCurGiven :1;
    unsigned VDIOgradCoeffGiven    :1;
    unsigned VDIOdepletionCapCoeffGiven :1;
    unsigned VDIObvGiven   :1;
    unsigned VDIOibvGiven   :1;
    unsigned VDIOjunctionCapGiven :1;
    unsigned VDIOjunctionPotGiven :1;
    unsigned VDIObrkdEmissionCoeffGiven :1;
    unsigned VDIOresistanceGiven :1;
    unsigned VDIOnGiven   :1;
    unsigned VDIOtransitTimeGiven :1;
    unsigned VDIOegGiven   :1;
    unsigned VDIOxtiGiven   :1;
    unsigned VDIOtrb1Given :1;
    unsigned VDIOtrb2Given :1;

    unsigned VDMOStcvthGiven :1;
    unsigned VDMOSrthjcGiven :1;
    unsigned VDMOSrthcaGiven :1;
    unsigned VDMOScthjGiven :1;
    unsigned VDMOSmuGiven :1;
    unsigned VDMOStexp0Given :1;
    unsigned VDMOStexp1Given :1;
    unsigned VDMOStrd1Given :1;
    unsigned VDMOStrd2Given :1;
    unsigned VDMOStrg1Given :1;
    unsigned VDMOStrg2Given :1;
    unsigned VDMOStrs1Given :1;
    unsigned VDMOStrs2Given :1;
    unsigned VDMOStksubthres1Given :1;
    unsigned VDMOStksubthres2Given :1;

    unsigned VDMOSvgsMaxGiven  :1;
    unsigned VDMOSvgdMaxGiven  :1;
    unsigned VDMOSvdsMaxGiven  :1;
    unsigned VDMOSvgsrMaxGiven  :1;
    unsigned VDMOSvgdrMaxGiven  :1;
    unsigned VDMOSrth_extGiven  :1;
    unsigned VDMOSpd_maxGiven  :1;
    unsigned VDMOSte_maxGiven  :1;
    unsigned VDMOSid_maxGiven  :1;
    unsigned VDMOSidr_maxGiven  :1;
    unsigned VDMOSderatingGiven  :1;
} VDMOSmodel;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

/* device parameters */
enum {
    VDMOS_OFF = 1,
    VDMOS_IC,
    VDMOS_IC_VDS,
    VDMOS_IC_VGS,
    VDMOS_CG,
    VDMOS_CS,
    VDMOS_POWER,
    VDMOS_TEMP,
    VDMOS_M,
    VDMOS_DTEMP,
    VDMOS_THERMAL,
};

/* model parameters */
enum {
    VDMOS_MOD_VTH = 101,
    VDMOS_MOD_KP,
    VDMOS_MOD_PHI,
    VDMOS_MOD_LAMBDA,
    VDMOS_MOD_THETA,
    VDMOS_MOD_RD,
    VDMOS_MOD_RS,
    VDMOS_MOD_RG,
    VDMOS_MOD_RQ,
    VDMOS_MOD_VQ,
    VDMOS_MOD_NMOS,
    VDMOS_MOD_PMOS,
    VDMOS_MOD_TNOM,
    VDMOS_MOD_KF,
    VDMOS_MOD_AF,
    VDMOS_MOD_TYPE,
    VDMOS_MOD_DMOS,
    VDMOS_MOD_CGDMIN,
    VDMOS_MOD_CGDMAX,
    VDMOS_MOD_A,
    VDMOS_MOD_CGS,
    VDMOS_MOD_MTRIODE,
    VDMOS_MOD_SUBSHIFT,
    VDMOS_MOD_KSUBTHRES,
    VDMOS_MOD_RDS,
    VDIO_MOD_IS,
    VDIO_MOD_VJ,
    VDIO_MOD_CJ,
    VDIO_MOD_MJ,
    VDIO_MOD_FC,
    VDIO_MOD_RB,
    VDIO_MOD_BV,
    VDIO_MOD_IBV,
    VDIO_MOD_NBV,
    VDIO_MOD_N,
    VDIO_MOD_TT,
    VDIO_MOD_EG,
    VDIO_MOD_XTI,
    VDIO_MOD_TRB1,
    VDIO_MOD_TRB2,
    VDMOS_MOD_TCVTH,
    VDMOS_MOD_RTHJC,
    VDMOS_MOD_RTHCA,
    VDMOS_MOD_CTHJ,
    VDMOS_MOD_MU,
    VDMOS_MOD_TEXP0,
    VDMOS_MOD_TEXP1,
    VDMOS_MOD_TRD1,
    VDMOS_MOD_TRD2,
    VDMOS_MOD_TRG1,
    VDMOS_MOD_TRG2,
    VDMOS_MOD_TRS1,
    VDMOS_MOD_TRS2,
    VDMOS_MOD_TKSUBTHRES1,
    VDMOS_MOD_TKSUBTHRES2,
    VDMOS_MOD_VGS_MAX,
    VDMOS_MOD_VGD_MAX,
    VDMOS_MOD_VDS_MAX,
    VDMOS_MOD_VGSR_MAX,
    VDMOS_MOD_VGDR_MAX,
    VDMOS_MOD_PD_MAX,
    VDMOS_MOD_ID_MAX,
    VDMOS_MOD_IDR_MAX,
    VDMOS_MOD_TE_MAX,
    VDMOS_MOD_RTH_EXT,
    VDMOS_MOD_DERATING,
};

/* device questions */
enum {
    VDMOS_CAPGS = 201,
    VDMOS_CAPGD,
    VDMOS_CAPDS,
    VDMOS_DNODE,
    VDMOS_GNODE,
    VDMOS_SNODE,
    VDMOS_TNODE,
    VDMOS_TCASE,
    VDMOS_DNODEPRIME,
    VDMOS_SNODEPRIME,
    VDMOS_SOURCECONDUCT,
    VDMOS_DRAINCONDUCT,
    VDMOS_VON,
    VDMOS_CD,
    VDMOS_GM,
    VDMOS_GDS,
    VDMOS_VGS,
    VDMOS_VDS,
    VDMOS_QGS,
    VDMOS_CQGS,
    VDMOS_QGD,
    VDMOS_CQGD,
    VDMOS_CDIO,
    VDMOS_SOURCERESIST,
    VDMOS_DRAINRESIST,
};

/* model questions */

void VDMOStempUpdate(VDMOSmodel *, VDMOSinstance *, double , CKTcircuit *);

#include "vdmosext.h"

#endif /*VDMOS*/

