/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef MOS6
#define MOS6

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"

/* declarations for level 5 MOSFETs */

/* information needed for each instance */

typedef struct sMOS6instance {

    struct GENinstance gen;

#define MOS6modPtr(inst) ((struct sMOS6model *)((inst)->gen.GENmodPtr))
#define MOS6nextInstance(inst) ((struct sMOS6instance *)((inst)->gen.GENnextInstance))
#define MOS6name gen.GENname
#define MOS6states gen.GENstate

    const int MOS6dNode;  /* number of the gate node of the mosfet */
    const int MOS6gNode;  /* number of the gate node of the mosfet */
    const int MOS6sNode;  /* number of the source node of the mosfet */
    const int MOS6bNode;  /* number of the bulk node of the mosfet */
    int MOS6dNodePrime; /* number of the internal drain node of the mosfet */
    int MOS6sNodePrime; /* number of the internal source node of the mosfet */

    double MOS6l;   /* the length of the channel region */
    double MOS6w;   /* the width of the channel region */
    double MOS6m;   /* the parallel multiplier */
    double MOS6drainArea;   /* the area of the drain diffusion */
    double MOS6sourceArea;  /* the area of the source diffusion */
    double MOS6drainSquares;    /* the length of the drain in squares */
    double MOS6sourceSquares;   /* the length of the source in squares */
    double MOS6drainPerimiter;
    double MOS6sourcePerimiter;
    double MOS6sourceConductance;   /*conductance of source(or 0):set in setup*/
    double MOS6drainConductance;    /*conductance of drain(or 0):set in setup*/
    double MOS6temp;    /* operating temperature of this instance */
    double MOS6dtemp;   /* instance temperature difference from circuit */

    double MOS6tKv;         /* temperature corrected drain linear cond. factor*/
    double MOS6tKc;         /* temperature corrected saturation cur. factor*/
    double MOS6tSurfMob;    /* temperature corrected surface mobility */
    double MOS6tPhi;        /* temperature corrected Phi */
    double MOS6tVto;        /* temperature corrected Vto */
    double MOS6tSatCur;     /* temperature corrected saturation Cur. */
    double MOS6tSatCurDens; /* temperature corrected saturation Cur. density*/
    double MOS6tCbd;        /* temperature corrected B-D Capacitance */
    double MOS6tCbs;        /* temperature corrected B-S Capacitance */
    double MOS6tCj;         /* temperature corrected Bulk bottom Capacitance */
    double MOS6tCjsw;       /* temperature corrected Bulk side Capacitance */
    double MOS6tBulkPot;    /* temperature corrected Bulk potential */
    double MOS6tDepCap;     /* temperature adjusted transition point in */
                            /* the cureve matching Fc * Vj */
    double MOS6tVbi;        /* temperature adjusted Vbi */

    double MOS6icVBS;   /* initial condition B-S voltage */
    double MOS6icVDS;   /* initial condition D-S voltage */
    double MOS6icVGS;   /* initial condition G-S voltage */
    double MOS6von;
    double MOS6vdsat;
    double MOS6sourceVcrit; /* Vcrit for pos. vds */
    double MOS6drainVcrit;  /* Vcrit for pos. vds */
    double MOS6cd;
    double MOS6cbs;
    double MOS6cbd;
    double MOS6gmbs;
    double MOS6gm;
    double MOS6gds;
    double MOS6gbd;
    double MOS6gbs;
    double MOS6capbd;
    double MOS6capbs;
    double MOS6Cbd;
    double MOS6Cbdsw;
    double MOS6Cbs;
    double MOS6Cbssw;
    double MOS6f2d;
    double MOS6f3d;
    double MOS6f4d;
    double MOS6f2s;
    double MOS6f3s;
    double MOS6f4s;
    int MOS6mode;       /* device mode : 1 = normal, -1 = inverse */


    unsigned MOS6off:1;  /* non-zero to indicate device is off for dc analysis*/
    unsigned MOS6tempGiven :1;  /* instance temperature specified */
    unsigned MOS6dtempGiven :1;
    unsigned MOS6lGiven :1;
    unsigned MOS6wGiven :1;
    unsigned MOS6mGiven :1;
    unsigned MOS6drainAreaGiven :1;
    unsigned MOS6sourceAreaGiven    :1;
    unsigned MOS6drainSquaresGiven  :1;
    unsigned MOS6sourceSquaresGiven :1;
    unsigned MOS6drainPerimiterGiven    :1;
    unsigned MOS6sourcePerimiterGiven   :1;
    unsigned MOS6dNodePrimeSet  :1;
    unsigned MOS6sNodePrimeSet  :1;
    unsigned MOS6icVBSGiven :1;
    unsigned MOS6icVDSGiven :1;
    unsigned MOS6icVGSGiven :1;
    unsigned MOS6vonGiven   :1;
    unsigned MOS6vdsatGiven :1;
    unsigned MOS6modeGiven  :1;


    double *MOS6DdPtr;      /* pointer to sparse matrix element at
                                     * (Drain node,drain node) */
    double *MOS6GgPtr;      /* pointer to sparse matrix element at
                                     * (gate node,gate node) */
    double *MOS6SsPtr;      /* pointer to sparse matrix element at
                                     * (source node,source node) */
    double *MOS6BbPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,bulk node) */
    double *MOS6DPdpPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,drain prime node) */
    double *MOS6SPspPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,source prime node) */
    double *MOS6DdpPtr;     /* pointer to sparse matrix element at
                                     * (drain node,drain prime node) */
    double *MOS6GbPtr;      /* pointer to sparse matrix element at
                                     * (gate node,bulk node) */
    double *MOS6GdpPtr;     /* pointer to sparse matrix element at
                                     * (gate node,drain prime node) */
    double *MOS6GspPtr;     /* pointer to sparse matrix element at
                                     * (gate node,source prime node) */
    double *MOS6SspPtr;     /* pointer to sparse matrix element at
                                     * (source node,source prime node) */
    double *MOS6BdpPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,drain prime node) */
    double *MOS6BspPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,source prime node) */
    double *MOS6DPspPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,source prime node) */
    double *MOS6DPdPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,drain node) */
    double *MOS6BgPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,gate node) */
    double *MOS6DPgPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,gate node) */

    double *MOS6SPgPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,gate node) */
    double *MOS6SPsPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,source node) */
    double *MOS6DPbPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,bulk node) */
    double *MOS6SPbPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,bulk node) */
    double *MOS6SPdpPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,drain prime node) */

    int  MOS6senParmNo;   /* parameter # for sensitivity use;
            set equal to 0  if  neither length
            nor width of the mosfet is a design
            parameter */
    unsigned MOS6sens_l :1;   /* field which indicates whether  
            length of the mosfet is a design
            parameter or not */
    unsigned MOS6sens_w :1;   /* field which indicates whether  
            width of the mosfet is a design
            parameter or not */
    unsigned MOS6senPertFlag :1; /* indictes whether the the 
            parameter of the particular instance is 
            to be perturbed */
    double MOS6cgs;
    double MOS6cgd;
    double MOS6cgb;
    double *MOS6sens;

#define MOS6senCgs MOS6sens /* contains pertured values of cgs */
#define MOS6senCgd MOS6sens + 6 /* contains perturbed values of cgd*/
#define MOS6senCgb MOS6sens + 12 /* contains perturbed values of cgb*/
#define MOS6senCbd MOS6sens + 18 /* contains perturbed values of cbd*/
#define MOS6senCbs MOS6sens + 24 /* contains perturbed values of cbs*/
#define MOS6senGds MOS6sens + 30 /* contains perturbed values of gds*/
#define MOS6senGbs MOS6sens + 36 /* contains perturbed values of gbs*/
#define MOS6senGbd MOS6sens + 42 /* contains perturbed values of gbd*/
#define MOS6senGm MOS6sens + 48 /* contains perturbed values of gm*/
#define MOS6senGmbs MOS6sens + 54 /* contains perturbed values of gmbs*/
#define MOS6dphigs_dl MOS6sens + 60
#define MOS6dphigd_dl MOS6sens + 61
#define MOS6dphigb_dl MOS6sens + 62
#define MOS6dphibs_dl MOS6sens + 63
#define MOS6dphibd_dl MOS6sens + 64
#define MOS6dphigs_dw MOS6sens + 65
#define MOS6dphigd_dw MOS6sens + 66
#define MOS6dphigb_dw MOS6sens + 67
#define MOS6dphibs_dw MOS6sens + 68
#define MOS6dphibd_dw MOS6sens + 69

} MOS6instance ;

#define MOS6vbd MOS6states+ 0   /* bulk-drain voltage */
#define MOS6vbs MOS6states+ 1   /* bulk-source voltage */
#define MOS6vgs MOS6states+ 2   /* gate-source voltage */
#define MOS6vds MOS6states+ 3   /* drain-source voltage */

#define MOS6capgs MOS6states+4  /* gate-source capacitor value */
#define MOS6qgs MOS6states+ 5   /* gate-source capacitor charge */
#define MOS6cqgs MOS6states+ 6  /* gate-source capacitor current */

#define MOS6capgd MOS6states+ 7 /* gate-drain capacitor value */
#define MOS6qgd MOS6states+ 8   /* gate-drain capacitor charge */
#define MOS6cqgd MOS6states+ 9  /* gate-drain capacitor current */

#define MOS6capgb MOS6states+10 /* gate-bulk capacitor value */
#define MOS6qgb MOS6states+ 11  /* gate-bulk capacitor charge */
#define MOS6cqgb MOS6states+ 12 /* gate-bulk capacitor current */

#define MOS6qbd MOS6states+ 13  /* bulk-drain capacitor charge */
#define MOS6cqbd MOS6states+ 14 /* bulk-drain capacitor current */

#define MOS6qbs MOS6states+ 15  /* bulk-source capacitor charge */
#define MOS6cqbs MOS6states+ 16 /* bulk-source capacitor current */

#define MOS6numStates 17

#define MOS6sensxpgs MOS6states+17 /* charge sensitivities and their derivatives.
                                    * +18 for the derivatives
                                    * pointer to the beginning of the array */
#define MOS6sensxpgd  MOS6states+19
#define MOS6sensxpgb  MOS6states+21
#define MOS6sensxpbs  MOS6states+23
#define MOS6sensxpbd  MOS6states+25

#define MOS6numSenStates 10


/* per model data */

    /* NOTE:  parameters marked 'input - use xxxx' are paramters for
     * which a temperature correction is applied in MOS6temp, thus
     * the MOS6xxxx value in the per-instance structure should be used
     * instead in all calculations 
     */


typedef struct sMOS6model {       /* model structure for a resistor */

    struct GENmodel gen;

#define MOS6modType gen.GENmodType
#define MOS6nextModel(inst) ((struct sMOS6model *)((inst)->gen.GENnextModel))
#define MOS6instances(inst) ((MOS6instance *)((inst)->gen.GENinstances))
#define MOS6modName gen.GENmodName

    int MOS6type;       /* device type : 1 = nmos,  -1 = pmos */
    double MOS6tnom;        /* temperature at which parameters measured */
    double MOS6latDiff;
    double MOS6jctSatCurDensity;    /* input - use tSatCurDens */
    double MOS6jctSatCur;   /* input - use tSatCur */
    double MOS6drainResistance;
    double MOS6sourceResistance;
    double MOS6sheetResistance;
    double MOS6kv;    /* input - use tKv */
    double MOS6nv;    /* drain linear conductance factor*/
    double MOS6kc;    /* input - use tKc */
    double MOS6nc;    /* saturation current coeff.*/
    double MOS6nvth;    /* threshold voltage coeff.*/
    double MOS6ps;    /* saturation current modification parameter*/
    double MOS6gateSourceOverlapCapFactor;
    double MOS6gateDrainOverlapCapFactor;
    double MOS6gateBulkOverlapCapFactor;
    double MOS6oxideCapFactor;
    double MOS6vt0; /* input - use tVto */
    double MOS6capBD;   /* input - use tCbd */
    double MOS6capBS;   /* input - use tCbs */
    double MOS6bulkCapFactor;   /* input - use tCj */
    double MOS6sideWallCapFactor;   /* input - use tCjsw */
    double MOS6bulkJctPotential;    /* input - use tBulkPot */
    double MOS6bulkJctBotGradingCoeff;
    double MOS6bulkJctSideGradingCoeff;
    double MOS6fwdCapDepCoeff;
    double MOS6phi; /* input - use tPhi */
    double MOS6gamma;
    double MOS6gamma1;  /* secondary back-gate effect parametr */
    double MOS6sigma;
    double MOS6lambda;
    double MOS6lamda0;
    double MOS6lamda1;
    double MOS6substrateDoping;
    int MOS6gateType;
    double MOS6surfaceStateDensity;
    double MOS6oxideThickness;
    double MOS6surfaceMobility; /* input - use tSurfMob */

    unsigned MOS6typeGiven  :1;
    unsigned MOS6latDiffGiven   :1;
    unsigned MOS6jctSatCurDensityGiven  :1;
    unsigned MOS6jctSatCurGiven :1;
    unsigned MOS6drainResistanceGiven   :1;
    unsigned MOS6sourceResistanceGiven  :1;
    unsigned MOS6sheetResistanceGiven   :1;
    unsigned MOS6kvGiven  :1;
    unsigned MOS6nvGiven  :1;
    unsigned MOS6kcGiven  :1;
    unsigned MOS6ncGiven  :1;
    unsigned MOS6nvthGiven  :1;
    unsigned MOS6psGiven  :1;
    unsigned MOS6gateSourceOverlapCapFactorGiven    :1;
    unsigned MOS6gateDrainOverlapCapFactorGiven :1;
    unsigned MOS6gateBulkOverlapCapFactorGiven  :1;
    unsigned MOS6vt0Given   :1;
    unsigned MOS6capBDGiven :1;
    unsigned MOS6capBSGiven :1;
    unsigned MOS6bulkCapFactorGiven :1;
    unsigned MOS6sideWallCapFactorGiven   :1;
    unsigned MOS6bulkJctPotentialGiven  :1;
    unsigned MOS6bulkJctBotGradingCoeffGiven    :1;
    unsigned MOS6bulkJctSideGradingCoeffGiven   :1;
    unsigned MOS6fwdCapDepCoeffGiven    :1;
    unsigned MOS6phiGiven   :1;
    unsigned MOS6gammaGiven :1;
    unsigned MOS6gamma1Given :1;
    unsigned MOS6sigmaGiven :1;
    unsigned MOS6lambdaGiven    :1;
    unsigned MOS6lamda0Given    :1;
    unsigned MOS6lamda1Given    :1;
    unsigned MOS6substrateDopingGiven   :1;
    unsigned MOS6gateTypeGiven  :1;
    unsigned MOS6surfaceStateDensityGiven   :1;
    unsigned MOS6oxideThicknessGiven    :1;
    unsigned MOS6surfaceMobilityGiven   :1;
    unsigned MOS6tnomGiven  :1;

} MOS6model;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

/* device parameters */
enum {
    MOS6_W = 1,
    MOS6_L,
    MOS6_AS,
    MOS6_AD,
    MOS6_PS,
    MOS6_PD,
    MOS6_NRS,
    MOS6_NRD,
    MOS6_OFF,
    MOS6_IC,
    MOS6_IC_VBS,
    MOS6_IC_VDS,
    MOS6_IC_VGS,
    MOS6_W_SENS,
    MOS6_L_SENS,
    MOS6_CB,
    MOS6_CG,
    MOS6_CS,
    MOS6_POWER,
    MOS6_TEMP,
    MOS6_DTEMP,
    MOS6_M,
};

/* model paramerers */
enum {
    MOS6_MOD_VTO = 101,
    MOS6_MOD_KV,
    MOS6_MOD_NV,
    MOS6_MOD_KC,
    MOS6_MOD_NC,
    MOS6_MOD_NVTH,
    MOS6_MOD_PS,
    MOS6_MOD_GAMMA,
    MOS6_MOD_GAMMA1,
    MOS6_MOD_SIGMA,
    MOS6_MOD_PHI,
    MOS6_MOD_LAMBDA,
    MOS6_MOD_LAMDA0,
    MOS6_MOD_LAMDA1,
    MOS6_MOD_RD,
    MOS6_MOD_RS,
    MOS6_MOD_CBD,
    MOS6_MOD_CBS,
    MOS6_MOD_IS,
    MOS6_MOD_PB,
    MOS6_MOD_CGSO,
    MOS6_MOD_CGDO,
    MOS6_MOD_CGBO,
    MOS6_MOD_CJ,
    MOS6_MOD_MJ,
    MOS6_MOD_CJSW,
    MOS6_MOD_MJSW,
    MOS6_MOD_JS,
    MOS6_MOD_TOX,
    MOS6_MOD_LD,
    MOS6_MOD_RSH,
    MOS6_MOD_U0,
    MOS6_MOD_FC,
    MOS6_MOD_NSUB,
    MOS6_MOD_TPG,
    MOS6_MOD_NSS,
    MOS6_MOD_NMOS,
    MOS6_MOD_PMOS,
    MOS6_MOD_TNOM,
    MOS6_MOD_TYPE,
};

/* device questions */
enum {
    MOS6_CGS = 201,
    MOS6_CGD,
    MOS6_DNODE,
    MOS6_GNODE,
    MOS6_SNODE,
    MOS6_BNODE,
    MOS6_DNODEPRIME,
    MOS6_SNODEPRIME,
    MOS6_SOURCECONDUCT,
    MOS6_DRAINCONDUCT,
    MOS6_VON,
    MOS6_VDSAT,
    MOS6_SOURCEVCRIT,
    MOS6_DRAINVCRIT,
    MOS6_CD,
    MOS6_CBS,
    MOS6_CBD,
    MOS6_GMBS,
    MOS6_GM,
    MOS6_GDS,
    MOS6_GBD,
    MOS6_GBS,
    MOS6_CAPBD,
    MOS6_CAPBS,
    MOS6_CAPZEROBIASBD,
    MOS6_CAPZEROBIASBDSW,
    MOS6_CAPZEROBIASBS,
    MOS6_CAPZEROBIASBSSW,
    MOS6_VBD,
    MOS6_VBS,
    MOS6_VGS,
    MOS6_VDS,
    MOS6_CAPGS,
    MOS6_QGS,
    MOS6_CQGS,
    MOS6_CAPGD,
    MOS6_QGD,
    MOS6_CQGD,
    MOS6_CAPGB,
    MOS6_QGB,
    MOS6_CQGB,
    MOS6_QBD,
    MOS6_CQBD,
    MOS6_QBS,
    MOS6_CQBS,
    MOS6_L_SENS_REAL,
    MOS6_L_SENS_IMAG,
    MOS6_L_SENS_MAG,
    MOS6_L_SENS_PH,
    MOS6_L_SENS_CPLX,
    MOS6_W_SENS_REAL,
    MOS6_W_SENS_IMAG,
    MOS6_W_SENS_MAG,
    MOS6_W_SENS_PH,
    MOS6_W_SENS_CPLX,
    MOS6_L_SENS_DC,
    MOS6_W_SENS_DC,
    MOS6_SOURCERESIST,
    MOS6_DRAINRESIST,
};

/* model questions */

#include "mos6ext.h"

#endif /*MOS6*/

