/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

#ifndef MOS1
#define MOS1

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* declarations for level 1 MOSFETs */

/* indices to the array of MOSFET(1) noise sources */

enum {
    MOS1RDNOIZ = 0,
    MOS1RSNOIZ,
    MOS1IDNOIZ,
    MOS1FLNOIZ,
    MOS1TOTNOIZ,
    /* finally, the number of noise sources */
    MOS1NSRCS
};

/* information needed for each instance */

typedef struct sMOS1instance {

    struct GENinstance gen;

#define MOS1modPtr(inst) ((struct sMOS1model *)((inst)->gen.GENmodPtr))
#define MOS1nextInstance(inst) ((struct sMOS1instance *)((inst)->gen.GENnextInstance))
#define MOS1name gen.GENname
#define MOS1states gen.GENstate

    const int MOS1dNode;  /* number of the gate node of the mosfet */
    const int MOS1gNode;  /* number of the gate node of the mosfet */
    const int MOS1sNode;  /* number of the source node of the mosfet */
    const int MOS1bNode;  /* number of the bulk node of the mosfet */
    int MOS1dNodePrime; /* number of the internal drain node of the mosfet */
    int MOS1sNodePrime; /* number of the internal source node of the mosfet */
    
    double MOS1m;   /* parallel device multiplier */

    double MOS1l;   /* the length of the channel region */
    double MOS1w;   /* the width of the channel region */
    double MOS1drainArea;   /* the area of the drain diffusion */
    double MOS1sourceArea;  /* the area of the source diffusion */
    double MOS1drainSquares;    /* the length of the drain in squares */
    double MOS1sourceSquares;   /* the length of the source in squares */
    double MOS1drainPerimiter;
    double MOS1sourcePerimiter;
    double MOS1sourceConductance;   /*conductance of source(or 0):set in setup*/
    double MOS1drainConductance;    /*conductance of drain(or 0):set in setup*/
    double MOS1temp;    /* operating temperature of this instance */
    double MOS1dtemp;   /* operating temperature of the instance relative to circuit temperature*/

    double MOS1tTransconductance;   /* temperature corrected transconductance*/
    double MOS1tSurfMob;            /* temperature corrected surface mobility */
    double MOS1tPhi;                /* temperature corrected Phi */
    double MOS1tVto;                /* temperature corrected Vto */
    double MOS1tSatCur;             /* temperature corrected saturation Cur. */
    double MOS1tSatCurDens; /* temperature corrected saturation Cur. density*/
    double MOS1tCbd;                /* temperature corrected B-D Capacitance */
    double MOS1tCbs;                /* temperature corrected B-S Capacitance */
    double MOS1tCj;         /* temperature corrected Bulk bottom Capacitance */
    double MOS1tCjsw;       /* temperature corrected Bulk side Capacitance */
    double MOS1tBulkPot;    /* temperature corrected Bulk potential */
    double MOS1tDepCap;     /* temperature adjusted transition point in */
                            /* the cureve matching Fc * Vj */
    double MOS1tVbi;        /* temperature adjusted Vbi */

    double MOS1icVBS;   /* initial condition B-S voltage */
    double MOS1icVDS;   /* initial condition D-S voltage */
    double MOS1icVGS;   /* initial condition G-S voltage */
    double MOS1von;
    double MOS1vdsat;
    double MOS1sourceVcrit; /* Vcrit for pos. vds */
    double MOS1drainVcrit;  /* Vcrit for pos. vds */
    double MOS1cd;
    double MOS1cbs;
    double MOS1cbd;
    double MOS1gmbs;
    double MOS1gm;
    double MOS1gds;
    double MOS1gbd;
    double MOS1gbs;
    double MOS1capbd;
    double MOS1capbs;
    double MOS1Cbd;
    double MOS1Cbdsw;
    double MOS1Cbs;
    double MOS1Cbssw;
    double MOS1f2d;
    double MOS1f3d;
    double MOS1f4d;
    double MOS1f2s;
    double MOS1f3s;
    double MOS1f4s;

/*
 * naming convention:
 * x = vgs
 * y = vbs
 * z = vds
 * cdr = cdrain
 */

#define	MOS1NDCOEFFS	30

#ifndef NODISTO
	double MOS1dCoeffs[MOS1NDCOEFFS];
#else /* NODISTO */
	double *MOS1dCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	capbs2		MOS1dCoeffs[0]
#define	capbs3		MOS1dCoeffs[1]
#define	capbd2		MOS1dCoeffs[2]
#define	capbd3		MOS1dCoeffs[3]
#define	gbs2		MOS1dCoeffs[4]
#define	gbs3		MOS1dCoeffs[5]
#define	gbd2		MOS1dCoeffs[6]
#define	gbd3		MOS1dCoeffs[7]
#define	capgb2		MOS1dCoeffs[8]
#define	capgb3		MOS1dCoeffs[9]
#define	cdr_x2		MOS1dCoeffs[10]
#define	cdr_y2		MOS1dCoeffs[11]
#define	cdr_z2		MOS1dCoeffs[12]
#define	cdr_xy		MOS1dCoeffs[13]
#define	cdr_yz		MOS1dCoeffs[14]
#define	cdr_xz		MOS1dCoeffs[15]
#define	cdr_x3		MOS1dCoeffs[16]
#define	cdr_y3		MOS1dCoeffs[17]
#define	cdr_z3		MOS1dCoeffs[18]
#define	cdr_x2z		MOS1dCoeffs[19]
#define	cdr_x2y		MOS1dCoeffs[20]
#define	cdr_y2z		MOS1dCoeffs[21]
#define	cdr_xy2		MOS1dCoeffs[22]
#define	cdr_xz2		MOS1dCoeffs[23]
#define	cdr_yz2		MOS1dCoeffs[24]
#define	cdr_xyz		MOS1dCoeffs[25]
#define	capgs2		MOS1dCoeffs[26]
#define	capgs3		MOS1dCoeffs[27]
#define	capgd2		MOS1dCoeffs[28]
#define	capgd3		MOS1dCoeffs[29]

#endif

#ifndef NONOISE
    double MOS1nVar[NSTATVARS][MOS1NSRCS];
#else /* NONOISE */
	double **MOS1nVar;
#endif /* NONOISE */

    int MOS1mode;       /* device mode : 1 = normal, -1 = inverse */


    unsigned MOS1off:1;  /* non-zero to indicate device is off for dc analysis*/
    unsigned MOS1tempGiven :1;  /* instance temperature specified */
    unsigned MOS1dtempGiven :1;  /* instance delta temperature specified */
    unsigned MOS1mGiven :1;
    unsigned MOS1lGiven :1;
    unsigned MOS1wGiven :1;
    unsigned MOS1drainAreaGiven :1;
    unsigned MOS1sourceAreaGiven    :1;
    unsigned MOS1drainSquaresGiven  :1;
    unsigned MOS1sourceSquaresGiven :1;
    unsigned MOS1drainPerimiterGiven    :1;
    unsigned MOS1sourcePerimiterGiven   :1;
    unsigned MOS1dNodePrimeSet  :1;
    unsigned MOS1sNodePrimeSet  :1;
    unsigned MOS1icVBSGiven :1;
    unsigned MOS1icVDSGiven :1;
    unsigned MOS1icVGSGiven :1;
    unsigned MOS1vonGiven   :1;
    unsigned MOS1vdsatGiven :1;
    unsigned MOS1modeGiven  :1;


    double *MOS1DdPtr;      /* pointer to sparse matrix element at
                                     * (Drain node,drain node) */
    double *MOS1GgPtr;      /* pointer to sparse matrix element at
                                     * (gate node,gate node) */
    double *MOS1SsPtr;      /* pointer to sparse matrix element at
                                     * (source node,source node) */
    double *MOS1BbPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,bulk node) */
    double *MOS1DPdpPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,drain prime node) */
    double *MOS1SPspPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,source prime node) */
    double *MOS1DdpPtr;     /* pointer to sparse matrix element at
                                     * (drain node,drain prime node) */
    double *MOS1GbPtr;      /* pointer to sparse matrix element at
                                     * (gate node,bulk node) */
    double *MOS1GdpPtr;     /* pointer to sparse matrix element at
                                     * (gate node,drain prime node) */
    double *MOS1GspPtr;     /* pointer to sparse matrix element at
                                     * (gate node,source prime node) */
    double *MOS1SspPtr;     /* pointer to sparse matrix element at
                                     * (source node,source prime node) */
    double *MOS1BdpPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,drain prime node) */
    double *MOS1BspPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,source prime node) */
    double *MOS1DPspPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,source prime node) */
    double *MOS1DPdPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,drain node) */
    double *MOS1BgPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,gate node) */
    double *MOS1DPgPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,gate node) */

    double *MOS1SPgPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,gate node) */
    double *MOS1SPsPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,source node) */
    double *MOS1DPbPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,bulk node) */
    double *MOS1SPbPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,bulk node) */
    double *MOS1SPdpPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,drain prime node) */

    int  MOS1senParmNo;   /* parameter # for sensitivity use;
            set equal to 0  if  neither length
            nor width of the mosfet is a design
            parameter */
    unsigned MOS1sens_l :1;   /* field which indicates whether  
            length of the mosfet is a design
            parameter or not */
    unsigned MOS1sens_w :1;   /* field which indicates whether  
            width of the mosfet is a design
            parameter or not */
    unsigned MOS1senPertFlag :1; /* indictes whether the the 
            parameter of the particular instance is 
            to be perturbed */
    double MOS1cgs;
    double MOS1cgd;
    double MOS1cgb;
    double *MOS1sens;

#define MOS1senCgs MOS1sens /* contains pertured values of cgs */
#define MOS1senCgd MOS1sens + 6 /* contains perturbed values of cgd*/
#define MOS1senCgb MOS1sens + 12 /* contains perturbed values of cgb*/
#define MOS1senCbd MOS1sens + 18 /* contains perturbed values of cbd*/
#define MOS1senCbs MOS1sens + 24 /* contains perturbed values of cbs*/
#define MOS1senGds MOS1sens + 30 /* contains perturbed values of gds*/
#define MOS1senGbs MOS1sens + 36 /* contains perturbed values of gbs*/
#define MOS1senGbd MOS1sens + 42 /* contains perturbed values of gbd*/
#define MOS1senGm MOS1sens + 48 /* contains perturbed values of gm*/
#define MOS1senGmbs MOS1sens + 54 /* contains perturbed values of gmbs*/
#define MOS1dphigs_dl MOS1sens + 60
#define MOS1dphigd_dl MOS1sens + 61
#define MOS1dphigb_dl MOS1sens + 62
#define MOS1dphibs_dl MOS1sens + 63
#define MOS1dphibd_dl MOS1sens + 64
#define MOS1dphigs_dw MOS1sens + 65
#define MOS1dphigd_dw MOS1sens + 66
#define MOS1dphigb_dw MOS1sens + 67
#define MOS1dphibs_dw MOS1sens + 68
#define MOS1dphibd_dw MOS1sens + 69

} MOS1instance ;

#define MOS1vbd MOS1states+ 0   /* bulk-drain voltage */
#define MOS1vbs MOS1states+ 1   /* bulk-source voltage */
#define MOS1vgs MOS1states+ 2   /* gate-source voltage */
#define MOS1vds MOS1states+ 3   /* drain-source voltage */

#define MOS1capgs MOS1states+4  /* gate-source capacitor value */
#define MOS1qgs MOS1states+ 5   /* gate-source capacitor charge */
#define MOS1cqgs MOS1states+ 6  /* gate-source capacitor current */

#define MOS1capgd MOS1states+ 7 /* gate-drain capacitor value */
#define MOS1qgd MOS1states+ 8   /* gate-drain capacitor charge */
#define MOS1cqgd MOS1states+ 9  /* gate-drain capacitor current */

#define MOS1capgb MOS1states+10 /* gate-bulk capacitor value */
#define MOS1qgb MOS1states+ 11  /* gate-bulk capacitor charge */
#define MOS1cqgb MOS1states+ 12 /* gate-bulk capacitor current */

#define MOS1qbd MOS1states+ 13  /* bulk-drain capacitor charge */
#define MOS1cqbd MOS1states+ 14 /* bulk-drain capacitor current */

#define MOS1qbs MOS1states+ 15  /* bulk-source capacitor charge */
#define MOS1cqbs MOS1states+ 16 /* bulk-source capacitor current */

#define MOS1numStates 17

#define MOS1sensxpgs MOS1states+17 /* charge sensitivities and their derivatives.
                                    * +18 for the derivatives
                                    * pointer to the beginning of the array */
#define MOS1sensxpgd  MOS1states+19
#define MOS1sensxpgb  MOS1states+21
#define MOS1sensxpbs  MOS1states+23
#define MOS1sensxpbd  MOS1states+25

#define MOS1numSenStates 10


/* per model data */

    /* NOTE:  parameters marked 'input - use xxxx' are paramters for
     * which a temperature correction is applied in MOS1temp, thus
     * the MOS1xxxx value in the per-instance structure should be used
     * instead in all calculations 
     */


typedef struct sMOS1model {       /* model structure for a resistor */

    struct GENmodel gen;

#define MOS1modType gen.GENmodType
#define MOS1nextModel(inst) ((struct sMOS1model *)((inst)->gen.GENnextModel))
#define MOS1instances(inst) ((MOS1instance *)((inst)->gen.GENinstances))
#define MOS1modName gen.GENmodName

    int MOS1type;       /* device type : 1 = nmos,  -1 = pmos */
    double MOS1tnom;        /* temperature at which parameters measured */
    double MOS1latDiff;
    double MOS1jctSatCurDensity;    /* input - use tSatCurDens */
    double MOS1jctSatCur;   /* input - use tSatCur */
    double MOS1drainResistance;
    double MOS1sourceResistance;
    double MOS1sheetResistance;
    double MOS1transconductance;    /* input - use tTransconductance */
    double MOS1gateSourceOverlapCapFactor;
    double MOS1gateDrainOverlapCapFactor;
    double MOS1gateBulkOverlapCapFactor;
    double MOS1oxideCapFactor;
    double MOS1vt0; /* input - use tVto */
    double MOS1capBD;   /* input - use tCbd */
    double MOS1capBS;   /* input - use tCbs */
    double MOS1bulkCapFactor;   /* input - use tCj */
    double MOS1sideWallCapFactor;   /* input - use tCjsw */
    double MOS1bulkJctPotential;    /* input - use tBulkPot */
    double MOS1bulkJctBotGradingCoeff;
    double MOS1bulkJctSideGradingCoeff;
    double MOS1fwdCapDepCoeff;
    double MOS1phi; /* input - use tPhi */
    double MOS1gamma;
    double MOS1lambda;
    double MOS1substrateDoping;
    int MOS1gateType;
    double MOS1surfaceStateDensity;
    double MOS1oxideThickness;
    double MOS1surfaceMobility; /* input - use tSurfMob */
    double MOS1fNcoef;
    double MOS1fNexp;

    unsigned MOS1typeGiven  :1;
    unsigned MOS1latDiffGiven   :1;
    unsigned MOS1jctSatCurDensityGiven  :1;
    unsigned MOS1jctSatCurGiven :1;
    unsigned MOS1drainResistanceGiven   :1;
    unsigned MOS1sourceResistanceGiven  :1;
    unsigned MOS1sheetResistanceGiven   :1;
    unsigned MOS1transconductanceGiven  :1;
    unsigned MOS1gateSourceOverlapCapFactorGiven    :1;
    unsigned MOS1gateDrainOverlapCapFactorGiven :1;
    unsigned MOS1gateBulkOverlapCapFactorGiven  :1;
    unsigned MOS1vt0Given   :1;
    unsigned MOS1capBDGiven :1;
    unsigned MOS1capBSGiven :1;
    unsigned MOS1bulkCapFactorGiven :1;
    unsigned MOS1sideWallCapFactorGiven   :1;
    unsigned MOS1bulkJctPotentialGiven  :1;
    unsigned MOS1bulkJctBotGradingCoeffGiven    :1;
    unsigned MOS1bulkJctSideGradingCoeffGiven   :1;
    unsigned MOS1fwdCapDepCoeffGiven    :1;
    unsigned MOS1phiGiven   :1;
    unsigned MOS1gammaGiven :1;
    unsigned MOS1lambdaGiven    :1;
    unsigned MOS1substrateDopingGiven   :1;
    unsigned MOS1gateTypeGiven  :1;
    unsigned MOS1surfaceStateDensityGiven   :1;
    unsigned MOS1oxideThicknessGiven    :1;
    unsigned MOS1surfaceMobilityGiven   :1;
    unsigned MOS1tnomGiven  :1;
    unsigned MOS1fNcoefGiven  :1;
    unsigned MOS1fNexpGiven   :1;

} MOS1model;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

/* device parameters */
enum {
    MOS1_W = 1,
    MOS1_L,
    MOS1_AS,
    MOS1_AD,
    MOS1_PS,
    MOS1_PD,
    MOS1_NRS,
    MOS1_NRD,
    MOS1_OFF,
    MOS1_IC,
    MOS1_IC_VBS,
    MOS1_IC_VDS,
    MOS1_IC_VGS,
    MOS1_W_SENS,
    MOS1_L_SENS,
    MOS1_CB,
    MOS1_CG,
    MOS1_CS,
    MOS1_POWER,
    MOS1_TEMP,
    MOS1_M,
    MOS1_DTEMP,
};

/* model paramerers */
enum {
    MOS1_MOD_VTO = 101,
    MOS1_MOD_KP,
    MOS1_MOD_GAMMA,
    MOS1_MOD_PHI,
    MOS1_MOD_LAMBDA,
    MOS1_MOD_RD,
    MOS1_MOD_RS,
    MOS1_MOD_CBD,
    MOS1_MOD_CBS,
    MOS1_MOD_IS,
    MOS1_MOD_PB,
    MOS1_MOD_CGSO,
    MOS1_MOD_CGDO,
    MOS1_MOD_CGBO,
    MOS1_MOD_CJ,
    MOS1_MOD_MJ,
    MOS1_MOD_CJSW,
    MOS1_MOD_MJSW,
    MOS1_MOD_JS,
    MOS1_MOD_TOX,
    MOS1_MOD_LD,
    MOS1_MOD_RSH,
    MOS1_MOD_U0,
    MOS1_MOD_FC,
    MOS1_MOD_NSUB,
    MOS1_MOD_TPG,
    MOS1_MOD_NSS,
    MOS1_MOD_NMOS,
    MOS1_MOD_PMOS,
    MOS1_MOD_TNOM,
    MOS1_MOD_KF,
    MOS1_MOD_AF,
    MOS1_MOD_TYPE,
};

/* device questions */
enum {
    MOS1_CGS = 201,
    MOS1_CGD,
    MOS1_DNODE,
    MOS1_GNODE,
    MOS1_SNODE,
    MOS1_BNODE,
    MOS1_DNODEPRIME,
    MOS1_SNODEPRIME,
    MOS1_SOURCECONDUCT,
    MOS1_DRAINCONDUCT,
    MOS1_VON,
    MOS1_VDSAT,
    MOS1_SOURCEVCRIT,
    MOS1_DRAINVCRIT,
    MOS1_CD,
    MOS1_CBS,
    MOS1_CBD,
    MOS1_GMBS,
    MOS1_GM,
    MOS1_GDS,
    MOS1_GBD,
    MOS1_GBS,
    MOS1_CAPBD,
    MOS1_CAPBS,
    MOS1_CAPZEROBIASBD,
    MOS1_CAPZEROBIASBDSW,
    MOS1_CAPZEROBIASBS,
    MOS1_CAPZEROBIASBSSW,
    MOS1_VBD,
    MOS1_VBS,
    MOS1_VGS,
    MOS1_VDS,
    MOS1_CAPGS,
    MOS1_QGS,
    MOS1_CQGS,
    MOS1_CAPGD,
    MOS1_QGD,
    MOS1_CQGD,
    MOS1_CAPGB,
    MOS1_QGB,
    MOS1_CQGB,
    MOS1_QBD,
    MOS1_CQBD,
    MOS1_QBS,
    MOS1_CQBS,
    MOS1_L_SENS_REAL,
    MOS1_L_SENS_IMAG,
    MOS1_L_SENS_MAG,
    MOS1_L_SENS_PH,
    MOS1_L_SENS_CPLX,
    MOS1_W_SENS_REAL,
    MOS1_W_SENS_IMAG,
    MOS1_W_SENS_MAG,
    MOS1_W_SENS_PH,
    MOS1_W_SENS_CPLX,
    MOS1_L_SENS_DC,
    MOS1_W_SENS_DC,
    MOS1_SOURCERESIST,
    MOS1_DRAINRESIST,
};

/* model questions */

#include "mos1ext.h"

#endif /*MOS1*/

