/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlanFixes
**********/

#ifndef MOS3
#define MOS3

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

    /* declarations for level 3 MOSFETs */

/* indices to the array of MOSFET(3) noise sources */

enum {
    MOS3RDNOIZ = 0,
    MOS3RSNOIZ,
    MOS3IDNOIZ,
    MOS3FLNOIZ,
    MOS3TOTNOIZ,
    /* finally, the number of noise sources */
    MOS3NSRCS
};

/* information needed for each instance */

typedef struct sMOS3instance {

    struct GENinstance gen;

#define MOS3modPtr(inst) ((struct sMOS3model *)((inst)->gen.GENmodPtr))
#define MOS3nextInstance(inst) ((struct sMOS3instance *)((inst)->gen.GENnextInstance))
#define MOS3name gen.GENname
#define MOS3states gen.GENstate

    const int MOS3dNode;  /* number of the gate node of the mosfet */
    const int MOS3gNode;  /* number of the gate node of the mosfet */
    const int MOS3sNode;  /* number of the source node of the mosfet */
    const int MOS3bNode;  /* number of the bulk node of the mosfet */
    int MOS3dNodePrime; /* number of the internal drain node of the mosfet */
    int MOS3sNodePrime; /* number of the internal source node of the mosfet */

    double MOS3m;   /* parallel device multiplier */
    double MOS3l;   /* the length of the channel region */
    double MOS3w;   /* the width of the channel region */
    double MOS3drainArea;   /* the area of the drain diffusion */
    double MOS3sourceArea;  /* the area of the source diffusion */
    double MOS3drainSquares;    /* the length of the drain in squares */
    double MOS3sourceSquares;   /* the length of the source in squares */
    double MOS3drainPerimiter;
    double MOS3sourcePerimiter;
    double MOS3sourceConductance;   /*conductance of source(or 0):set in setup*/
    double MOS3drainConductance;    /*conductance of drain(or 0):set in setup*/
    double MOS3temp;    /* operating temperature of this instance */
    double MOS3dtemp;    /* temperature difference for this instance */

    double MOS3tTransconductance;   /* temperature corrected transconductance*/
    double MOS3tSurfMob;            /* temperature corrected surface mobility */
    double MOS3tPhi;                /* temperature corrected Phi */
    double MOS3tVto;                /* temperature corrected Vto */
    double MOS3tSatCur;             /* temperature corrected saturation Cur. */
    double MOS3tSatCurDens; /* temperature corrected saturation Cur. density*/
    double MOS3tCbd;                /* temperature corrected B-D Capacitance */
    double MOS3tCbs;                /* temperature corrected B-S Capacitance */
    double MOS3tCj;         /* temperature corrected Bulk bottom Capacitance */
    double MOS3tCjsw;       /* temperature corrected Bulk side Capacitance */
    double MOS3tBulkPot;    /* temperature corrected Bulk potential */
    double MOS3tDepCap;     /* temperature adjusted transition point in */
                            /* the cureve matching Fc * Vj */
    double MOS3tVbi;        /* temperature adjusted Vbi */

    double MOS3icVBS;   /* initial condition B-S voltage */
    double MOS3icVDS;   /* initial condition D-S voltage */
    double MOS3icVGS;   /* initial condition G-S voltage */
    double MOS3von;
    double MOS3vdsat;
    double MOS3sourceVcrit; /* vcrit for pos. vds */
    double MOS3drainVcrit;  /* vcrit for neg. vds */
    double MOS3cd;
    double MOS3cbs;
    double MOS3cbd;
    double MOS3gmbs;
    double MOS3gm;
    double MOS3gds;
    double MOS3gbd;
    double MOS3gbs;
    double MOS3capbd;
    double MOS3capbs;
    double MOS3Cbd;
    double MOS3Cbdsw;
    double MOS3Cbs;
    double MOS3Cbssw;
    double MOS3f2d;
    double MOS3f3d;
    double MOS3f4d;
    double MOS3f2s;
    double MOS3f3s;
    double MOS3f4s;
    int MOS3mode;       /* device mode : 1 = normal, -1 = inverse */


    unsigned MOS3off :1;/* non-zero to indicate device is off for dc analysis*/
    unsigned MOS3tempGiven :1;  /* instance temperature specified */
    unsigned MOS3dtempGiven :1;  /* instance temperature difference specified */
    unsigned MOS3mGiven :1;
    unsigned MOS3lGiven :1;
    unsigned MOS3wGiven :1;
    unsigned MOS3drainAreaGiven :1;
    unsigned MOS3sourceAreaGiven    :1;
    unsigned MOS3drainSquaresGiven  :1;
    unsigned MOS3sourceSquaresGiven :1;
    unsigned MOS3drainPerimiterGiven    :1;
    unsigned MOS3sourcePerimiterGiven   :1;
    unsigned MOS3dNodePrimeSet  :1;
    unsigned MOS3sNodePrimeSet  :1;
    unsigned MOS3icVBSGiven :1;
    unsigned MOS3icVDSGiven :1;
    unsigned MOS3icVGSGiven :1;
    unsigned MOS3vonGiven   :1;
    unsigned MOS3vdsatGiven :1;
    unsigned MOS3modeGiven  :1;


    double *MOS3DdPtr;      /* pointer to sparse matrix element at
                                     * (Drain node,drain node) */
    double *MOS3GgPtr;      /* pointer to sparse matrix element at
                                     * (gate node,gate node) */
    double *MOS3SsPtr;      /* pointer to sparse matrix element at
                                     * (source node,source node) */
    double *MOS3BbPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,bulk node) */
    double *MOS3DPdpPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,drain prime node) */
    double *MOS3SPspPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,source prime node) */
    double *MOS3DdpPtr;     /* pointer to sparse matrix element at
                                     * (drain node,drain prime node) */
    double *MOS3GbPtr;      /* pointer to sparse matrix element at
                                     * (gate node,bulk node) */
    double *MOS3GdpPtr;     /* pointer to sparse matrix element at
                                     * (gate node,drain prime node) */
    double *MOS3GspPtr;     /* pointer to sparse matrix element at
                                     * (gate node,source prime node) */
    double *MOS3SspPtr;     /* pointer to sparse matrix element at
                                     * (source node,source prime node) */
    double *MOS3BdpPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,drain prime node) */
    double *MOS3BspPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,source prime node) */
    double *MOS3DPspPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,source prime node) */
    double *MOS3DPdPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,drain node) */
    double *MOS3BgPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,gate node) */
    double *MOS3DPgPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,gate node) */

    double *MOS3SPgPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,gate node) */
    double *MOS3SPsPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,source node) */
    double *MOS3DPbPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,bulk node) */
    double *MOS3SPbPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,bulk node) */
    double *MOS3SPdpPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,drain prime node) */
    int  MOS3senParmNo;   /* parameter # for sensitivity use;
            set equal to 0  if  neither length
            nor width of the mosfet is a design
            parameter */
    unsigned MOS3sens_l :1;   /* field which indicates whether  
              length of the mosfet is a design
              parameter or not */
    unsigned MOS3sens_w :1;   /* field which indicates whether  
              width of the mosfet is a design
              parameter or not */
    unsigned MOS3senPertFlag :1; /* indictes whether the the parameter of
                        the particular instance is to be perturbed */
    double MOS3cgs;
    double MOS3cgd;
    double MOS3cgb;
    double *MOS3sens;

#define MOS3senGdpr MOS3sens
#define MOS3senGspr MOS3sens + 1
#define MOS3senCgs MOS3sens + 2 /* contains pertured values of cgs */
#define MOS3senCgd MOS3sens + 8 /* contains perturbed values of cgd*/
#define MOS3senCgb MOS3sens + 14 /* contains perturbed values of cgb*/
#define MOS3senCbd MOS3sens + 20 /* contains perturbed values of cbd*/
#define MOS3senCbs MOS3sens + 26 /* contains perturbed values of cbs*/
#define MOS3senGds MOS3sens + 32 /* contains perturbed values of gds*/
#define MOS3senGbs MOS3sens + 38 /* contains perturbed values of gbs*/
#define MOS3senGbd MOS3sens + 44 /* contains perturbed values of gbd*/
#define MOS3senGm MOS3sens + 50 /* contains perturbed values of gm*/
#define MOS3senGmbs MOS3sens + 56 /* contains perturbed values of gmbs*/
#define MOS3dphigs_dl MOS3sens + 62
#define MOS3dphigd_dl MOS3sens + 63
#define MOS3dphigb_dl MOS3sens + 64
#define MOS3dphibs_dl MOS3sens + 65
#define MOS3dphibd_dl MOS3sens + 66
#define MOS3dphigs_dw MOS3sens + 67
#define MOS3dphigd_dw MOS3sens + 68
#define MOS3dphigb_dw MOS3sens + 69
#define MOS3dphibs_dw MOS3sens + 70
#define MOS3dphibd_dw MOS3sens + 71

    /* 		distortion stuff 	*/
/*
 * naming convention:
 * x = vgs
 * y = vbs
 * z = vds
 * cdr = cdrain
 */


#define	MOS3NDCOEFFS	30

#ifndef NODISTO
	double MOS3dCoeffs[MOS3NDCOEFFS];
#else /* NODISTO */
	double *MOS3dCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	capbs2		MOS3dCoeffs[0]
#define	capbs3		MOS3dCoeffs[1]
#define	capbd2		MOS3dCoeffs[2]
#define	capbd3		MOS3dCoeffs[3]
#define	gbs2		MOS3dCoeffs[4]
#define	gbs3		MOS3dCoeffs[5]
#define	gbd2		MOS3dCoeffs[6]
#define	gbd3		MOS3dCoeffs[7]
#define	capgb2		MOS3dCoeffs[8]
#define	capgb3		MOS3dCoeffs[9]
#define	cdr_x2		MOS3dCoeffs[10]
#define	cdr_y2		MOS3dCoeffs[11]
#define	cdr_z2		MOS3dCoeffs[12]
#define	cdr_xy		MOS3dCoeffs[13]
#define	cdr_yz		MOS3dCoeffs[14]
#define	cdr_xz		MOS3dCoeffs[15]
#define	cdr_x3		MOS3dCoeffs[16]
#define	cdr_y3		MOS3dCoeffs[17]
#define	cdr_z3		MOS3dCoeffs[18]
#define	cdr_x2z		MOS3dCoeffs[19]
#define	cdr_x2y		MOS3dCoeffs[20]
#define	cdr_y2z		MOS3dCoeffs[21]
#define	cdr_xy2		MOS3dCoeffs[22]
#define	cdr_xz2		MOS3dCoeffs[23]
#define	cdr_yz2		MOS3dCoeffs[24]
#define	cdr_xyz		MOS3dCoeffs[25]
#define	capgs2		MOS3dCoeffs[26]
#define	capgs3		MOS3dCoeffs[27]
#define	capgd2		MOS3dCoeffs[28]
#define	capgd3		MOS3dCoeffs[29]

#endif

    /* 		end distortion coeffs.  	*/

#ifndef NONOISE
    double MOS3nVar[NSTATVARS][MOS3NSRCS];
#else /* NONOISE */
	double **MOS3nVar;
#endif /* NONOISE */

} MOS3instance ;

#define MOS3vbd MOS3states+ 0
#define MOS3vbs MOS3states+ 1
#define MOS3vgs MOS3states+ 2
#define MOS3vds MOS3states+ 3

/* meyer capacitances */
#define MOS3capgs MOS3states+ 4 /* gate-source capacitor value */
#define MOS3qgs MOS3states+ 5   /* gate-source capacitor charge */
#define MOS3cqgs MOS3states+ 6  /* gate-source capacitor current */

#define MOS3capgd MOS3states+ 7 /* gate-drain capacitor value */
#define MOS3qgd MOS3states+ 8   /* gate-drain capacitor charge */
#define MOS3cqgd MOS3states+ 9  /* gate-drain capacitor current */

#define MOS3capgb MOS3states+ 10/* gate-bulk capacitor value */
#define MOS3qgb MOS3states+ 11  /* gate-bulk capacitor charge */
#define MOS3cqgb MOS3states+ 12 /* gate-bulk capacitor current */

/* diode capacitances */
#define MOS3qbd MOS3states+ 13  /* bulk-drain capacitor charge */
#define MOS3cqbd MOS3states+ 14 /* bulk-drain capacitor current */

#define MOS3qbs MOS3states+ 15  /* bulk-source capacitor charge */
#define MOS3cqbs MOS3states+ 16 /* bulk-source capacitor current */ 

#define MOS3NUMSTATES 17


#define MOS3sensxpgs MOS3states+17 /* charge sensitivities and their derivatives
                                     +18 for the derivatives - pointer to the
                     beginning of the array */
#define MOS3sensxpgd  MOS3states+19
#define MOS3sensxpgb  MOS3states+21
#define MOS3sensxpbs  MOS3states+23
#define MOS3sensxpbd  MOS3states+25

#define MOS3numSenStates 10


/* per model data */

    /* NOTE:  parameters marked 'input - use xxxx' are paramters for
     * which a temperature correction is applied in MOS3temp, thus
     * the MOS3xxxx value in the per-instance structure should be used
     * instead in all calculations 
     */

typedef struct sMOS3model {       /* model structure for a resistor */

    struct GENmodel gen;

#define MOS3modType gen.GENmodType
#define MOS3nextModel(inst) ((struct sMOS3model *)((inst)->gen.GENnextModel))
#define MOS3instances(inst) ((MOS3instance *)((inst)->gen.GENinstances))
#define MOS3modName gen.GENmodName

    int MOS3type;       /* device type : 1 = nmos,  -1 = pmos */
    double MOS3tnom;        /* temperature at which parameters measured */
    double MOS3latDiff;
    double MOS3lengthAdjust;    /* New parm: mask adjustment to length */
    double MOS3widthNarrow;     /* New parm to reduce effective width */
    double MOS3widthAdjust;     /* New parm: mask adjustment to width */
    double MOS3delvt0;          /* New parm: adjustment calculated vtO */
    double MOS3jctSatCurDensity;    /* input - use tSatCurDens*/
    double MOS3jctSatCur;   /* input - use tSatCur instead */
    double MOS3drainResistance;
    double MOS3sourceResistance;
    double MOS3sheetResistance;
    double MOS3transconductance; /* input - use tTransconductance */
    double MOS3gateSourceOverlapCapFactor;
    double MOS3gateDrainOverlapCapFactor;
    double MOS3gateBulkOverlapCapFactor;
    double MOS3oxideCapFactor;
    double MOS3vt0; /* input - use tVto */
    double MOS3capBD;   /* input - use tCbs */
    double MOS3capBS;   /* input - use tCbd */
    double MOS3bulkCapFactor;   /* input - use tCj */
    double MOS3sideWallCapFactor;   /* input - use tCjsw */
    double MOS3bulkJctPotential;    /* input - use tBulkPot */
    double MOS3bulkJctBotGradingCoeff;
    double MOS3bulkJctSideGradingCoeff;
    double MOS3fwdCapDepCoeff;
    double MOS3phi; /* input - use tPhi */
    double MOS3gamma;
    double MOS3substrateDoping;
    int MOS3gateType;
    double MOS3surfaceStateDensity;
    double MOS3oxideThickness;
    double MOS3surfaceMobility; /* input - use tSurfMob */
    double MOS3eta;
    double MOS3junctionDepth;
    double MOS3coeffDepLayWidth; /* xd */
    double MOS3narrowFactor;    /* delta */
    double MOS3delta;   /* input delta */
    double MOS3fastSurfaceStateDensity; /* nfs */
    double MOS3theta;   /* theta */
    double MOS3maxDriftVel; /* vmax */
    double MOS3alpha;   /* alpha */
    double MOS3kappa;   /* kappa */
    double MOS3fNcoef;
    double MOS3fNexp;

    unsigned MOS3typeGiven  :1;
    unsigned MOS3latDiffGiven   :1;
    unsigned MOS3lengthAdjustGiven  :1;
    unsigned MOS3widthNarrowGiven   :1;
    unsigned MOS3widthAdjustGiven   :1;
    unsigned MOS3delvt0Given        :1;
    unsigned MOS3jctSatCurDensityGiven  :1;
    unsigned MOS3jctSatCurGiven :1;
    unsigned MOS3drainResistanceGiven   :1;
    unsigned MOS3sourceResistanceGiven  :1;
    unsigned MOS3sheetResistanceGiven   :1;
    unsigned MOS3transconductanceGiven  :1;
    unsigned MOS3gateSourceOverlapCapFactorGiven    :1;
    unsigned MOS3gateDrainOverlapCapFactorGiven :1;
    unsigned MOS3gateBulkOverlapCapFactorGiven  :1;
    unsigned MOS3vt0Given   :1;
    unsigned MOS3capBDGiven :1;
    unsigned MOS3capBSGiven :1;
    unsigned MOS3bulkCapFactorGiven :1;
    unsigned MOS3sideWallCapFactorGiven   :1;
    unsigned MOS3bulkJctPotentialGiven  :1;
    unsigned MOS3bulkJctBotGradingCoeffGiven    :1;
    unsigned MOS3bulkJctSideGradingCoeffGiven   :1;
    unsigned MOS3fwdCapDepCoeffGiven    :1;
    unsigned MOS3phiGiven   :1;
    unsigned MOS3gammaGiven :1;
    unsigned MOS3substrateDopingGiven   :1;
    unsigned MOS3gateTypeGiven  :1;
    unsigned MOS3surfaceStateDensityGiven   :1;
    unsigned MOS3oxideThicknessGiven    :1;
    unsigned MOS3surfaceMobilityGiven   :1;
    unsigned MOS3etaGiven   :1;
    unsigned MOS3junctionDepthGiven :1;
    unsigned MOS3deltaGiven :1; /* delta */
    unsigned MOS3fastSurfaceStateDensityGiven   :1; /* nfs */
    unsigned MOS3thetaGiven :1; /* theta */
    unsigned MOS3maxDriftVelGiven   :1; /* vmax */
    unsigned MOS3kappaGiven :1; /* kappa */
    unsigned MOS3tnomGiven :1;  /* Tnom was given? */
    unsigned MOS3fNcoefGiven :1;
    unsigned MOS3fNexpGiven :1;

} MOS3model;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

/* device parameters */
enum {
    MOS3_W = 1,
    MOS3_L,
    MOS3_AS,
    MOS3_AD,
    MOS3_PS,
    MOS3_PD,
    MOS3_NRS,
    MOS3_NRD,
    MOS3_OFF,
    MOS3_IC,
    MOS3_IC_VBS,
    MOS3_IC_VDS,
    MOS3_IC_VGS,
    MOS3_W_SENS,
    MOS3_L_SENS,
    MOS3_CB,
    MOS3_CG,
    MOS3_CS,
    MOS3_POWER,
    MOS3_CGS,
    MOS3_CGD,
    MOS3_DNODE,
    MOS3_GNODE,
    MOS3_SNODE,
    MOS3_BNODE,
    MOS3_DNODEPRIME,
    MOS3_SNODEPRIME,
    MOS3_SOURCECONDUCT,
    MOS3_DRAINCONDUCT,
    MOS3_VON,
    MOS3_VDSAT,
    MOS3_SOURCEVCRIT,
    MOS3_DRAINVCRIT,
    MOS3_CD,
    MOS3_CBS,
    MOS3_CBD,
    MOS3_GMBS,
    MOS3_GM,
    MOS3_GDS,
    MOS3_GBD,
    MOS3_GBS,
    MOS3_CAPBD,
    MOS3_CAPBS,
    MOS3_CAPZEROBIASBD,
    MOS3_CAPZEROBIASBDSW,
    MOS3_CAPZEROBIASBS,
    MOS3_CAPZEROBIASBSSW,
    MOS3_VBD,
    MOS3_VBS,
    MOS3_VGS,
    MOS3_VDS,
    MOS3_CAPGS,
    MOS3_QGS,
    MOS3_CQGS,
    MOS3_CAPGD,
    MOS3_QGD,
    MOS3_CQGD,
    MOS3_CAPGB,
    MOS3_QGB,
    MOS3_CQGB,
    MOS3_QBD,
    MOS3_CQBD,
    MOS3_QBS,
    MOS3_CQBS,
    MOS3_W_SENS_REAL,
    MOS3_W_SENS_IMAG,
    MOS3_W_SENS_MAG,
    MOS3_W_SENS_PH,
    MOS3_W_SENS_CPLX,
    MOS3_L_SENS_REAL,
    MOS3_L_SENS_IMAG,
    MOS3_L_SENS_MAG,
    MOS3_L_SENS_PH,
    MOS3_L_SENS_CPLX,
    MOS3_W_SENS_DC,
    MOS3_L_SENS_DC,
    MOS3_TEMP,
    MOS3_SOURCERESIST,
    MOS3_DRAINRESIST,
    MOS3_M,
    MOS3_DTEMP,
};

/* model parameters */
enum {
    MOS3_MOD_VTO = 101,
    MOS3_MOD_KP,
    MOS3_MOD_GAMMA,
    MOS3_MOD_PHI,
    MOS3_MOD_RD,
    MOS3_MOD_RS,
    MOS3_MOD_CBD,
    MOS3_MOD_CBS,
    MOS3_MOD_IS,
    MOS3_MOD_PB,
    MOS3_MOD_CGSO,
    MOS3_MOD_CGDO,
    MOS3_MOD_CGBO,
    MOS3_MOD_RSH,
    MOS3_MOD_CJ,
    MOS3_MOD_MJ,
    MOS3_MOD_CJSW,
    MOS3_MOD_MJSW,
    MOS3_MOD_JS,
    MOS3_MOD_TOX,
    MOS3_MOD_LD,
    MOS3_MOD_U0,
    MOS3_MOD_FC,
    MOS3_MOD_NSUB,
    MOS3_MOD_TPG,
    MOS3_MOD_NSS,
    MOS3_MOD_ETA,
    MOS3_MOD_DELTA,
    MOS3_MOD_NFS,
    MOS3_MOD_THETA,
    MOS3_MOD_VMAX,
    MOS3_MOD_KAPPA,
    MOS3_MOD_NMOS,
    MOS3_MOD_PMOS,
    MOS3_MOD_XJ,
    MOS3_MOD_UEXP,
    MOS3_MOD_NEFF,
    MOS3_MOD_XD,
    MOS3_MOD_ALPHA,
    MOS3_DELTA,
    MOS3_MOD_TNOM,
    MOS3_MOD_KF,
    MOS3_MOD_AF,
    MOS3_MOD_TYPE,
    MOS3_MOD_XL,
    MOS3_MOD_WD,
    MOS3_MOD_XW,
    MOS3_MOD_DELVTO,
};

/* device questions */


/* model questions */

#include "mos3ext.h"

#endif /*MOS3*/
