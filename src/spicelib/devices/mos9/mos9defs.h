/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#ifndef MOS9
#define MOS9

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

    /* declarations for level 9 MOSFETs */

/* indices to the array of MOSFET(9) noise sources */

enum {
    MOS9RDNOIZ = 0,
    MOS9RSNOIZ,
    MOS9IDNOIZ,
    MOS9FLNOIZ,
    MOS9TOTNOIZ,
    /* finally, the number of noise sources */
    MOS9NSRCS
};

/* information needed for each instance */

typedef struct sMOS9instance {

    struct GENinstance gen;

#define MOS9modPtr(inst) ((struct sMOS9model *)((inst)->gen.GENmodPtr))
#define MOS9nextInstance(inst) ((struct sMOS9instance *)((inst)->gen.GENnextInstance))
#define MOS9name gen.GENname
#define MOS9states gen.GENstate

    const int MOS9dNode;  /* number of the gate node of the mosfet */
    const int MOS9gNode;  /* number of the gate node of the mosfet */
    const int MOS9sNode;  /* number of the source node of the mosfet */
    const int MOS9bNode;  /* number of the bulk node of the mosfet */
    int MOS9dNodePrime; /* number of the internal drain node of the mosfet */
    int MOS9sNodePrime; /* number of the internal source node of the mosfet */

    double MOS9m;   /* parallel device multiplier */
    double MOS9l;   /* the length of the channel region */
    double MOS9w;   /* the width of the channel region */
    double MOS9drainArea;   /* the area of the drain diffusion */
    double MOS9sourceArea;  /* the area of the source diffusion */
    double MOS9drainSquares;    /* the length of the drain in squares */
    double MOS9sourceSquares;   /* the length of the source in squares */
    double MOS9drainPerimiter;
    double MOS9sourcePerimiter;
    double MOS9sourceConductance;   /*conductance of source(or 0):set in setup*/
    double MOS9drainConductance;    /*conductance of drain(or 0):set in setup*/
    double MOS9temp;    /* operating temperature of this instance */
    double MOS9dtemp;   /* instance temperature difference */

    double MOS9tTransconductance;   /* temperature corrected transconductance*/
    double MOS9tSurfMob;            /* temperature corrected surface mobility */
    double MOS9tPhi;                /* temperature corrected Phi */
    double MOS9tVto;                /* temperature corrected Vto */
    double MOS9tSatCur;             /* temperature corrected saturation Cur. */
    double MOS9tSatCurDens; /* temperature corrected saturation Cur. density*/
    double MOS9tCbd;                /* temperature corrected B-D Capacitance */
    double MOS9tCbs;                /* temperature corrected B-S Capacitance */
    double MOS9tCj;         /* temperature corrected Bulk bottom Capacitance */
    double MOS9tCjsw;       /* temperature corrected Bulk side Capacitance */
    double MOS9tBulkPot;    /* temperature corrected Bulk potential */
    double MOS9tDepCap;     /* temperature adjusted transition point in */
                            /* the cureve matching Fc * Vj */
    double MOS9tVbi;        /* temperature adjusted Vbi */

    double MOS9icVBS;   /* initial condition B-S voltage */
    double MOS9icVDS;   /* initial condition D-S voltage */
    double MOS9icVGS;   /* initial condition G-S voltage */
    double MOS9von;
    double MOS9vdsat;
    double MOS9sourceVcrit; /* vcrit for pos. vds */
    double MOS9drainVcrit;  /* vcrit for neg. vds */
    double MOS9cd;
    double MOS9cbs;
    double MOS9cbd;
    double MOS9gmbs;
    double MOS9gm;
    double MOS9gds;
    double MOS9gbd;
    double MOS9gbs;
    double MOS9capbd;
    double MOS9capbs;
    double MOS9Cbd;
    double MOS9Cbdsw;
    double MOS9Cbs;
    double MOS9Cbssw;
    double MOS9f2d;
    double MOS9f3d;
    double MOS9f4d;
    double MOS9f2s;
    double MOS9f3s;
    double MOS9f4s;
    int MOS9mode;       /* device mode : 1 = normal, -1 = inverse */


    unsigned MOS9off :1;/* non-zero to indicate device is off for dc analysis*/
    unsigned MOS9tempGiven :1;  /* instance temperature specified */
    unsigned MOS9dtempGiven :1; /* instance temperature difference specified*/

    unsigned MOS9mGiven :1;

    unsigned MOS9lGiven :1;
    unsigned MOS9wGiven :1;
    unsigned MOS9drainAreaGiven :1;
    unsigned MOS9sourceAreaGiven    :1;
    unsigned MOS9drainSquaresGiven  :1;
    unsigned MOS9sourceSquaresGiven :1;
    unsigned MOS9drainPerimiterGiven    :1;
    unsigned MOS9sourcePerimiterGiven   :1;
    unsigned MOS9dNodePrimeSet  :1;
    unsigned MOS9sNodePrimeSet  :1;
    unsigned MOS9icVBSGiven :1;
    unsigned MOS9icVDSGiven :1;
    unsigned MOS9icVGSGiven :1;
    unsigned MOS9vonGiven   :1;
    unsigned MOS9vdsatGiven :1;
    unsigned MOS9modeGiven  :1;


    double *MOS9DdPtr;      /* pointer to sparse matrix element at
                                     * (Drain node,drain node) */
    double *MOS9GgPtr;      /* pointer to sparse matrix element at
                                     * (gate node,gate node) */
    double *MOS9SsPtr;      /* pointer to sparse matrix element at
                                     * (source node,source node) */
    double *MOS9BbPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,bulk node) */
    double *MOS9DPdpPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,drain prime node) */
    double *MOS9SPspPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,source prime node) */
    double *MOS9DdpPtr;     /* pointer to sparse matrix element at
                                     * (drain node,drain prime node) */
    double *MOS9GbPtr;      /* pointer to sparse matrix element at
                                     * (gate node,bulk node) */
    double *MOS9GdpPtr;     /* pointer to sparse matrix element at
                                     * (gate node,drain prime node) */
    double *MOS9GspPtr;     /* pointer to sparse matrix element at
                                     * (gate node,source prime node) */
    double *MOS9SspPtr;     /* pointer to sparse matrix element at
                                     * (source node,source prime node) */
    double *MOS9BdpPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,drain prime node) */
    double *MOS9BspPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,source prime node) */
    double *MOS9DPspPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,source prime node) */
    double *MOS9DPdPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,drain node) */
    double *MOS9BgPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,gate node) */
    double *MOS9DPgPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,gate node) */

    double *MOS9SPgPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,gate node) */
    double *MOS9SPsPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,source node) */
    double *MOS9DPbPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,bulk node) */
    double *MOS9SPbPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,bulk node) */
    double *MOS9SPdpPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,drain prime node) */
    int  MOS9senParmNo;   /* parameter # for sensitivity use;
            set equal to 0  if  neither length
            nor width of the mosfet is a design
            parameter */
    unsigned MOS9sens_l :1;   /* field which indicates whether  
              length of the mosfet is a design
              parameter or not */
    unsigned MOS9sens_w :1;   /* field which indicates whether  
              width of the mosfet is a design
              parameter or not */
    unsigned MOS9senPertFlag :1; /* indictes whether the the parameter of
                        the particular instance is to be perturbed */
    double MOS9cgs;
    double MOS9cgd;
    double MOS9cgb;
    double *MOS9sens;

#define MOS9senGdpr MOS9sens
#define MOS9senGspr MOS9sens + 1
#define MOS9senCgs MOS9sens + 2 /* contains pertured values of cgs */
#define MOS9senCgd MOS9sens + 8 /* contains perturbed values of cgd*/
#define MOS9senCgb MOS9sens + 14 /* contains perturbed values of cgb*/
#define MOS9senCbd MOS9sens + 20 /* contains perturbed values of cbd*/
#define MOS9senCbs MOS9sens + 26 /* contains perturbed values of cbs*/
#define MOS9senGds MOS9sens + 32 /* contains perturbed values of gds*/
#define MOS9senGbs MOS9sens + 38 /* contains perturbed values of gbs*/
#define MOS9senGbd MOS9sens + 44 /* contains perturbed values of gbd*/
#define MOS9senGm MOS9sens + 50 /* contains perturbed values of gm*/
#define MOS9senGmbs MOS9sens + 56 /* contains perturbed values of gmbs*/
#define MOS9dphigs_dl MOS9sens + 62
#define MOS9dphigd_dl MOS9sens + 63
#define MOS9dphigb_dl MOS9sens + 64
#define MOS9dphibs_dl MOS9sens + 65
#define MOS9dphibd_dl MOS9sens + 66
#define MOS9dphigs_dw MOS9sens + 67
#define MOS9dphigd_dw MOS9sens + 68
#define MOS9dphigb_dw MOS9sens + 69
#define MOS9dphibs_dw MOS9sens + 70
#define MOS9dphibd_dw MOS9sens + 71

    /* 		distortion stuff 	*/
/*
 * naming convention:
 * x = vgs
 * y = vbs
 * z = vds
 * cdr = cdrain
 */


#define	MOS9NDCOEFFS	30

#ifndef NODISTO
	double MOS9dCoeffs[MOS9NDCOEFFS];
#else /* NODISTO */
	double *MOS9dCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	capbs2		MOS9dCoeffs[0]
#define	capbs3		MOS9dCoeffs[1]
#define	capbd2		MOS9dCoeffs[2]
#define	capbd3		MOS9dCoeffs[3]
#define	gbs2		MOS9dCoeffs[4]
#define	gbs3		MOS9dCoeffs[5]
#define	gbd2		MOS9dCoeffs[6]
#define	gbd3		MOS9dCoeffs[7]
#define	capgb2		MOS9dCoeffs[8]
#define	capgb3		MOS9dCoeffs[9]
#define	cdr_x2		MOS9dCoeffs[10]
#define	cdr_y2		MOS9dCoeffs[11]
#define	cdr_z2		MOS9dCoeffs[12]
#define	cdr_xy		MOS9dCoeffs[13]
#define	cdr_yz		MOS9dCoeffs[14]
#define	cdr_xz		MOS9dCoeffs[15]
#define	cdr_x3		MOS9dCoeffs[16]
#define	cdr_y3		MOS9dCoeffs[17]
#define	cdr_z3		MOS9dCoeffs[18]
#define	cdr_x2z		MOS9dCoeffs[19]
#define	cdr_x2y		MOS9dCoeffs[20]
#define	cdr_y2z		MOS9dCoeffs[21]
#define	cdr_xy2		MOS9dCoeffs[22]
#define	cdr_xz2		MOS9dCoeffs[23]
#define	cdr_yz2		MOS9dCoeffs[24]
#define	cdr_xyz		MOS9dCoeffs[25]
#define	capgs2		MOS9dCoeffs[26]
#define	capgs3		MOS9dCoeffs[27]
#define	capgd2		MOS9dCoeffs[28]
#define	capgd3		MOS9dCoeffs[29]

#endif

    /* 		end distortion coeffs.  	*/

#ifndef NONOISE
    double MOS9nVar[NSTATVARS][MOS9NSRCS];
#else /* NONOISE */
	double **MOS9nVar;
#endif /* NONOISE */

} MOS9instance ;

#define MOS9vbd MOS9states+ 0
#define MOS9vbs MOS9states+ 1
#define MOS9vgs MOS9states+ 2
#define MOS9vds MOS9states+ 3

/* meyer capacitances */
#define MOS9capgs MOS9states+ 4 /* gate-source capacitor value */
#define MOS9qgs MOS9states+ 5   /* gate-source capacitor charge */
#define MOS9cqgs MOS9states+ 6  /* gate-source capacitor current */

#define MOS9capgd MOS9states+ 7 /* gate-drain capacitor value */
#define MOS9qgd MOS9states+ 8   /* gate-drain capacitor charge */
#define MOS9cqgd MOS9states+ 9  /* gate-drain capacitor current */

#define MOS9capgb MOS9states+ 10/* gate-bulk capacitor value */
#define MOS9qgb MOS9states+ 11  /* gate-bulk capacitor charge */
#define MOS9cqgb MOS9states+ 12 /* gate-bulk capacitor current */

/* diode capacitances */
#define MOS9qbd MOS9states+ 13  /* bulk-drain capacitor charge */
#define MOS9cqbd MOS9states+ 14 /* bulk-drain capacitor current */

#define MOS9qbs MOS9states+ 15  /* bulk-source capacitor charge */
#define MOS9cqbs MOS9states+ 16 /* bulk-source capacitor current */ 

#define MOS9NUMSTATES 17


#define MOS9sensxpgs MOS9states+17 /* charge sensitivities and their derivatives
                                     +18 for the derivatives - pointer to the
                     beginning of the array */
#define MOS9sensxpgd  MOS9states+19
#define MOS9sensxpgb  MOS9states+21
#define MOS9sensxpbs  MOS9states+23
#define MOS9sensxpbd  MOS9states+25

#define MOS9numSenStates 10


/* per model data */

    /* NOTE:  parameters marked 'input - use xxxx' are paramters for
     * which a temperature correction is applied in MOS9temp, thus
     * the MOS9xxxx value in the per-instance structure should be used
     * instead in all calculations 
     */

typedef struct sMOS9model {       /* model structure for a resistor */

    struct GENmodel gen;

#define MOS9modType gen.GENmodType
#define MOS9nextModel(inst) ((struct sMOS9model *)((inst)->gen.GENnextModel))
#define MOS9instances(inst) ((MOS9instance *)((inst)->gen.GENinstances))
#define MOS9modName gen.GENmodName

    int MOS9type;       /* device type : 1 = nmos,  -1 = pmos */
    double MOS9tnom;        /* temperature at which parameters measured */
    double MOS9latDiff;
    double MOS9lengthAdjust;    /* New parm: mask adjustment to length */
    double MOS9widthNarrow;     /* New parm to reduce effective width */
    double MOS9widthAdjust;     /* New parm: mask adjustment to width */
    double MOS9delvt0;          /* New parm: adjustment to calculated vtO */
    double MOS9jctSatCurDensity;    /* input - use tSatCurDens*/
    double MOS9jctSatCur;   /* input - use tSatCur instead */
    double MOS9drainResistance;
    double MOS9sourceResistance;
    double MOS9sheetResistance;
    double MOS9transconductance; /* input - use tTransconductance */
    double MOS9gateSourceOverlapCapFactor;
    double MOS9gateDrainOverlapCapFactor;
    double MOS9gateBulkOverlapCapFactor;
    double MOS9oxideCapFactor;
    double MOS9vt0; /* input - use tVto */
    double MOS9capBD;   /* input - use tCbs */
    double MOS9capBS;   /* input - use tCbd */
    double MOS9bulkCapFactor;   /* input - use tCj */
    double MOS9sideWallCapFactor;   /* input - use tCjsw */
    double MOS9bulkJctPotential;    /* input - use tBulkPot */
    double MOS9bulkJctBotGradingCoeff;
    double MOS9bulkJctSideGradingCoeff;
    double MOS9fwdCapDepCoeff;
    double MOS9phi; /* input - use tPhi */
    double MOS9gamma;
    double MOS9substrateDoping;
    int MOS9gateType;
    double MOS9surfaceStateDensity;
    double MOS9oxideThickness;
    double MOS9surfaceMobility; /* input - use tSurfMob */
    double MOS9eta;
    double MOS9junctionDepth;
    double MOS9coeffDepLayWidth; /* xd */
    double MOS9narrowFactor;    /* delta */
    double MOS9delta;   /* input delta */
    double MOS9fastSurfaceStateDensity; /* nfs */
    double MOS9theta;   /* theta */
    double MOS9maxDriftVel; /* vmax */
    double MOS9alpha;   /* alpha */
    double MOS9kappa;   /* kappa */
    double MOS9fNcoef;
    double MOS9fNexp;

    unsigned MOS9typeGiven  :1;
    unsigned MOS9latDiffGiven   :1;
    unsigned MOS9lengthAdjustGiven  :1;
    unsigned MOS9widthNarrowGiven   :1;
    unsigned MOS9widthAdjustGiven   :1;
    unsigned MOS9delvt0Given        :1;
    unsigned MOS9jctSatCurDensityGiven  :1;
    unsigned MOS9jctSatCurGiven :1;
    unsigned MOS9drainResistanceGiven   :1;
    unsigned MOS9sourceResistanceGiven  :1;
    unsigned MOS9sheetResistanceGiven   :1;
    unsigned MOS9transconductanceGiven  :1;
    unsigned MOS9gateSourceOverlapCapFactorGiven    :1;
    unsigned MOS9gateDrainOverlapCapFactorGiven :1;
    unsigned MOS9gateBulkOverlapCapFactorGiven  :1;
    unsigned MOS9vt0Given   :1;
    unsigned MOS9capBDGiven :1;
    unsigned MOS9capBSGiven :1;
    unsigned MOS9bulkCapFactorGiven :1;
    unsigned MOS9sideWallCapFactorGiven   :1;
    unsigned MOS9bulkJctPotentialGiven  :1;
    unsigned MOS9bulkJctBotGradingCoeffGiven    :1;
    unsigned MOS9bulkJctSideGradingCoeffGiven   :1;
    unsigned MOS9fwdCapDepCoeffGiven    :1;
    unsigned MOS9phiGiven   :1;
    unsigned MOS9gammaGiven :1;
    unsigned MOS9substrateDopingGiven   :1;
    unsigned MOS9gateTypeGiven  :1;
    unsigned MOS9surfaceStateDensityGiven   :1;
    unsigned MOS9oxideThicknessGiven    :1;
    unsigned MOS9surfaceMobilityGiven   :1;
    unsigned MOS9etaGiven   :1;
    unsigned MOS9junctionDepthGiven :1;
    unsigned MOS9deltaGiven :1; /* delta */
    unsigned MOS9fastSurfaceStateDensityGiven   :1; /* nfs */
    unsigned MOS9thetaGiven :1; /* theta */
    unsigned MOS9maxDriftVelGiven   :1; /* vmax */
    unsigned MOS9kappaGiven :1; /* kappa */
    unsigned MOS9tnomGiven :1;  /* Tnom was given? */
    unsigned MOS9fNcoefGiven :1;
    unsigned MOS9fNexpGiven :1;

} MOS9model;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

/* device parameters */
enum {
    MOS9_W = 1,
    MOS9_L,
    MOS9_AS,
    MOS9_AD,
    MOS9_PS,
    MOS9_PD,
    MOS9_NRS,
    MOS9_NRD,
    MOS9_OFF,
    MOS9_IC,
    MOS9_IC_VBS,
    MOS9_IC_VDS,
    MOS9_IC_VGS,
    MOS9_W_SENS,
    MOS9_L_SENS,
    MOS9_CB,
    MOS9_CG,
    MOS9_CS,
    MOS9_POWER,
    MOS9_CGS,
    MOS9_CGD,
    MOS9_DNODE,
    MOS9_GNODE,
    MOS9_SNODE,
    MOS9_BNODE,
    MOS9_DNODEPRIME,
    MOS9_SNODEPRIME,
    MOS9_SOURCECONDUCT,
    MOS9_DRAINCONDUCT,
    MOS9_VON,
    MOS9_VDSAT,
    MOS9_SOURCEVCRIT,
    MOS9_DRAINVCRIT,
    MOS9_CD,
    MOS9_CBS,
    MOS9_CBD,
    MOS9_GMBS,
    MOS9_GM,
    MOS9_GDS,
    MOS9_GBD,
    MOS9_GBS,
    MOS9_CAPBD,
    MOS9_CAPBS,
    MOS9_CAPZEROBIASBD,
    MOS9_CAPZEROBIASBDSW,
    MOS9_CAPZEROBIASBS,
    MOS9_CAPZEROBIASBSSW,
    MOS9_VBD,
    MOS9_VBS,
    MOS9_VGS,
    MOS9_VDS,
    MOS9_CAPGS,
    MOS9_QGS,
    MOS9_CQGS,
    MOS9_CAPGD,
    MOS9_QGD,
    MOS9_CQGD,
    MOS9_CAPGB,
    MOS9_QGB,
    MOS9_CQGB,
    MOS9_QBD,
    MOS9_CQBD,
    MOS9_QBS,
    MOS9_CQBS,
    MOS9_W_SENS_REAL,
    MOS9_W_SENS_IMAG,
    MOS9_W_SENS_MAG,
    MOS9_W_SENS_PH,
    MOS9_W_SENS_CPLX,
    MOS9_L_SENS_REAL,
    MOS9_L_SENS_IMAG,
    MOS9_L_SENS_MAG,
    MOS9_L_SENS_PH,
    MOS9_L_SENS_CPLX,
    MOS9_W_SENS_DC,
    MOS9_L_SENS_DC,
    MOS9_TEMP,
    MOS9_SOURCERESIST,
    MOS9_DRAINRESIST,
    MOS9_M,
    MOS9_DTEMP,
};

/* model parameters */
enum {
    MOS9_MOD_VTO = 101,
    MOS9_MOD_KP,
    MOS9_MOD_GAMMA,
    MOS9_MOD_PHI,
    MOS9_MOD_RD,
    MOS9_MOD_RS,
    MOS9_MOD_CBD,
    MOS9_MOD_CBS,
    MOS9_MOD_IS,
    MOS9_MOD_PB,
    MOS9_MOD_CGSO,
    MOS9_MOD_CGDO,
    MOS9_MOD_CGBO,
    MOS9_MOD_RSH,
    MOS9_MOD_CJ,
    MOS9_MOD_MJ,
    MOS9_MOD_CJSW,
    MOS9_MOD_MJSW,
    MOS9_MOD_JS,
    MOS9_MOD_TOX,
    MOS9_MOD_LD,
    MOS9_MOD_U0,
    MOS9_MOD_FC,
    MOS9_MOD_NSUB,
    MOS9_MOD_TPG,
    MOS9_MOD_NSS,
    MOS9_MOD_ETA,
    MOS9_MOD_DELTA,
    MOS9_MOD_NFS,
    MOS9_MOD_THETA,
    MOS9_MOD_VMAX,
    MOS9_MOD_KAPPA,
    MOS9_MOD_NMOS,
    MOS9_MOD_PMOS,
    MOS9_MOD_XJ,
    MOS9_MOD_UEXP,
    MOS9_MOD_NEFF,
    MOS9_MOD_XD,
    MOS9_MOD_ALPHA,
    MOS9_DELTA,
    MOS9_MOD_TNOM,
    MOS9_MOD_KF,
    MOS9_MOD_AF,
    MOS9_MOD_TYPE,
    MOS9_MOD_XL,
    MOS9_MOD_WD,
    MOS9_MOD_XW,
    MOS9_MOD_DELVTO,
};

/* device questions */


/* model questions */

#include "mos9ext.h"

#endif /*MOS9*/

