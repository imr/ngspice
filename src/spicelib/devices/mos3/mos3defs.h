/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlanFixes
**********/

#ifndef MOS3
#define MOS3

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

    /* declarations for level 3 MOSFETs */

/* information needed for each instance */

typedef struct sMOS3instance {
    struct sMOS3model *MOS3modPtr;  /* backpointer to model */
    struct sMOS3instance *MOS3nextInstance;  /* pointer to next instance of 
                                              *current model*/
    IFuid MOS3name; /* pointer to character string naming this instance */
    int MOS3owner;  /* number of owner process */
    int MOS3states;     /* index into state table for this device */
    int MOS3dNode;  /* number of the gate node of the mosfet */
    int MOS3gNode;  /* number of the gate node of the mosfet */
    int MOS3sNode;  /* number of the source node of the mosfet */
    int MOS3bNode;  /* number of the bulk node of the mosfet */
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
/* indices to the array of MOSFET(3) noise sources */

#define MOS3RDNOIZ       0
#define MOS3RSNOIZ       1
#define MOS3IDNOIZ       2
#define MOS3FLNOIZ 3
#define MOS3TOTNOIZ    4

#define MOS3NSRCS     5     /* the number of MOSFET(3) noise sources */

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
    int MOS3modType;    /* type index of this device type */
    struct sMOS3model *MOS3nextModel;    /* pointer to next possible model 
                                          *in linked list */
    MOS3instance * MOS3instances; /* pointer to list of instances 
                                   * that have this model */
    IFuid MOS3modName;       /* pointer to character string naming this model */
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
#define MOS3_W 1
#define MOS3_L 2
#define MOS3_AS 3
#define MOS3_AD 4
#define MOS3_PS 5
#define MOS3_PD 6
#define MOS3_NRS 7
#define MOS3_NRD 8
#define MOS3_OFF 9
#define MOS3_IC 10
#define MOS3_IC_VBS 11
#define MOS3_IC_VDS 12
#define MOS3_IC_VGS 13
#define MOS3_W_SENS 14
#define MOS3_L_SENS 15
#define MOS3_CB 16
#define MOS3_CG 17
#define MOS3_CS 18
#define MOS3_POWER 19
#define MOS3_CGS                20
#define MOS3_CGD                21
#define MOS3_DNODE              22
#define MOS3_GNODE              23
#define MOS3_SNODE              24
#define MOS3_BNODE              25
#define MOS3_DNODEPRIME         26
#define MOS3_SNODEPRIME         27
#define MOS3_SOURCECONDUCT      28
#define MOS3_DRAINCONDUCT       29
#define MOS3_VON                30
#define MOS3_VDSAT              31
#define MOS3_SOURCEVCRIT        32
#define MOS3_DRAINVCRIT         33
#define MOS3_CD                 34
#define MOS3_CBS                35
#define MOS3_CBD                36
#define MOS3_GMBS               37
#define MOS3_GM                 38
#define MOS3_GDS                39
#define MOS3_GBD                40
#define MOS3_GBS                41
#define MOS3_CAPBD              42
#define MOS3_CAPBS              43
#define MOS3_CAPZEROBIASBD      44
#define MOS3_CAPZEROBIASBDSW    45
#define MOS3_CAPZEROBIASBS      46
#define MOS3_CAPZEROBIASBSSW    47
#define MOS3_VBD                48
#define MOS3_VBS                49
#define MOS3_VGS                50
#define MOS3_VDS                51
#define MOS3_CAPGS              52
#define MOS3_QGS                53
#define MOS3_CQGS               54
#define MOS3_CAPGD              55
#define MOS3_QGD                56
#define MOS3_CQGD               57
#define MOS3_CAPGB              58
#define MOS3_QGB                59
#define MOS3_CQGB               60
#define MOS3_QBD                61
#define MOS3_CQBD               62
#define MOS3_QBS                63
#define MOS3_CQBS               64
#define MOS3_W_SENS_REAL        65
#define MOS3_W_SENS_IMAG        66
#define MOS3_W_SENS_MAG         67 
#define MOS3_W_SENS_PH          68
#define MOS3_W_SENS_CPLX        69
#define MOS3_L_SENS_REAL        70
#define MOS3_L_SENS_IMAG        71
#define MOS3_L_SENS_MAG         72
#define MOS3_L_SENS_PH          73
#define MOS3_L_SENS_CPLX        74
#define MOS3_W_SENS_DC          75
#define MOS3_L_SENS_DC          76
#define MOS3_TEMP               77
#define MOS3_SOURCERESIST       78
#define MOS3_DRAINRESIST        79
#define MOS3_M                  80
#define MOS3_DTEMP              81

/* model parameters */
#define MOS3_MOD_VTO 101
#define MOS3_MOD_KP 102
#define MOS3_MOD_GAMMA 103
#define MOS3_MOD_PHI 104
#define MOS3_MOD_RD 105
#define MOS3_MOD_RS 106
#define MOS3_MOD_CBD 107
#define MOS3_MOD_CBS 108
#define MOS3_MOD_IS 109
#define MOS3_MOD_PB 110
#define MOS3_MOD_CGSO 111
#define MOS3_MOD_CGDO 112
#define MOS3_MOD_CGBO 113
#define MOS3_MOD_RSH 114
#define MOS3_MOD_CJ 115
#define MOS3_MOD_MJ 116
#define MOS3_MOD_CJSW 117
#define MOS3_MOD_MJSW 118
#define MOS3_MOD_JS 119
#define MOS3_MOD_TOX 120
#define MOS3_MOD_LD 121
#define MOS3_MOD_U0 122
#define MOS3_MOD_FC 123
#define MOS3_MOD_NSUB 124
#define MOS3_MOD_TPG 125
#define MOS3_MOD_NSS 126
#define MOS3_MOD_ETA 127
#define MOS3_MOD_DELTA 128
#define MOS3_MOD_NFS 129
#define MOS3_MOD_THETA 130
#define MOS3_MOD_VMAX 131
#define MOS3_MOD_KAPPA 132
#define MOS3_MOD_NMOS 133
#define MOS3_MOD_PMOS 134
#define MOS3_MOD_XJ 135
#define MOS3_MOD_UEXP 136
#define MOS3_MOD_NEFF 137
#define MOS3_MOD_XD 138
#define MOS3_MOD_ALPHA 139
#define MOS3_DELTA 140
#define MOS3_MOD_TNOM 141
#define MOS3_MOD_KF 142
#define MOS3_MOD_AF 143
#define MOS3_MOD_TYPE 144

#define MOS3_MOD_XL 145
#define MOS3_MOD_WD 146
#define MOS3_MOD_XW 147
#define MOS3_MOD_DELVTO 148

/* device questions */


/* model questions */

#include "mos3ext.h"

#endif /*MOS3*/
