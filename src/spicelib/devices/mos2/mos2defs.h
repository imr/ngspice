/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFIxes
**********/

#ifndef MOS2
#define MOS2

#include "ifsim.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

/* declarations for level 2 MOSFETs */

/* information needed for each instance */

typedef struct sMOS2instance {
    struct sMOS2model *MOS2modPtr;  /* backpointer to model */
    struct sMOS2instance *MOS2nextInstance;  /* pointer to next instance of
                                              *current model*/
    IFuid MOS2name; /* pointer to character string naming this instance */
    int MOS2owner;  /* number of owner process */
    int MOS2states;     /* index into state table for this device */
    int MOS2dNode;  /* number of the gate node of the mosfet */
    int MOS2gNode;  /* number of the gate node of the mosfet */
    int MOS2sNode;  /* number of the source node of the mosfet */
    int MOS2bNode;  /* number of the bulk node of the mosfet */
    int MOS2dNodePrime; /* number of the internal drain node of the mosfet */
    int MOS2sNodePrime; /* number of the internal source node of the mosfet */


    int MOS2mode;       /* device mode : 1 = normal, -1 = inverse */

    unsigned MOS2mGiven :1;

    unsigned MOS2off :1;/* non-zero to indicate device is off for dc analysis*/
    unsigned MOS2lGiven :1;
    unsigned MOS2wGiven :1;
    unsigned MOS2drainAreaGiven :1;
    unsigned MOS2sourceAreaGiven    :1;
    unsigned MOS2drainSquaresGiven  :1;
    unsigned MOS2sourceSquaresGiven :1;
    unsigned MOS2drainPerimiterGiven    :1;
    unsigned MOS2sourcePerimiterGiven   :1;
    unsigned MOS2dNodePrimeSet  :1;
    unsigned MOS2sNodePrimeSet  :1;
    unsigned MOS2icVBSGiven :1;
    unsigned MOS2icVDSGiven :1;
    unsigned MOS2icVGSGiven :1;
    unsigned MOS2vonGiven   :1;
    unsigned MOS2vdsatGiven :1;
    unsigned MOS2tempGiven  :1; /* per-instance temperature specified? */
    unsigned MOS2dtempGiven :1; /* per-instance temperature difference specified? */
    unsigned MOS2sens_l :1;   /* field which indicates whether  
                                  length of the mosfet is a design
                                  parameter or not */
    unsigned MOS2sens_w :1;   /* field which indicates whether  
                                  width of the mosfet is a design
                                  parameter or not */
    unsigned MOS2senPertFlag :1; /* indictes whether the the parameter of
                            the particular instance is to be perturbed */


    double *MOS2DdPtr;      /* pointer to sparse matrix element at
                                     * (Drain node,drain node) */
    double *MOS2GgPtr;      /* pointer to sparse matrix element at
                                     * (gate node,gate node) */
    double *MOS2SsPtr;      /* pointer to sparse matrix element at
                                     * (source node,source node) */
    double *MOS2BbPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,bulk node) */
    double *MOS2DPdpPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,drain prime node) */
    double *MOS2SPspPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,source prime node) */
    double *MOS2DdpPtr;     /* pointer to sparse matrix element at
                                     * (drain node,drain prime node) */
    double *MOS2GbPtr;      /* pointer to sparse matrix element at
                                     * (gate node,bulk node) */
    double *MOS2GdpPtr;     /* pointer to sparse matrix element at
                                     * (gate node,drain prime node) */
    double *MOS2GspPtr;     /* pointer to sparse matrix element at
                                     * (gate node,source prime node) */
    double *MOS2SspPtr;     /* pointer to sparse matrix element at
                                     * (source node,source prime node) */
    double *MOS2BdpPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,drain prime node) */
    double *MOS2BspPtr;     /* pointer to sparse matrix element at
                                     * (bulk node,source prime node) */
    double *MOS2DPspPtr;    /* pointer to sparse matrix element at
                                     * (drain prime node,source prime node) */
    double *MOS2DPdPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,drain node) */
    double *MOS2BgPtr;      /* pointer to sparse matrix element at
                                     * (bulk node,gate node) */
    double *MOS2DPgPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,gate node) */

    double *MOS2SPgPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,gate node) */
    double *MOS2SPsPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,source node) */
    double *MOS2DPbPtr;     /* pointer to sparse matrix element at
                                     * (drain prime node,bulk node) */
    double *MOS2SPbPtr;     /* pointer to sparse matrix element at
                                     * (source prime node,bulk node) */
    double *MOS2SPdpPtr;    /* pointer to sparse matrix element at
                                     * (source prime node,drain prime node) */
    int  MOS2senParmNo;   /* parameter # for sensitivity use;
            set equal to 0  if  neither length
            nor width of the mosfet is a design
            parameter */
    double MOS2cgs;
    double MOS2cgd;
    double MOS2cgb;
    double *MOS2sens;

#define MOS2senCgs MOS2sens /* contains pertured values of cgs */
#define MOS2senCgd MOS2sens + 6 /* contains perturbed values of cgd*/
#define MOS2senCgb MOS2sens + 12 /* contains perturbed values of cgb*/
#define MOS2senCbd MOS2sens + 18 /* contains perturbed values of cbd*/
#define MOS2senCbs MOS2sens + 24 /* contains perturbed values of cbs*/
#define MOS2senGds MOS2sens + 30 /* contains perturbed values of gds*/
#define MOS2senGbs MOS2sens + 36 /* contains perturbed values of gbs*/
#define MOS2senGbd MOS2sens + 42 /* contains perturbed values of gbd*/
#define MOS2senGm MOS2sens + 48 /* contains perturbed values of gm*/
#define MOS2senGmbs MOS2sens + 54 /* contains perturbed values of gmbs*/
#define MOS2dphigs_dl MOS2sens + 60
#define MOS2dphigd_dl MOS2sens + 61
#define MOS2dphigb_dl MOS2sens + 62
#define MOS2dphibs_dl MOS2sens + 63
#define MOS2dphibd_dl MOS2sens + 64
#define MOS2dphigs_dw MOS2sens + 65
#define MOS2dphigd_dw MOS2sens + 66
#define MOS2dphigb_dw MOS2sens + 67
#define MOS2dphibs_dw MOS2sens + 68
#define MOS2dphibd_dw MOS2sens + 69

    double MOS2temp;        /* temperature at which this instance operates */
    double MOS2dtemp;       /* difference of instance temperature from circuit temperature */
    double MOS2tTransconductance;   /* temperature corrected transconductance*/
    double MOS2tSurfMob;            /* temperature corrected surface mobility */
    double MOS2tPhi;                /* temperature corrected Phi */
    double MOS2tVto;                /* temperature corrected Vto */
    double MOS2tSatCur;             /* temperature corrected saturation Cur. */
    double MOS2tSatCurDens; /* temperature corrected saturation Cur. density*/
    double MOS2tCbd;                /* temperature corrected B-D Capacitance */
    double MOS2tCbs;                /* temperature corrected B-S Capacitance */
    double MOS2tCj;         /* temperature corrected Bulk bottom Capacitance */
    double MOS2tCjsw;       /* temperature corrected Bulk side Capacitance */
    double MOS2tBulkPot;    /* temperature corrected Bulk potential */
    double MOS2tDepCap;     /* temperature adjusted transition point in */
                            /* the cureve matching Fc * Vj */
    double MOS2tVbi;        /* temperature adjusted Vbi */

    double MOS2m;   /* parallel device multiplier */

    double MOS2l;   /* the length of the channel region */
    double MOS2w;   /* the width of the channel region */
    double MOS2drainArea;   /* the area of the drain diffusion */
    double MOS2sourceArea;  /* the area of the source diffusion */
    double MOS2drainSquares;    /* the length of the drain in squares */
    double MOS2sourceSquares;   /* the length of the source in squares */
    double MOS2drainPerimiter;
    double MOS2sourcePerimiter;
    double MOS2sourceConductance;   /*conductance of source(or 0):set in setup*/
    double MOS2drainConductance;    /*conductance of drain(or 0):set in setup*/

    double MOS2icVBS;   /* initial condition B-S voltage */
    double MOS2icVDS;   /* initial condition D-S voltage */
    double MOS2icVGS;   /* initial condition G-S voltage */
    double MOS2von;
    double MOS2vdsat;
    double MOS2sourceVcrit; /* Vcrit for pos. vds */
    double MOS2drainVcrit;  /* Vcrit for pos. vds */
    double MOS2cd;
    double MOS2cbs;
    double MOS2cbd;
    double MOS2gmbs;
    double MOS2gm;
    double MOS2gds;
    double MOS2gbd;
    double MOS2gbs;
    double MOS2capbd;
    double MOS2capbs;
    double MOS2Cbd;
    double MOS2Cbdsw;
    double MOS2Cbs;
    double MOS2Cbssw;
    double MOS2f2d;
    double MOS2f3d;
    double MOS2f4d;
    double MOS2f2s;
    double MOS2f3s;
    double MOS2f4s;

    /* 		distortion stuff 	*/
/*
 * naming convention:
 * x = vgs
 * y = vbs
 * z = vds
 * cdr = cdrain
 */



#define	MOS2NDCOEFFS	30

#ifndef NODISTO
	double MOS2dCoeffs[MOS2NDCOEFFS];
#else /* NODISTO */
	double *MOS2dCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	capbs2		MOS2dCoeffs[0]
#define	capbs3		MOS2dCoeffs[1]
#define	capbd2		MOS2dCoeffs[2]
#define	capbd3		MOS2dCoeffs[3]
#define	gbs2		MOS2dCoeffs[4]
#define	gbs3		MOS2dCoeffs[5]
#define	gbd2		MOS2dCoeffs[6]
#define	gbd3		MOS2dCoeffs[7]
#define	capgb2		MOS2dCoeffs[8]
#define	capgb3		MOS2dCoeffs[9]
#define	cdr_x2		MOS2dCoeffs[10]
#define	cdr_y2		MOS2dCoeffs[11]
#define	cdr_z2		MOS2dCoeffs[12]
#define	cdr_xy		MOS2dCoeffs[13]
#define	cdr_yz		MOS2dCoeffs[14]
#define	cdr_xz		MOS2dCoeffs[15]
#define	cdr_x3		MOS2dCoeffs[16]
#define	cdr_y3		MOS2dCoeffs[17]
#define	cdr_z3		MOS2dCoeffs[18]
#define	cdr_x2z		MOS2dCoeffs[19]
#define	cdr_x2y		MOS2dCoeffs[20]
#define	cdr_y2z		MOS2dCoeffs[21]
#define	cdr_xy2		MOS2dCoeffs[22]
#define	cdr_xz2		MOS2dCoeffs[23]
#define	cdr_yz2		MOS2dCoeffs[24]
#define	cdr_xyz		MOS2dCoeffs[25]
#define	capgs2		MOS2dCoeffs[26]
#define	capgs3		MOS2dCoeffs[27]
#define	capgd2		MOS2dCoeffs[28]
#define	capgd3		MOS2dCoeffs[29]
    /* 		end distortion coeffs.  	*/

#endif

/* indices to the array of MOSFET(2) noise sources */

#define MOS2RDNOIZ       0
#define MOS2RSNOIZ       1
#define MOS2IDNOIZ       2
#define MOS2FLNOIZ 3
#define MOS2TOTNOIZ    4

#define MOS2NSRCS     5     /* the number of MOS2FET noise sources */

#ifndef NONOISE
    double MOS2nVar[NSTATVARS][MOS2NSRCS];
#else /* NONOISE */
	double **MOS2nVar;
#endif /* NONOISE */

    

} MOS2instance ;


#define MOS2vbd MOS2states+ 0   /* bulk-drain voltage */
#define MOS2vbs MOS2states+ 1   /* bulk-source voltage */
#define MOS2vgs MOS2states+ 2   /* gate-source voltage */
#define MOS2vds MOS2states+ 3   /* drain-source voltage */

#define MOS2capgs MOS2states+4  /* gate-source capacitor value */
#define MOS2qgs MOS2states+ 5   /* gate-source capacitor charge */
#define MOS2cqgs MOS2states+ 6  /* gate-source capacitor current */

#define MOS2capgd MOS2states+ 7 /* gate-drain capacitor value */
#define MOS2qgd MOS2states+ 8   /* gate-drain capacitor charge */
#define MOS2cqgd MOS2states+ 9  /* gate-drain capacitor current */

#define MOS2capgb MOS2states+10 /* gate-bulk capacitor value */
#define MOS2qgb MOS2states+ 11  /* gate-bulk capacitor charge */
#define MOS2cqgb MOS2states+ 12 /* gate-bulk capacitor current */

#define MOS2qbd MOS2states+ 13  /* bulk-drain capacitor charge */
#define MOS2cqbd MOS2states+ 14 /* bulk-drain capacitor current */

#define MOS2qbs MOS2states+ 15  /* bulk-source capacitor charge */
#define MOS2cqbs MOS2states+ 16 /* bulk-source capacitor current */

#define MOS2numStates 17


#define MOS2sensxpgs MOS2states+17 /* charge sensitivities and their derivatives
                                     +18 for the derivatives - pointer to the
                     beginning of the array */

#define MOS2sensxpgd  MOS2states+19
#define MOS2sensxpgb  MOS2states+21
#define MOS2sensxpbs  MOS2states+23
#define MOS2sensxpbd  MOS2states+25

#define MOS2numSenStates 10


/* per model data */


        /* NOTE:  parameters makred 'input - use xxxx' are parameters for
         * which a temperature correction is applied in MOS2temp, thus
         * the MOS3xxxx value in the per-instance structure should be used
         * instead in all calculations
         */

typedef struct sMOS2model {       /* model structure for a resistor */
    int MOS2modType;    /* type index of this device type */
    struct sMOS2model *MOS2nextModel;    /* pointer to next possible model 
                                          *in linked list */
    MOS2instance * MOS2instances; /* pointer to list of instances 
                                   * that have this model */
    IFuid MOS2modName;       /* pointer to character string naming this model */
    int MOS2type;       /* device type : 1 = nmos,  -1 = pmos */
    int MOS2gateType;

    double MOS2tnom;    /* temperature at which parms were measured */
    double MOS2latDiff;
    double MOS2jctSatCurDensity;    /* input - use tSatCurDens */
    double MOS2jctSatCur;   /* input - use tSatCur */
    double MOS2drainResistance;
    double MOS2sourceResistance;
    double MOS2sheetResistance;
    double MOS2transconductance;    /* input - use tTransconductance */
    double MOS2gateSourceOverlapCapFactor;
    double MOS2gateDrainOverlapCapFactor;
    double MOS2gateBulkOverlapCapFactor;
    double MOS2oxideCapFactor;
    double MOS2vt0; /* input - use tVto */
    double MOS2capBD;   /* input - use tCbd */
    double MOS2capBS;   /* input - use tCbs */
    double MOS2bulkCapFactor;   /* input - use tCj */
    double MOS2sideWallCapFactor;   /* input - use tCjsw */
    double MOS2bulkJctPotential;    /* input - use tBulkPot */
    double MOS2bulkJctBotGradingCoeff;
    double MOS2bulkJctSideGradingCoeff;
    double MOS2fwdCapDepCoeff;
    double MOS2phi;     /* input - use tPhi */
    double MOS2gamma;
    double MOS2lambda;
    double MOS2substrateDoping;
    double MOS2surfaceStateDensity;
    double MOS2fastSurfaceStateDensity; /* nfs */
    double MOS2oxideThickness;
    double MOS2surfaceMobility;
    double MOS2fNcoef;
    double MOS2fNexp;

    double MOS2narrowFactor;    /* delta */
    double MOS2critFieldExp;    /* uexp */
    double MOS2critField;   /* ucrit */
    double MOS2maxDriftVel; /* vmax */
    double MOS2xd;
    double MOS2junctionDepth;   /* xj */
    double MOS2channelCharge;   /* neff */

    unsigned MOS2tnomGiven  :1; /* user specified parm. meas. temp */
    unsigned MOS2typeGiven  :1;
    unsigned MOS2latDiffGiven   :1;
    unsigned MOS2jctSatCurDensityGiven  :1;
    unsigned MOS2jctSatCurGiven :1;
    unsigned MOS2drainResistanceGiven   :1;
    unsigned MOS2sourceResistanceGiven  :1;
    unsigned MOS2sheetResistanceGiven   :1;
    unsigned MOS2transconductanceGiven  :1;
    unsigned MOS2gateSourceOverlapCapFactorGiven    :1;
    unsigned MOS2gateDrainOverlapCapFactorGiven :1;
    unsigned MOS2gateBulkOverlapCapFactorGiven  :1;
    unsigned MOS2vt0Given   :1;
    unsigned MOS2capBDGiven :1;
    unsigned MOS2capBSGiven :1;
    unsigned MOS2bulkCapFactorGiven :1;
    unsigned MOS2sideWallCapFactorGiven   :1;
    unsigned MOS2bulkJctPotentialGiven  :1;
    unsigned MOS2bulkJctBotGradingCoeffGiven    :1;
    unsigned MOS2bulkJctSideGradingCoeffGiven   :1;
    unsigned MOS2fwdCapDepCoeffGiven    :1;
    unsigned MOS2phiGiven   :1;
    unsigned MOS2gammaGiven :1;
    unsigned MOS2lambdaGiven    :1;
    unsigned MOS2substrateDopingGiven   :1;
    unsigned MOS2gateTypeGiven  :1;
    unsigned MOS2surfaceStateDensityGiven   :1;
    unsigned MOS2fastSurfaceStateDensityGiven   :1; /* nfs */
    unsigned MOS2oxideThicknessGiven    :1;
    unsigned MOS2surfaceMobilityGiven   :1;
    unsigned MOS2narrowFactorGiven  :1; /* delta */
    unsigned MOS2critFieldExpGiven  :1; /* uexp */
    unsigned MOS2critFieldGiven :1; /* ucrit */
    unsigned MOS2maxDriftVelGiven   :1; /* vmax */
    unsigned MOS2junctionDepthGiven :1; /* xj */
    unsigned MOS2channelChargeGiven :1; /* neff */
    unsigned MOS2fNcoefGiven :1;
    unsigned MOS2fNexpGiven :1;

} MOS2model;

#ifndef NMOS
#define NMOS 1
#define PMOS -1
#endif /*NMOS*/

/* device parameters */
#define MOS2_W 1
#define MOS2_L 2
#define MOS2_AS 3
#define MOS2_AD 4
#define MOS2_PS 5
#define MOS2_PD 6
#define MOS2_NRS 7
#define MOS2_NRD 8
#define MOS2_OFF 9
#define MOS2_IC 10
#define MOS2_IC_VBS 11
#define MOS2_IC_VDS 12
#define MOS2_IC_VGS 13
#define MOS2_W_SENS 14
#define MOS2_L_SENS 15
#define MOS2_CB 16
#define MOS2_CG 17
#define MOS2_CS 18
#define MOS2_POWER 19
#define MOS2_CGS                20
#define MOS2_CGD                21
#define MOS2_DNODE              22
#define MOS2_GNODE              23
#define MOS2_SNODE              24
#define MOS2_BNODE              25
#define MOS2_DNODEPRIME         26
#define MOS2_SNODEPRIME         27
#define MOS2_SOURCECONDUCT      28
#define MOS2_DRAINCONDUCT       29
#define MOS2_VON                30
#define MOS2_VDSAT              31
#define MOS2_SOURCEVCRIT        32
#define MOS2_DRAINVCRIT         33
#define MOS2_CD                 34
#define MOS2_CBS                35
#define MOS2_CBD                36
#define MOS2_GMBS               37
#define MOS2_GM                 38
#define MOS2_GDS                39
#define MOS2_GBD                40
#define MOS2_GBS                41
#define MOS2_CAPBD              42
#define MOS2_CAPBS              43
#define MOS2_CAPZEROBIASBD      44
#define MOS2_CAPZEROBIASBDSW    45
#define MOS2_CAPZEROBIASBS      46
#define MOS2_CAPZEROBIASBSSW    47
#define MOS2_VBD                48
#define MOS2_VBS                49
#define MOS2_VGS                50
#define MOS2_VDS                51
#define MOS2_CAPGS              52
#define MOS2_QGS                53
#define MOS2_CQGS               54
#define MOS2_CAPGD              55
#define MOS2_QGD                56
#define MOS2_CQGD               57
#define MOS2_CAPGB              58
#define MOS2_QGB                59
#define MOS2_CQGB               60
#define MOS2_QBD                61
#define MOS2_CQBD               62
#define MOS2_QBS                63
#define MOS2_CQBS               64
#define MOS2_W_SENS_REAL        65
#define MOS2_W_SENS_IMAG        66
#define MOS2_W_SENS_MAG         67 
#define MOS2_W_SENS_PH          68 
#define MOS2_W_SENS_CPLX        69
#define MOS2_L_SENS_REAL        70
#define MOS2_L_SENS_IMAG        71
#define MOS2_L_SENS_MAG         72
#define MOS2_L_SENS_PH          73
#define MOS2_L_SENS_CPLX        74
#define MOS2_L_SENS_DC          75
#define MOS2_W_SENS_DC          76
#define MOS2_TEMP               77
#define MOS2_SOURCERESIST       78
#define MOS2_DRAINRESIST        79
#define MOS2_M                  80
#define MOS2_DTEMP              81

/* model paramerers */
#define MOS2_MOD_VTO 101
#define MOS2_MOD_KP 102
#define MOS2_MOD_GAMMA 103
#define MOS2_MOD_PHI 104
#define MOS2_MOD_LAMBDA 105
#define MOS2_MOD_RD 106
#define MOS2_MOD_RS 107
#define MOS2_MOD_CBD 108
#define MOS2_MOD_CBS 109
#define MOS2_MOD_IS 110
#define MOS2_MOD_PB 111
#define MOS2_MOD_CGSO 112
#define MOS2_MOD_CGDO 113
#define MOS2_MOD_CGBO 114
#define MOS2_MOD_CJ 115
#define MOS2_MOD_MJ 116
#define MOS2_MOD_CJSW 117
#define MOS2_MOD_MJSW 118
#define MOS2_MOD_JS 119
#define MOS2_MOD_TOX 120
#define MOS2_MOD_LD 121
#define MOS2_MOD_RSH 122
#define MOS2_MOD_U0 123
#define MOS2_MOD_FC 124
#define MOS2_MOD_NSUB 125
#define MOS2_MOD_TPG 126
#define MOS2_MOD_NSS 127
#define MOS2_MOD_NFS 128
#define MOS2_MOD_DELTA 129
#define MOS2_MOD_UEXP 130
#define MOS2_MOD_VMAX 131
#define MOS2_MOD_XJ 132
#define MOS2_MOD_NEFF 133
#define MOS2_MOD_UCRIT 134
#define MOS2_MOD_NMOS 135
#define MOS2_MOD_PMOS 136
#define MOS2_MOD_TNOM 137
#define MOS2_MOD_KF 139
#define MOS2_MOD_AF 140
#define MOS2_MOD_TYPE 141


/* model questions */

#include "mos2ext.h"

#endif /*MOS2*/

