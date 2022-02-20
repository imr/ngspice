/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
**********/
#ifndef DIO
#define DIO

#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

            /* data structures used to describe diodes */

/* indices to array of diode noise  sources */

enum {
    DIORSNOIZ = 0,
    DIOIDNOIZ,
    DIOFLNOIZ,
    DIOTOTNOIZ,
    /* finally, the number of noise sources */
    DIONSRCS
};

/* information needed per instance */

typedef struct sDIOinstance {

    struct GENinstance gen;

#define DIOmodPtr(inst) ((struct sDIOmodel *)((inst)->gen.GENmodPtr))
#define DIOnextInstance(inst) ((struct sDIOinstance *)((inst)->gen.GENnextInstance))
#define DIOname gen.GENname
#define DIOstate gen.GENstate

    const int DIOposNode;     /* number of positive node of diode */
    const int DIOnegNode;     /* number of negative node of diode */
    const int DIOtempNode;    /* number of the temperature node of the diode */
    int DIOposPrimeNode;      /* number of positive prime node of diode */

    double *DIOposPosPrimePtr;      /* pointer to sparse matrix at
                                     * (positive,positive prime) */
    double *DIOnegPosPrimePtr;      /* pointer to sparse matrix at
                                     * (negative,positive prime) */
    double *DIOposPrimePosPtr;      /* pointer to sparse matrix at
                                     * (positive prime,positive) */
    double *DIOposPrimeNegPtr;      /* pointer to sparse matrix at
                                     * (positive prime,negative) */
    double *DIOposPosPtr;   /* pointer to sparse matrix at
                             * (positive,positive) */
    double *DIOnegNegPtr;   /* pointer to sparse matrix at
                             * (negative,negative) */
    double *DIOposPrimePosPrimePtr; /* pointer to sparse matrix at
                                     * (positive prime,positive prime) */

    /* self heating */
    double *DIOtempPosPtr;
    double *DIOtempPosPrimePtr;
    double *DIOtempNegPtr;
    double *DIOtempTempPtr;
    double *DIOposTempPtr;
    double *DIOposPrimeTempPtr;
    double *DIOnegTempPtr;

    double DIOcap;   /* stores the diode capacitance */

    double *DIOsens; /* stores the perturbed values of geq and ceq in ac
                         sensitivity analyis */

    int DIOsenParmNo ;   /* parameter # for sensitivity use;
                          * set equal to  0 if not a design parameter*/

    unsigned DIOoff : 1;   /* 'off' flag for diode */
    unsigned DIOareaGiven : 1;   /* flag to indicate area was specified */
    unsigned DIOpjGiven : 1;   /* flag to indicate perimeter was specified */
    unsigned DIOwGiven : 1;   /* flag to indicate width was specified */
    unsigned DIOlGiven : 1;   /* flag to indicate length was specified */
    unsigned DIOmGiven : 1;   /* flag to indicate multiplier was specified */

    unsigned DIOinitCondGiven : 1;  /* flag to indicate ic was specified */
    unsigned DIOsenPertFlag :1; /* indictes whether the the parameter of
                               the particular instance is to be perturbed */
    unsigned DIOtempGiven : 1;  /* flag to indicate temperature was specified */
    unsigned DIOdtempGiven : 1; /* flag to indicate dtemp given */

    unsigned DIOlengthMetalGiven : 1; /* Length of metal capacitor (level=3) */
    unsigned DIOlengthPolyGiven : 1;  /* Length of polysilicon capacitor (level=3) */
    unsigned DIOwidthMetalGiven : 1;  /* Width of metal capacitor (level=3) */
    unsigned DIOwidthPolyGiven : 1;   /* Width of polysilicon capacitor (level=3) */

    double DIOarea;     /* area factor for the diode */
    double DIOpj;       /* perimeter for the diode */
    double DIOw;        /* width for the diode */
    double DIOl;        /* length for the diode */
    double DIOm;        /* multiplier for the diode */
    int    DIOthermal;  /* flag indicate self heating on */

    double DIOlengthMetal;   /* Length of metal capacitor (level=3) */
    double DIOlengthPoly;    /* Length of polysilicon capacitor (level=3) */
    double DIOwidthMetal;    /* Width of metal capacitor (level=3) */
    double DIOwidthPoly;     /* Width of polysilicon capacitor (level=3) */

    double DIOinitCond;      /* initial condition */
    double DIOtemp;          /* temperature of the instance */
    double DIOdtemp;         /* delta temperature of instance */
    double DIOtJctPot;       /* temperature adjusted junction potential */
    double DIOtJctCap;       /* temperature adjusted junction capacitance */
    double DIOtJctSWPot;     /* temperature adjusted sidewall junction potential */
    double DIOtJctSWCap;     /* temperature adjusted sidewall junction capacitance */
    double DIOtTransitTime;  /* temperature adjusted transit time */
    double DIOtGradingCoeff; /* temperature adjusted grading coefficient (MJ) */
    double DIOtConductance;    /* temperature adjusted series conductance */
    double DIOtConductance_dT; /* temperature adjusted series conductance temperature derivative */

    double DIOtDepCap;       /* temperature adjusted transition point in */
                             /* the curve matching (Fc * Vj ) */
    double DIOtDepSWCap;     /* temperature adjusted transition point in */
                             /* the curve matching (Fcs * Vjs ) */
    double DIOtSatCur;         /* temperature adjusted saturation current */
    double DIOtSatCur_dT;      /* temperature adjusted saturation current temperature derivative */
    double DIOtSatSWCur;       /* temperature adjusted side wall saturation current */
    double DIOtSatSWCur_dT;    /* temperature adjusted side wall saturation current temperature derivative */
    double DIOtTunSatCur;      /* tunneling saturation current */
    double DIOtTunSatCur_dT;   /* tunneling saturation current temperature derivative */
    double DIOtTunSatSWCur;    /* sidewall tunneling saturation current */
    double DIOtTunSatSWCur_dT; /* sidewall tunneling saturation current temperature derivative */

    double DIOtVcrit;   /* temperature adjusted V crit */
    double DIOtF1;      /* temperature adjusted f1 */
    double DIOtBrkdwnV; /* temperature adjusted breakdown voltage */

    double DIOtF2;     /* coeff. for capacitance equation precomputation */
    double DIOtF3;     /* coeff. for capacitance equation precomputation */
    double DIOtF2SW;   /* coeff. for capacitance equation precomputation */
    double DIOtF3SW;   /* coeff. for capacitance equation precomputation */

    double DIOforwardKneeCurrent; /* Forward Knee current */
    double DIOreverseKneeCurrent; /* Reverse Knee current */
    double DIOjunctionCap;        /* geometry adjusted junction capacitance */
    double DIOjunctionSWCap;      /* geometry adjusted junction sidewall capacitance */
    double DIOtRecSatCur;         /* temperature adjusted recombination saturation current */
    double DIOtRecSatCur_dT;      /* temperature adjusted recombination saturation current */

    double DIOdIth_dVrs;
    double DIOdIth_dVdio;
    double DIOdIth_dT;
    double DIOgcTt;
    double DIOdIrs_dT;
    double DIOdIdio_dT;

    double DIOcmetal; /* parasitic metal overlap capacitance */
    double DIOcpoly;  /* parasitic polysilicon overlap capacitance */

/*
 * naming convention:
 * x = vdiode
 */

/* the following are relevant to s.s. sinusoidal distortion analysis */

#define DIONDCOEFFS        6

#ifndef NODISTO
        double DIOdCoeffs[DIONDCOEFFS];
#else /* NODISTO */
        double *DIOdCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define        id_x2                DIOdCoeffs[0]
#define        id_x3                DIOdCoeffs[1]
#define        cdif_x2                DIOdCoeffs[2]
#define        cdif_x3                DIOdCoeffs[3]
#define        cjnc_x2                DIOdCoeffs[4]
#define        cjnc_x3                DIOdCoeffs[5]

#endif

#ifndef NONOISE
     double DIOnVar[NSTATVARS][DIONSRCS];
#else /* NONOISE */
        double **DIOnVar;
#endif /* NONOISE */

} DIOinstance ;

#define DIOsenGeq DIOsens /* stores the perturbed values of geq */
#define DIOsenCeq DIOsens + 3 /* stores the perturbed values of ceq */
#define DIOdphidp DIOsens + 6


#define DIOvoltage DIOstate
#define DIOcurrent DIOstate+1
#define DIOconduct DIOstate+2
#define DIOcapCharge DIOstate+3
#define DIOcapCurrent DIOstate+4

#define DIOqth DIOstate+5     /* thermal capacitor charge */
#define DIOcqth DIOstate+6    /* thermal capacitor current */

#define DIOdeltemp DIOstate+7 /* thermal voltage over rth0 */
#define DIOdIdio_dT DIOstate+8

#define DIOnumStates 9

#define DIOsensxp DIOstate+9    /* charge sensitivities and their derivatives.
                                 * +10 for the derivatives - pointer to the
                                 * beginning of the array */

#define DIOnumSenStates 2


/* per model data */

typedef struct sDIOmodel {       /* model structure for a diode */

    struct GENmodel gen;

#define DIOmodType gen.GENmodType
#define DIOnextModel(inst) ((struct sDIOmodel *)((inst)->gen.GENnextModel))
#define DIOinstances(inst) ((DIOinstance *)((inst)->gen.GENinstances))
#define DIOmodName gen.GENmodName

    unsigned DIOlevelGiven : 1;
    unsigned DIOsatCurGiven : 1;
    unsigned DIOsatSWCurGiven : 1;

    unsigned DIOresistGiven : 1;
    unsigned DIOresistTemp1Given : 1;
    unsigned DIOresistTemp2Given : 1;
    unsigned DIOemissionCoeffGiven : 1;
    unsigned DIOswEmissionCoeffGiven : 1;
    unsigned DIObrkdEmissionCoeffGiven : 1;
    unsigned DIOtransitTimeGiven : 1;
    unsigned DIOtranTimeTemp1Given : 1;
    unsigned DIOtranTimeTemp2Given : 1;
    unsigned DIOjunctionCapGiven : 1;
    unsigned DIOjunctionPotGiven : 1;
    unsigned DIOgradingCoeffGiven : 1;
    unsigned DIOgradCoeffTemp1Given : 1;
    unsigned DIOgradCoeffTemp2Given : 1;
    unsigned DIOjunctionSWCapGiven : 1;
    unsigned DIOjunctionSWPotGiven : 1;
    unsigned DIOgradingSWCoeffGiven : 1;
    unsigned DIOforwardKneeCurrentGiven : 1;
    unsigned DIOreverseKneeCurrentGiven : 1;

    unsigned DIOtlevGiven : 1;
    unsigned DIOtlevcGiven : 1;
    unsigned DIOactivationEnergyGiven : 1;
    unsigned DIOsaturationCurrentExpGiven : 1;
    unsigned DIOctaGiven : 1;
    unsigned DIOctpGiven : 1;
    unsigned DIOtpbGiven : 1;
    unsigned DIOtphpGiven : 1;
    unsigned DIOdepletionCapCoeffGiven : 1;
    unsigned DIOdepletionSWcapCoeffGiven :1;
    unsigned DIObreakdownVoltageGiven : 1;
    unsigned DIObreakdownCurrentGiven : 1;
    unsigned DIOtcvGiven : 1;
    unsigned DIOnomTempGiven : 1;
    unsigned DIOfNcoefGiven : 1;
    unsigned DIOfNexpGiven : 1;
    unsigned DIOareaGiven : 1;
    unsigned DIOpjGiven : 1;

    unsigned DIOtunSatCurGiven : 1;
    unsigned DIOtunSatSWCurGiven : 1;
    unsigned DIOtunEmissionCoeffGiven : 1;
    unsigned DIOtunSaturationCurrentExpGiven : 1;
    unsigned DIOtunEGcorrectionFactorGiven : 1;
    unsigned DIOfv_maxGiven : 1;
    unsigned DIObv_maxGiven : 1;
    unsigned DIOid_maxGiven : 1;
    unsigned DIOpd_maxGiven : 1;
    unsigned DIOte_maxGiven : 1;
    unsigned DIOrecSatCurGiven : 1;
    unsigned DIOrecEmissionCoeffGiven : 1;

    unsigned DIOrth0Given :1;
    unsigned DIOcth0Given :1;

    unsigned DIOlengthMetalGiven : 1;     /* Length of metal capacitor (level=3) */
    unsigned DIOlengthPolyGiven : 1;      /* Length of polysilicon capacitor (level=3) */
    unsigned DIOwidthMetalGiven : 1;      /* Width of metal capacitor (level=3) */
    unsigned DIOwidthPolyGiven : 1;       /* Width of polysilicon capacitor (level=3) */
    unsigned DIOmetalOxideThickGiven : 1; /* Thickness of the metal to bulk oxide (level=3) */
    unsigned DIOpolyOxideThickGiven : 1;  /* Thickness of the polysilicon to bulk oxide (level=3) */
    unsigned DIOmetalMaskOffsetGiven : 1; /* Masking and etching effects in metal (level=3)") */
    unsigned DIOpolyMaskOffsetGiven : 1;  /* Masking and etching effects in polysilicon (level=3) */

    int    DIOlevel;   /* level selector */
    double DIOsatCur;   /* saturation current */
    double DIOsatSWCur;   /* Sidewall saturation current */

    double DIOresist;             /* ohmic series resistance */
    double DIOresistTemp1;        /* series resistance 1st order temp. coeff. */
    double DIOresistTemp2;        /* series resistance 2nd order temp. coeff. */
    double DIOconductance;        /* conductance corresponding to ohmic R */
    double DIOemissionCoeff;      /* emission coefficient (N) */
    double DIOswEmissionCoeff;    /* Sidewall emission coefficient (NS) */
    double DIObrkdEmissionCoeff;  /* Breakdown emission coefficient (NBV) */
    double DIOtransitTime;        /* transit time (TT) */
    double DIOtranTimeTemp1;      /* transit time 1st order coefficient */
    double DIOtranTimeTemp2;      /* transit time 2nd order coefficient */
    double DIOjunctionCap;        /* Junction Capacitance (Cj0) */
    double DIOjunctionPot;        /* Junction Potential (Vj) or (PB) */
    double DIOgradingCoeff;       /* grading coefficient (m) or (mj) */
    double DIOgradCoeffTemp1;     /* grading coefficient 1st order temp. coeff.*/
    double DIOgradCoeffTemp2;     /* grading coefficient 2nd order temp. coeff.*/
    double DIOjunctionSWCap;      /* Sidewall Junction Capacitance (Cjsw) */
    double DIOjunctionSWPot;      /* Sidewall Junction Potential (Vjsw) or (PBSW) */
    double DIOgradingSWCoeff;     /* Sidewall grading coefficient (mjsw) */
    double DIOforwardKneeCurrent; /* Forward Knee current (IKF) */
    double DIOreverseKneeCurrent; /* Reverse Knee current (IKR) */

    int    DIOtlev; /* Diode temperature equation selector */
    int    DIOtlevc; /* Diode temperature equation selector */
    double DIOactivationEnergy; /* activation energy (EG) */
    double DIOsaturationCurrentExp; /* Saturation current exponential (XTI) */
    double DIOcta; /* Area junction temperature coefficient */
    double DIOctp; /* Perimeter junction temperature coefficient */
    double DIOtpb; /* Area junction potential temperature coefficient */
    double DIOtphp; /* Perimeter junction potential temperature coefficient */
    double DIOdepletionCapCoeff;    /* Depletion Cap fraction coefficient (FC)*/
    double DIOdepletionSWcapCoeff;    /* Depletion sw-Cap fraction coefficient (FCS)*/
    double DIObreakdownVoltage; /* Voltage at reverse breakdown */
    double DIObreakdownCurrent; /* Current at above voltage */
    double DIOtcv; /* Reverse breakdown voltage temperature coefficient */
    double DIOarea;     /* area factor for the diode */
    double DIOpj;       /* perimeter for the diode */

    double DIOnomTemp;  /* nominal temperature at which parms measured */
    double DIOfNcoef;
    double DIOfNexp;

    double DIOtunSatCur;        /* tunneling saturation current (JTUN) */
    double DIOtunSatSWCur;      /* sidewall tunneling saturation current (JTUNSW) */
    double DIOtunEmissionCoeff; /* tunneling emission coefficient (NTUN) */
    double DIOtunSaturationCurrentExp; /* exponent for the tunneling current temperature (XTITUN) */
    double DIOtunEGcorrectionFactor; /* EG correction factor for tunneling (KEG) */
    double DIOfv_max; /* maximum voltage in forward direction */
    double DIObv_max; /* maximum voltage in reverse direction */
    double DIOid_max; /* maximum current */
    double DIOpd_max; /* maximum power dissipation */
    double DIOte_max; /* maximum temperature */
    double DIOrecSatCur; /* Recombination saturation current */
    double DIOrecEmissionCoeff; /* Recombination emission coefficient */

    double DIOrth0;
    double DIOcth0;

    double DIOlengthMetal;     /* Length of metal capacitor (level=3) */
    double DIOlengthPoly;      /* Length of polysilicon capacitor (level=3) */
    double DIOwidthMetal;      /* Width of metal capacitor (level=3) */
    double DIOwidthPoly;       /* Width of polysilicon capacitor (level=3) */
    double DIOmetalOxideThick; /* Thickness of the metal to bulk oxide (level=3) */
    double DIOpolyOxideThick;  /* Thickness of the polysilicon to bulk oxide (level=3) */
    double DIOmetalMaskOffset; /* Masking and etching effects in metal (level=3)") */
    double DIOpolyMaskOffset;  /* Masking and etching effects in polysilicon (level=3) */

} DIOmodel;

/* device parameters */
enum {
    DIO_AREA = 1,
    DIO_IC,
    DIO_OFF,
    DIO_CURRENT,
    DIO_VOLTAGE,
    DIO_CHARGE,
    DIO_CAPCUR,
    DIO_CONDUCT,
    DIO_AREA_SENS,
    DIO_POWER,
    DIO_TEMP,
    DIO_QUEST_SENS_REAL,
    DIO_QUEST_SENS_IMAG,
    DIO_QUEST_SENS_MAG,
    DIO_QUEST_SENS_PH,
    DIO_QUEST_SENS_CPLX,
    DIO_QUEST_SENS_DC,
    DIO_CAP,
    DIO_PJ,
    DIO_W,
    DIO_L,
    DIO_M,
    DIO_DTEMP,
    DIO_THERMAL,
    DIO_LM,
    DIO_LP,
    DIO_WM,
    DIO_WP,
};

/* model parameters */
enum {
    DIO_MOD_LEVEL = 100,
    DIO_MOD_IS,
    DIO_MOD_RS,
    DIO_MOD_N,
    DIO_MOD_TT,
    DIO_MOD_CJO,
    DIO_MOD_VJ,
    DIO_MOD_M,
    DIO_MOD_EG,
    DIO_MOD_XTI,
    DIO_MOD_FC,
    DIO_MOD_BV,
    DIO_MOD_IBV,
    DIO_MOD_D,
    DIO_MOD_COND,
    DIO_MOD_TNOM,
    DIO_MOD_KF,
    DIO_MOD_AF,
    DIO_MOD_JSW,
    DIO_MOD_CJSW,
    DIO_MOD_VJSW,
    DIO_MOD_MJSW,
    DIO_MOD_IKF,
    DIO_MOD_IKR,
    DIO_MOD_FCS,
    DIO_MOD_TTT1,
    DIO_MOD_TTT2,
    DIO_MOD_TM1,
    DIO_MOD_TM2,
    DIO_MOD_TRS,
    DIO_MOD_TRS2,
    DIO_MOD_TLEV,
    DIO_MOD_TLEVC,
    DIO_MOD_CTA,
    DIO_MOD_CTP,
    DIO_MOD_TPB,
    DIO_MOD_TPHP,
    DIO_MOD_TCV,
    DIO_MOD_NBV,
    DIO_MOD_AREA,
    DIO_MOD_PJ,
    DIO_MOD_NS,
    DIO_MOD_JTUN,
    DIO_MOD_JTUNSW,
    DIO_MOD_NTUN,
    DIO_MOD_XTITUN,
    DIO_MOD_KEG,
    DIO_MOD_FV_MAX,
    DIO_MOD_BV_MAX,
    DIO_MOD_ID_MAX,
    DIO_MOD_TE_MAX,
    DIO_MOD_PD_MAX,
    DIO_MOD_ISR,
    DIO_MOD_NR,
    DIO_MOD_RTH0,
    DIO_MOD_CTH0,

    DIO_MOD_LM,
    DIO_MOD_LP,
    DIO_MOD_WM,
    DIO_MOD_WP,
    DIO_MOD_XOM,
    DIO_MOD_XOI,
    DIO_MOD_XM,
    DIO_MOD_XP,
};

void DIOtempUpdate(DIOmodel *inModel, DIOinstance *here, double Temp, CKTcircuit *ckt);

#include "dioext.h"
#endif /*DIO*/
