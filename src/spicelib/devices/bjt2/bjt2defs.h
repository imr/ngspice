/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/

#ifndef BJT2
#define BJT2

#include "cktdefs.h"
#include "ifsim.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

/* structures to describe Bipolar Junction Transistors */

/* data needed to describe a single instance */

typedef struct sBJT2instance {
    struct sBJT2model *BJT2modPtr;    /* backpointer to model */
    struct sBJT2instance *BJT2nextInstance;   /* pointer to next instance of
                                             * current model*/
    IFuid BJT2name;  /* pointer to character string naming this instance */
    int BJT2owner;  /* number of owner process */
    int BJT2state; /* pointer to start of state vector for bjt2 */

    int BJT2colNode; /* number of collector node of bjt2 */
    int BJT2baseNode;    /* number of base node of bjt2 */
    int BJT2emitNode;    /* number of emitter node of bjt2 */
    int BJT2substNode;   /* number of substrate node of bjt2 */
    int BJT2colPrimeNode;    /* number of internal collector node of bjt2 */
    int BJT2basePrimeNode;   /* number of internal base node of bjt2 */
    int BJT2emitPrimeNode;   /* number of internal emitter node of bjt2 */
    int BJT2substConNode;   /* number of node which substrate is connected to */
                            /* Substrate connection is either base prime      *
                             * or collector prime depending on whether        *
                             * the device is VERTICAL or LATERAL              */
    double BJT2area;     /* (emitter) area factor for the bjt2 */
    double BJT2areab;    /* base area factor for the bjt2 */
    double BJT2areac;    /* collector area factor for the bjt2 */
    double BJT2m;        /* parallel multiplier */
    double BJT2icVBE;    /* initial condition voltage B-E*/
    double BJT2icVCE;    /* initial condition voltage C-E*/
    double BJT2temp;     /* instance temperature */
    double BJT2dtemp;    /* instance delta temperature from circuit */
    double BJT2tSatCur;  /* temperature adjusted saturation current */
    double BJT2tSubSatCur; /* temperature adjusted subst. saturation current */
    double BJT2tEmitterConduct;   /* emitter conductance */
    double BJT2tCollectorConduct; /* collector conductance */
    double BJT2tBaseResist;  /* temperature adjusted base resistance */
    double BJT2tMinBaseResist;  /* temperature adjusted base resistance */
    double BJT2tBetaF;   /* temperature adjusted forward beta */
    double BJT2tBetaR;   /* temperature adjusted reverse beta */
    double BJT2tBEleakCur;   /* temperature adjusted B-E leakage current */
    double BJT2tBCleakCur;   /* temperature adjusted B-C leakage current */
    double BJT2tBEcap;   /* temperature adjusted B-E capacitance */
    double BJT2tBEpot;   /* temperature adjusted B-E potential */
    double BJT2tBCcap;   /* temperature adjusted B-C capacitance */
    double BJT2tBCpot;   /* temperature adjusted B-C potential */
    double BJT2tSubcap;   /* temperature adjusted Substrate capacitance */
    double BJT2tSubpot;   /* temperature adjusted Substrate potential */
    double BJT2tDepCap;  /* temperature adjusted join point in diode curve */
    double BJT2tf1;      /* temperature adjusted polynomial coefficient */
    double BJT2tf4;      /* temperature adjusted polynomial coefficient */
    double BJT2tf5;      /* temperature adjusted polynomial coefficient */
    double BJT2tVcrit;   /* temperature adjusted critical voltage */
    double BJT2tSubVcrit; /* temperature adjusted substrate critical voltage */

    double *BJT2colColPrimePtr;  /* pointer to sparse matrix at
                             * (collector,collector prime) */
    double *BJT2baseBasePrimePtr;    /* pointer to sparse matrix at
                             * (base,base prime) */
    double *BJT2emitEmitPrimePtr;    /* pointer to sparse matrix at
                             * (emitter,emitter prime) */
    double *BJT2colPrimeColPtr;  /* pointer to sparse matrix at
                             * (collector prime,collector) */
    double *BJT2colPrimeBasePrimePtr;    /* pointer to sparse matrix at
                             * (collector prime,base prime) */
    double *BJT2colPrimeEmitPrimePtr;    /* pointer to sparse matrix at
                             * (collector prime,emitter prime) */
    double *BJT2basePrimeBasePtr;    /* pointer to sparse matrix at
                             * (base prime,base ) */
    double *BJT2basePrimeColPrimePtr;    /* pointer to sparse matrix at
                             * (base prime,collector prime) */
    double *BJT2basePrimeEmitPrimePtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *BJT2emitPrimeEmitPtr;    /* pointer to sparse matrix at
                             * (emitter prime,emitter) */
    double *BJT2emitPrimeColPrimePtr;    /* pointer to sparse matrix at
                             * (emitter prime,collector prime) */
    double *BJT2emitPrimeBasePrimePtr;   /* pointer to sparse matrix at
                             * (emitter prime,base prime) */
    double *BJT2colColPtr;   /* pointer to sparse matrix at
                             * (collector,collector) */
    double *BJT2baseBasePtr; /* pointer to sparse matrix at
                             * (base,base) */
    double *BJT2emitEmitPtr; /* pointer to sparse matrix at
                             * (emitter,emitter) */
    double *BJT2colPrimeColPrimePtr; /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *BJT2basePrimeBasePrimePtr;   /* pointer to sparse matrix at
                             * (base prime,base prime) */
    double *BJT2emitPrimeEmitPrimePtr;   /* pointer to sparse matrix at
                             * (emitter prime,emitter prime) */
    double *BJT2substSubstPtr;   /* pointer to sparse matrix at
                             * (substrate,substrate) */
    double *BJT2substConSubstPtr;    /* pointer to sparse matrix at
                             * (Substrate connection, substrate) */
    double *BJT2substSubstConPtr;    /* pointer to sparse matrix at
                             * (substrate, Substrate connection) */
    double *BJT2substConSubstConPtr;    /* pointer to sparse matrix at
                             * (Substrate connection, Substrate connection) */
                            /* Substrate connection is either base prime *
                             * or collector prime depending on whether   *
                             * the device is VERTICAL or LATERAL         */
    double *BJT2baseColPrimePtr; /* pointer to sparse matrix at
                             * (base,collector prime) */
    double *BJT2colPrimeBasePtr; /* pointer to sparse matrix at
                             * (collector prime,base) */

    unsigned BJT2off :1;   /* 'off' flag for bjt2 */
    unsigned BJT2tempGiven    :1; /* temperature given  for bjt2 instance*/
    unsigned BJT2dtempGiven   :1; /* temperature given  for bjt2 instance*/    
    unsigned BJT2areaGiven    :1; /* flag to indicate (emitter) area was specified */
    unsigned BJT2areabGiven   :1; /* flag to indicate base area was specified */
    unsigned BJT2areacGiven   :1; /* flag to indicate collector area was specified */
    unsigned BJT2mGiven       :1; /* flag to indicate m parameter specified */
    unsigned BJT2icVBEGiven   :1; /* flag to indicate VBE init. cond. given */
    unsigned BJT2icVCEGiven   :1; /* flag to indicate VCE init. cond. given */
    unsigned BJT2senPertFlag  :1; /* indictes whether the the parameter of
                        the particular instance is to be perturbed */

    int  BJT2senParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/
    double BJT2capbe;
    double BJT2capbc;
    double BJT2capsub;
    double BJT2capbx;
    double *BJT2sens;

#define BJT2senGpi BJT2sens /* stores the perturbed values of gpi */
#define BJT2senGmu BJT2sens+5 /* stores the perturbed values of gmu */
#define BJT2senGm BJT2sens+10 /* stores the perturbed values of gm */
#define BJT2senGo BJT2sens+15 /* stores the perturbed values of go */
#define BJT2senGx BJT2sens+20 /* stores the perturbed values of gx */
#define BJT2senCpi BJT2sens+25 /* stores the perturbed values of cpi */
#define BJT2senCmu BJT2sens+30 /* stores the perturbed values of cmu */
#define BJT2senCbx BJT2sens+35 /* stores the perturbed values of cbx */
#define BJT2senCmcb BJT2sens+40 /* stores the perturbed values of cmcb */
#define BJT2senCsub BJT2sens+45 /* stores the perturbed values of csub */
#define BJT2dphibedp BJT2sens+51
#define BJT2dphibcdp BJT2sens+52
#define BJT2dphisubdp BJT2sens+53
#define BJT2dphibxdp BJT2sens+54

/*
 * distortion stuff
 * the following naming convention is used:
 * x = vbe
 * y = vbc
 * z = vbb
 * w = vbed (vbe delayed for the linear gm delay)
 * therefore ic_xyz stands for the coefficient of the vbe*vbc*vbb
 * term in the multidimensional Taylor expansion for ic; and ibb_x2y
 * for the coeff. of the vbe*vbe*vbc term in the ibb expansion.
 */

#define BJT2NDCOEFFS	65

#ifndef NODISTO
	double BJT2dCoeffs[BJT2NDCOEFFS];
#else /* NODISTO */
	double *BJT2dCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	ic_x		BJT2dCoeffs[0]
#define	ic_y		BJT2dCoeffs[1]
#define	ic_xd		BJT2dCoeffs[2]
#define	ic_x2		BJT2dCoeffs[3]
#define	ic_y2		BJT2dCoeffs[4]
#define	ic_w2		BJT2dCoeffs[5]
#define	ic_xy		BJT2dCoeffs[6]
#define	ic_yw		BJT2dCoeffs[7]
#define	ic_xw		BJT2dCoeffs[8]
#define	ic_x3		BJT2dCoeffs[9]
#define	ic_y3		BJT2dCoeffs[10]
#define	ic_w3		BJT2dCoeffs[11]
#define	ic_x2w		BJT2dCoeffs[12]
#define	ic_x2y		BJT2dCoeffs[13]
#define	ic_y2w		BJT2dCoeffs[14]
#define	ic_xy2		BJT2dCoeffs[15]
#define	ic_xw2		BJT2dCoeffs[16]
#define	ic_yw2		BJT2dCoeffs[17]
#define	ic_xyw		BJT2dCoeffs[18]

#define	ib_x		BJT2dCoeffs[19]
#define	ib_y		BJT2dCoeffs[20]
#define	ib_x2		BJT2dCoeffs[21]
#define	ib_y2		BJT2dCoeffs[22]
#define	ib_xy		BJT2dCoeffs[23]
#define	ib_x3		BJT2dCoeffs[24]
#define	ib_y3		BJT2dCoeffs[25]
#define	ib_x2y		BJT2dCoeffs[26]
#define	ib_xy2		BJT2dCoeffs[27]

#define	ibb_x		BJT2dCoeffs[28]
#define	ibb_y		BJT2dCoeffs[29]
#define	ibb_z		BJT2dCoeffs[30]
#define	ibb_x2		BJT2dCoeffs[31]
#define	ibb_y2		BJT2dCoeffs[32]
#define	ibb_z2		BJT2dCoeffs[33]
#define	ibb_xy		BJT2dCoeffs[34]
#define	ibb_yz		BJT2dCoeffs[35]
#define	ibb_xz		BJT2dCoeffs[36]
#define	ibb_x3		BJT2dCoeffs[37]
#define	ibb_y3		BJT2dCoeffs[38]
#define	ibb_z3		BJT2dCoeffs[39]
#define	ibb_x2z		BJT2dCoeffs[40]
#define	ibb_x2y		BJT2dCoeffs[41]
#define	ibb_y2z		BJT2dCoeffs[42]
#define	ibb_xy2		BJT2dCoeffs[43]
#define	ibb_xz2		BJT2dCoeffs[44]
#define	ibb_yz2		BJT2dCoeffs[45]
#define	ibb_xyz		BJT2dCoeffs[46]

#define	qbe_x		BJT2dCoeffs[47]
#define	qbe_y		BJT2dCoeffs[48]
#define	qbe_x2		BJT2dCoeffs[49]
#define	qbe_y2		BJT2dCoeffs[50]
#define	qbe_xy		BJT2dCoeffs[51]
#define	qbe_x3		BJT2dCoeffs[52]
#define	qbe_y3		BJT2dCoeffs[53]
#define	qbe_x2y		BJT2dCoeffs[54]
#define	qbe_xy2		BJT2dCoeffs[55]

#define	capbc1		BJT2dCoeffs[56]
#define	capbc2		BJT2dCoeffs[57]
#define	capbc3		BJT2dCoeffs[58]

#define	capbx1		BJT2dCoeffs[59]
#define	capbx2		BJT2dCoeffs[60]
#define	capbx3		BJT2dCoeffs[61]

#define	capsc1		BJT2dCoeffs[62]
#define	capsc2		BJT2dCoeffs[63]
#define	capsc3		BJT2dCoeffs[64]

#endif

/* indices to array of BJT2 noise sources */

#define BJT2RCNOIZ       0
#define BJT2RBNOIZ       1
#define BJT2_RE_NOISE       2
#define BJT2ICNOIZ       3
#define BJT2IBNOIZ       4
#define BJT2FLNOIZ 5
#define BJT2TOTNOIZ    6

#define BJT2NSRCS     7     /* the number of BJT2 noise sources */

#ifndef NONOISE
      double BJT2nVar[NSTATVARS][BJT2NSRCS];
#else /*NONOISE*/
      double **BJT2nVar;
#endif /*NONOISE*/
/* the above to avoid allocating memory when it is not needed */

} BJT2instance ;

/* entries in the state vector for bjt2: */
#define BJT2vbe BJT2state
#define BJT2vbc BJT2state+1
#define BJT2cc BJT2state+2
#define BJT2cb BJT2state+3
#define BJT2gpi BJT2state+4
#define BJT2gmu BJT2state+5
#define BJT2gm BJT2state+6
#define BJT2go BJT2state+7
#define BJT2qbe BJT2state+8
#define BJT2cqbe BJT2state+9
#define BJT2qbc BJT2state+10
#define BJT2cqbc BJT2state+11
#define BJT2qsub BJT2state+12
#define BJT2cqsub BJT2state+13
#define BJT2qbx BJT2state+14
#define BJT2cqbx BJT2state+15
#define BJT2gx BJT2state+16
#define BJT2cexbc BJT2state+17
#define BJT2geqcb BJT2state+18
#define BJT2gcsub BJT2state+19
#define BJT2geqbx BJT2state+20
#define BJT2vsub BJT2state+21
#define BJT2cdsub BJT2state+22
#define BJT2gdsub BJT2state+23
#define BJT2numStates 24

#define BJT2sensxpbe BJT2state+24 /* charge sensitivities and their
                   derivatives. +25 for the derivatives -
                   pointer to the beginning of the array */
#define BJT2sensxpbc BJT2state+26
#define BJT2sensxpsub BJT2state+28
#define BJT2sensxpbx BJT2state+30

#define BJT2numSenStates 8


/* per model data */
typedef struct sBJT2model {          /* model structure for a bjt2 */
    int BJT2modType; /* type index of this device type */
    struct sBJT2model *BJT2nextModel; /* pointer to next possible model in
                                     * linked list */
    BJT2instance * BJT2instances; /* pointer to list of instances
                                 * that have this model */
    IFuid BJT2modName; /* pointer to character string naming this model */
    int BJT2type;


    int BJT2subs;
    double BJT2tnom; /* nominal temperature */
    double BJT2satCur;   /* input - don't use */
    double BJT2subSatCur;   /* input - don't use */
    double BJT2betaF;    /* input - don't use */
    double BJT2emissionCoeffF;
    double BJT2earlyVoltF;
    double BJT2rollOffF;
    double BJT2leakBEcurrent;    /* input - don't use */
    double BJT2c2;
    double BJT2leakBEemissionCoeff;
    double BJT2betaR;    /* input - don't use */
    double BJT2emissionCoeffR;
    double BJT2earlyVoltR;
    double BJT2rollOffR;
    double BJT2leakBCcurrent;    /* input - don't use */
    double BJT2c4;
    double BJT2leakBCemissionCoeff;
    double BJT2baseResist;
    double BJT2baseCurrentHalfResist;
    double BJT2minBaseResist;
    double BJT2emitterResist;
    double BJT2collectorResist;
    double BJT2depletionCapBE;   /* input - don't use */
    double BJT2potentialBE;  /* input - don't use */
    double BJT2junctionExpBE;
    double BJT2transitTimeF;
    double BJT2transitTimeBiasCoeffF;
    double BJT2transitTimeFVBC;
    double BJT2transitTimeHighCurrentF;
    double BJT2excessPhase;
    double BJT2depletionCapBC;   /* input - don't use */
    double BJT2potentialBC;  /* input - don't use */
    double BJT2junctionExpBC;
    double BJT2baseFractionBCcap;
    double BJT2transitTimeR;
    double BJT2capSub;
    double BJT2potentialSubstrate;
    double BJT2exponentialSubstrate;
    double BJT2betaExp;
    double BJT2energyGap;
    double BJT2tempExpIS;
    double BJT2reTempCoeff1;
    double BJT2reTempCoeff2;
    double BJT2rcTempCoeff1;
    double BJT2rcTempCoeff2;
    double BJT2rbTempCoeff1;
    double BJT2rbTempCoeff2;
    double BJT2rbmTempCoeff1;
    double BJT2rbmTempCoeff2;
    double BJT2depletionCapCoeff;
    double BJT2fNcoef;
    double BJT2fNexp;
    
    double BJT2invEarlyVoltF;    /* inverse of BJT2earlyVoltF */
    double BJT2invEarlyVoltR;    /* inverse of BJT2earlyVoltR */
    double BJT2invRollOffF;  /* inverse of BJT2rollOffF */
    double BJT2invRollOffR;  /* inverse of BJT2rollOffR */
    double BJT2collectorConduct; /* collector conductance */
    double BJT2emitterConduct;   /* emitter conductance */
    double BJT2transitTimeVBCFactor; /* */
    double BJT2excessPhaseFactor;
    double BJT2f2;
    double BJT2f3;
    double BJT2f6;
    double BJT2f7;

    unsigned BJT2subsGiven : 1;
    unsigned BJT2tnomGiven : 1;
    unsigned BJT2satCurGiven : 1;
    unsigned BJT2subSatCurGiven : 1;
    unsigned BJT2betaFGiven : 1;
    unsigned BJT2emissionCoeffFGiven : 1;
    unsigned BJT2earlyVoltFGiven : 1;
    unsigned BJT2rollOffFGiven : 1;
    unsigned BJT2leakBEcurrentGiven : 1;
    unsigned BJT2c2Given : 1;
    unsigned BJT2leakBEemissionCoeffGiven : 1;
    unsigned BJT2betaRGiven : 1;
    unsigned BJT2emissionCoeffRGiven : 1;
    unsigned BJT2earlyVoltRGiven : 1;
    unsigned BJT2rollOffRGiven : 1;
    unsigned BJT2leakBCcurrentGiven : 1;
    unsigned BJT2c4Given : 1;
    unsigned BJT2leakBCemissionCoeffGiven : 1;
    unsigned BJT2baseResistGiven : 1;
    unsigned BJT2baseCurrentHalfResistGiven : 1;
    unsigned BJT2minBaseResistGiven : 1;
    unsigned BJT2emitterResistGiven : 1;
    unsigned BJT2collectorResistGiven : 1;
    unsigned BJT2depletionCapBEGiven : 1;
    unsigned BJT2potentialBEGiven : 1;
    unsigned BJT2junctionExpBEGiven : 1;
    unsigned BJT2transitTimeFGiven : 1;
    unsigned BJT2transitTimeBiasCoeffFGiven : 1;
    unsigned BJT2transitTimeFVBCGiven : 1;
    unsigned BJT2transitTimeHighCurrentFGiven : 1;
    unsigned BJT2excessPhaseGiven : 1;
    unsigned BJT2depletionCapBCGiven : 1;
    unsigned BJT2potentialBCGiven : 1;
    unsigned BJT2junctionExpBCGiven : 1;
    unsigned BJT2baseFractionBCcapGiven : 1;
    unsigned BJT2transitTimeRGiven : 1;
    unsigned BJT2capSubGiven : 1;
    unsigned BJT2potentialSubstrateGiven : 1;
    unsigned BJT2exponentialSubstrateGiven : 1;
    unsigned BJT2betaExpGiven : 1;
    unsigned BJT2energyGapGiven : 1;
    unsigned BJT2tempExpISGiven : 1;
    unsigned BJT2reTempCoeff1Given : 1;
    unsigned BJT2reTempCoeff2Given : 1;
    unsigned BJT2rcTempCoeff1Given : 1;
    unsigned BJT2rcTempCoeff2Given : 1;
    unsigned BJT2rbTempCoeff1Given : 1;
    unsigned BJT2rbTempCoeff2Given : 1;
    unsigned BJT2rbmTempCoeff1Given : 1;
    unsigned BJT2rbmTempCoeff2Given : 1;
    unsigned BJT2depletionCapCoeffGiven : 1;
    unsigned BJT2fNcoefGiven : 1;
    unsigned BJT2fNexpGiven :1;
} BJT2model;

#ifndef NPN
#define NPN 1
#define PNP -1
#endif /*NPN*/

/* 
 *  BJT2 defaults to vertical for both NPN and
 *  PNP devices. It is possible to alter this
 *  behavior defining the GEOMETRY_COMPAT macro. 
 */
#ifndef VERTICAL
#define VERTICAL 1
#define LATERAL -1
#endif /* VERTICAL */


/* device parameters */
#define BJT2_AREA 1
#define BJT2_OFF 2
#define BJT2_IC_VBE 3
#define BJT2_IC_VCE 4
#define BJT2_IC 5
#define BJT2_AREA_SENS 6
#define BJT2_TEMP 7
#define BJT2_DTEMP 8
#define BJT2_M 9
#define BJT2_AREAB 10
#define BJT2_AREAC 11

/* model parameters */
#define BJT2_MOD_NPN 101
#define BJT2_MOD_PNP 102
#define BJT2_MOD_IS 103
#define BJT2_MOD_ISS 146
#define BJT2_MOD_BF 104
#define BJT2_MOD_NF 105
#define BJT2_MOD_VAF 106
#define BJT2_MOD_IKF 107
#define BJT2_MOD_ISE 108
#define BJT2_MOD_C2 109 
#define BJT2_MOD_NE 110
#define BJT2_MOD_BR 111
#define BJT2_MOD_NR 112
#define BJT2_MOD_VAR 113
#define BJT2_MOD_IKR 114
#define BJT2_MOD_ISC 115
#define BJT2_MOD_C4 116
#define BJT2_MOD_NC 117
#define BJT2_MOD_RB 118
#define BJT2_MOD_IRB 119
#define BJT2_MOD_RBM 120
#define BJT2_MOD_RE 121
#define BJT2_MOD_RC 122
#define BJT2_MOD_CJE 123
#define BJT2_MOD_VJE 124
#define BJT2_MOD_MJE 125
#define BJT2_MOD_TF 126
#define BJT2_MOD_XTF 127
#define BJT2_MOD_VTF 128
#define BJT2_MOD_ITF 129
#define BJT2_MOD_PTF 130
#define BJT2_MOD_CJC 131
#define BJT2_MOD_VJC 132
#define BJT2_MOD_MJC 133
#define BJT2_MOD_XCJC 134
#define BJT2_MOD_TR 135
#define BJT2_MOD_CJS 136
#define BJT2_MOD_VJS 137
#define BJT2_MOD_MJS 138
#define BJT2_MOD_XTB 139
#define BJT2_MOD_EG 140
#define BJT2_MOD_XTI 141
#define BJT2_MOD_FC 142
#define BJT2_MOD_TNOM 143
#define BJT2_MOD_AF 144
#define BJT2_MOD_KF 145
#define BJT2_MOD_SUBS 147
#define BJT2_MOD_TRE1 148
#define BJT2_MOD_TRE2 149
#define BJT2_MOD_TRC1 150
#define BJT2_MOD_TRC2 151
#define BJT2_MOD_TRB1 152
#define BJT2_MOD_TRB2 153
#define BJT2_MOD_TRBM1 154
#define BJT2_MOD_TRBM2 155


/* device questions */
#define BJT2_QUEST_FT             201
#define BJT2_QUEST_COLNODE        202
#define BJT2_QUEST_BASENODE       203
#define BJT2_QUEST_EMITNODE       204
#define BJT2_QUEST_SUBSTNODE      205
#define BJT2_QUEST_COLPRIMENODE   206
#define BJT2_QUEST_BASEPRIMENODE  207
#define BJT2_QUEST_EMITPRIMENODE  208
#define BJT2_QUEST_VBE            209
#define BJT2_QUEST_VBC            210
#define BJT2_QUEST_CC             211
#define BJT2_QUEST_CB             212
#define BJT2_QUEST_GPI            213
#define BJT2_QUEST_GMU            214
#define BJT2_QUEST_GM             215
#define BJT2_QUEST_GO             216
#define BJT2_QUEST_QBE            217
#define BJT2_QUEST_CQBE           218
#define BJT2_QUEST_QBC            219
#define BJT2_QUEST_CQBC           220
#define BJT2_QUEST_QSUB           221
#define BJT2_QUEST_CQSUB          222
#define BJT2_QUEST_QBX            223
#define BJT2_QUEST_CQBX           224
#define BJT2_QUEST_GX             225
#define BJT2_QUEST_CEXBC          226
#define BJT2_QUEST_GEQCB          227
#define BJT2_QUEST_GCSUB          228
#define BJT2_QUEST_GDSUB          243
#define BJT2_QUEST_GEQBX          229
#define BJT2_QUEST_SENS_REAL      230
#define BJT2_QUEST_SENS_IMAG      231
#define BJT2_QUEST_SENS_MAG       232
#define BJT2_QUEST_SENS_PH        233
#define BJT2_QUEST_SENS_CPLX      234
#define BJT2_QUEST_SENS_DC        235
#define BJT2_QUEST_CE             236
#define BJT2_QUEST_CS             237
#define BJT2_QUEST_POWER          238

#define BJT2_QUEST_CPI            239
#define BJT2_QUEST_CMU            240
#define BJT2_QUEST_CBX            241
#define BJT2_QUEST_CSUB           242

/* model questions */
#define BJT2_MOD_INVEARLYF             301
#define BJT2_MOD_INVEARLYR             302
#define BJT2_MOD_INVROLLOFFF           303
#define BJT2_MOD_INVROLLOFFR           304
#define BJT2_MOD_COLCONDUCT            305
#define BJT2_MOD_EMITTERCONDUCT        306
#define BJT2_MOD_TRANSVBCFACT          307
#define BJT2_MOD_EXCESSPHASEFACTOR     308
#define BJT2_MOD_TYPE		      309
#define BJT2_MOD_QUEST_SUBS		      310

#include "bjt2ext.h"
#endif /*BJT2*/
