/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef BJT
#define BJT

#include "cktdefs.h"
#include "ifsim.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

/* structures to describe Bipolar Junction Transistors */

/* data needed to describe a single instance */

typedef struct sBJTinstance {
    struct sBJTmodel *BJTmodPtr;    /* backpointer to model */
    struct sBJTinstance *BJTnextInstance;   /* pointer to next instance of
                                             * current model*/
    IFuid BJTname;  /* pointer to character string naming this instance */
    int BJTowner;  /* number of owner process */
    int BJTstate; /* pointer to start of state vector for bjt */

    int BJTcolNode; /* number of collector node of bjt */
    int BJTbaseNode;    /* number of base node of bjt */
    int BJTemitNode;    /* number of emitter node of bjt */
    int BJTsubstNode;   /* number of substrate node of bjt */
    int BJTcolPrimeNode;    /* number of internal collector node of bjt */
    int BJTbasePrimeNode;   /* number of internal base node of bjt */
    int BJTemitPrimeNode;   /* number of internal emitter node of bjt */
    double BJTarea;      /* (emitter) area factor for the bjt */
    double BJTareab;     /* base area factor for the bjt */
    double BJTareac;     /* collector area factor for the bjt */
    double BJTm;        /* parallel multiplier */
    double BJTicVBE;    /* initial condition voltage B-E*/
    double BJTicVCE;    /* initial condition voltage C-E*/
    double BJTtemp;     /* instance temperature */
    double BJTdtemp;     /* instance delta temperature from circuit */
    double BJTtSatCur;  /* temperature adjusted saturation current */
    double BJTtBetaF;   /* temperature adjusted forward beta */
    double BJTtBetaR;   /* temperature adjusted reverse beta */
    double BJTtBEleakCur;   /* temperature adjusted B-E leakage current */
    double BJTtBCleakCur;   /* temperature adjusted B-C leakage current */
    double BJTtBEcap;   /* temperature adjusted B-E capacitance */
    double BJTtBEpot;   /* temperature adjusted B-E potential */
    double BJTtBCcap;   /* temperature adjusted B-C capacitance */
    double BJTtBCpot;   /* temperature adjusted B-C potential */
    double BJTtDepCap;  /* temperature adjusted join point in diode curve */
    double BJTtf1;      /* temperature adjusted polynomial coefficient */
    double BJTtf4;      /* temperature adjusted polynomial coefficient */
    double BJTtf5;      /* temperature adjusted polynomial coefficient */
    double BJTtVcrit;   /* temperature adjusted critical voltage */

    double *BJTcolColPrimePtr;  /* pointer to sparse matrix at
                             * (collector,collector prime) */
    double *BJTbaseBasePrimePtr;    /* pointer to sparse matrix at
                             * (base,base prime) */
    double *BJTemitEmitPrimePtr;    /* pointer to sparse matrix at
                             * (emitter,emitter prime) */
    double *BJTcolPrimeColPtr;  /* pointer to sparse matrix at
                             * (collector prime,collector) */
    double *BJTcolPrimeBasePrimePtr;    /* pointer to sparse matrix at
                             * (collector prime,base prime) */
    double *BJTcolPrimeEmitPrimePtr;    /* pointer to sparse matrix at
                             * (collector prime,emitter prime) */
    double *BJTbasePrimeBasePtr;    /* pointer to sparse matrix at
                             * (base prime,base ) */
    double *BJTbasePrimeColPrimePtr;    /* pointer to sparse matrix at
                             * (base prime,collector prime) */
    double *BJTbasePrimeEmitPrimePtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *BJTemitPrimeEmitPtr;    /* pointer to sparse matrix at
                             * (emitter prime,emitter) */
    double *BJTemitPrimeColPrimePtr;    /* pointer to sparse matrix at
                             * (emitter prime,collector prime) */
    double *BJTemitPrimeBasePrimePtr;   /* pointer to sparse matrix at
                             * (emitter prime,base prime) */
    double *BJTcolColPtr;   /* pointer to sparse matrix at
                             * (collector,collector) */
    double *BJTbaseBasePtr; /* pointer to sparse matrix at
                             * (base,base) */
    double *BJTemitEmitPtr; /* pointer to sparse matrix at
                             * (emitter,emitter) */
    double *BJTcolPrimeColPrimePtr; /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *BJTbasePrimeBasePrimePtr;   /* pointer to sparse matrix at
                             * (base prime,base prime) */
    double *BJTemitPrimeEmitPrimePtr;   /* pointer to sparse matrix at
                             * (emitter prime,emitter prime) */
    double *BJTsubstSubstPtr;   /* pointer to sparse matrix at
                             * (substrate,substrate) */
    double *BJTcolPrimeSubstPtr;    /* pointer to sparse matrix at
                             * (collector prime,substrate) */
    double *BJTsubstColPrimePtr;    /* pointer to sparse matrix at
                             * (substrate,collector prime) */
    double *BJTbaseColPrimePtr; /* pointer to sparse matrix at
                             * (base,collector prime) */
    double *BJTcolPrimeBasePtr; /* pointer to sparse matrix at
                             * (collector prime,base) */

    unsigned BJToff         :1;   /* 'off' flag for bjt */
    unsigned BJTtempGiven   :1; /* temperature given  for bjt instance*/
    unsigned BJTdtempGiven  :1; /* delta temperature given  for bjt instance*/
    unsigned BJTareaGiven   :1; /* flag to indicate area was specified */
    unsigned BJTareabGiven   :1; /* flag to indicate base area was specified */
    unsigned BJTareacGiven   :1; /* flag to indicate collector area was specified */
    unsigned BJTmGiven      :1; /* flag to indicate m parameter specified */
    unsigned BJTicVBEGiven  :1; /* flag to indicate VBE init. cond. given */
    unsigned BJTicVCEGiven  :1; /* flag to indicate VCE init. cond. given */
    unsigned BJTsenPertFlag :1; /* indictes whether the the parameter of
                        the particular instance is to be perturbed */

    int  BJTsenParmNo;   /* parameter # for sensitivity use;
            set equal to  0 if not a design parameter*/
    double BJTcapbe;
    double BJTcapbc;
    double BJTcapcs;
    double BJTcapbx;
    double *BJTsens;

#define BJTsenGpi BJTsens /* stores the perturbed values of gpi */
#define BJTsenGmu BJTsens+5 /* stores the perturbed values of gmu */
#define BJTsenGm BJTsens+10 /* stores the perturbed values of gm */
#define BJTsenGo BJTsens+15 /* stores the perturbed values of go */
#define BJTsenGx BJTsens+20 /* stores the perturbed values of gx */
#define BJTsenCpi BJTsens+25 /* stores the perturbed values of cpi */
#define BJTsenCmu BJTsens+30 /* stores the perturbed values of cmu */
#define BJTsenCbx BJTsens+35 /* stores the perturbed values of cbx */
#define BJTsenCmcb BJTsens+40 /* stores the perturbed values of cmcb */
#define BJTsenCcs BJTsens+45 /* stores the perturbed values of ccs */
#define BJTdphibedp BJTsens+51
#define BJTdphibcdp BJTsens+52
#define BJTdphicsdp BJTsens+53
#define BJTdphibxdp BJTsens+54

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

#define BJTNDCOEFFS	65

#ifndef NODISTO
	double BJTdCoeffs[BJTNDCOEFFS];
#else /* NODISTO */
	double *BJTdCoeffs;
#endif /* NODISTO */

#ifndef CONFIG

#define	ic_x		BJTdCoeffs[0]
#define	ic_y		BJTdCoeffs[1]
#define	ic_xd		BJTdCoeffs[2]
#define	ic_x2		BJTdCoeffs[3]
#define	ic_y2		BJTdCoeffs[4]
#define	ic_w2		BJTdCoeffs[5]
#define	ic_xy		BJTdCoeffs[6]
#define	ic_yw		BJTdCoeffs[7]
#define	ic_xw		BJTdCoeffs[8]
#define	ic_x3		BJTdCoeffs[9]
#define	ic_y3		BJTdCoeffs[10]
#define	ic_w3		BJTdCoeffs[11]
#define	ic_x2w		BJTdCoeffs[12]
#define	ic_x2y		BJTdCoeffs[13]
#define	ic_y2w		BJTdCoeffs[14]
#define	ic_xy2		BJTdCoeffs[15]
#define	ic_xw2		BJTdCoeffs[16]
#define	ic_yw2		BJTdCoeffs[17]
#define	ic_xyw		BJTdCoeffs[18]

#define	ib_x		BJTdCoeffs[19]
#define	ib_y		BJTdCoeffs[20]
#define	ib_x2		BJTdCoeffs[21]
#define	ib_y2		BJTdCoeffs[22]
#define	ib_xy		BJTdCoeffs[23]
#define	ib_x3		BJTdCoeffs[24]
#define	ib_y3		BJTdCoeffs[25]
#define	ib_x2y		BJTdCoeffs[26]
#define	ib_xy2		BJTdCoeffs[27]

#define	ibb_x		BJTdCoeffs[28]
#define	ibb_y		BJTdCoeffs[29]
#define	ibb_z		BJTdCoeffs[30]
#define	ibb_x2		BJTdCoeffs[31]
#define	ibb_y2		BJTdCoeffs[32]
#define	ibb_z2		BJTdCoeffs[33]
#define	ibb_xy		BJTdCoeffs[34]
#define	ibb_yz		BJTdCoeffs[35]
#define	ibb_xz		BJTdCoeffs[36]
#define	ibb_x3		BJTdCoeffs[37]
#define	ibb_y3		BJTdCoeffs[38]
#define	ibb_z3		BJTdCoeffs[39]
#define	ibb_x2z		BJTdCoeffs[40]
#define	ibb_x2y		BJTdCoeffs[41]
#define	ibb_y2z		BJTdCoeffs[42]
#define	ibb_xy2		BJTdCoeffs[43]
#define	ibb_xz2		BJTdCoeffs[44]
#define	ibb_yz2		BJTdCoeffs[45]
#define	ibb_xyz		BJTdCoeffs[46]

#define	qbe_x		BJTdCoeffs[47]
#define	qbe_y		BJTdCoeffs[48]
#define	qbe_x2		BJTdCoeffs[49]
#define	qbe_y2		BJTdCoeffs[50]
#define	qbe_xy		BJTdCoeffs[51]
#define	qbe_x3		BJTdCoeffs[52]
#define	qbe_y3		BJTdCoeffs[53]
#define	qbe_x2y		BJTdCoeffs[54]
#define	qbe_xy2		BJTdCoeffs[55]

#define	capbc1		BJTdCoeffs[56]
#define	capbc2		BJTdCoeffs[57]
#define	capbc3		BJTdCoeffs[58]

#define	capbx1		BJTdCoeffs[59]
#define	capbx2		BJTdCoeffs[60]
#define	capbx3		BJTdCoeffs[61]

#define	capsc1		BJTdCoeffs[62]
#define	capsc2		BJTdCoeffs[63]
#define	capsc3		BJTdCoeffs[64]

#endif

/* indices to array of BJT noise sources */

#define BJTRCNOIZ       0
#define BJTRBNOIZ       1
#define BJT_RE_NOISE       2
#define BJTICNOIZ       3
#define BJTIBNOIZ       4
#define BJTFLNOIZ 5
#define BJTTOTNOIZ    6

#define BJTNSRCS     7     /* the number of BJT noise sources */

#ifndef NONOISE
      double BJTnVar[NSTATVARS][BJTNSRCS];
#else /*NONOISE*/
      double **BJTnVar;
#endif /*NONOISE*/
/* the above to avoid allocating memory when it is not needed */

} BJTinstance ;

/* entries in the state vector for bjt: */
#define BJTvbe BJTstate
#define BJTvbc BJTstate+1
#define BJTcc BJTstate+2
#define BJTcb BJTstate+3
#define BJTgpi BJTstate+4
#define BJTgmu BJTstate+5
#define BJTgm BJTstate+6
#define BJTgo BJTstate+7
#define BJTqbe BJTstate+8
#define BJTcqbe BJTstate+9
#define BJTqbc BJTstate+10
#define BJTcqbc BJTstate+11
#define BJTqcs BJTstate+12
#define BJTcqcs BJTstate+13
#define BJTqbx BJTstate+14
#define BJTcqbx BJTstate+15
#define BJTgx BJTstate+16
#define BJTcexbc BJTstate+17
#define BJTgeqcb BJTstate+18
#define BJTgccs BJTstate+19
#define BJTgeqbx BJTstate+20
#define BJTnumStates 21

#define BJTsensxpbe BJTstate+21 /* charge sensitivities and their
                   derivatives. +22 for the derivatives -
                   pointer to the beginning of the array */
#define BJTsensxpbc BJTstate+23
#define BJTsensxpcs BJTstate+25
#define BJTsensxpbx BJTstate+27

#define BJTnumSenStates 8

/* per model data */
typedef struct sBJTmodel {          /* model structure for a bjt */
    int BJTmodType; /* type index of this device type */
    struct sBJTmodel *BJTnextModel; /* pointer to next possible model in
                                     * linked list */
    BJTinstance * BJTinstances; /* pointer to list of instances
                                 * that have this model */
    IFuid BJTmodName; /* pointer to character string naming this model */
    int BJTtype;

    double BJTtnom; /* nominal temperature */
    double BJTsatCur;   /* input - don't use */
    double BJTbetaF;    /* input - don't use */
    double BJTemissionCoeffF;
    double BJTearlyVoltF;
    double BJTrollOffF;
    double BJTleakBEcurrent;    /* input - don't use */
    double BJTc2;
    double BJTleakBEemissionCoeff;
    double BJTbetaR;    /* input - don't use */
    double BJTemissionCoeffR;
    double BJTearlyVoltR;
    double BJTrollOffR;
    double BJTleakBCcurrent;    /* input - don't use */
    double BJTc4;
    double BJTleakBCemissionCoeff;
    double BJTbaseResist;
    double BJTbaseCurrentHalfResist;
    double BJTminBaseResist;
    double BJTemitterResist;
    double BJTcollectorResist;
    double BJTdepletionCapBE;   /* input - don't use */
    double BJTpotentialBE;  /* input - don't use */
    double BJTjunctionExpBE;
    double BJTtransitTimeF;
    double BJTtransitTimeBiasCoeffF;
    double BJTtransitTimeFVBC;
    double BJTtransitTimeHighCurrentF;
    double BJTexcessPhase;
    double BJTdepletionCapBC;   /* input - don't use */
    double BJTpotentialBC;  /* input - don't use */
    double BJTjunctionExpBC;
    double BJTbaseFractionBCcap;
    double BJTtransitTimeR;
    double BJTcapCS;
    double BJTpotentialSubstrate;
    double BJTexponentialSubstrate;
    double BJTbetaExp;
    double BJTenergyGap;
    double BJTtempExpIS;
    double BJTdepletionCapCoeff;
    double BJTfNcoef;
    double BJTfNexp;
    
    double BJTinvEarlyVoltF;    /* inverse of BJTearlyVoltF */
    double BJTinvEarlyVoltR;    /* inverse of BJTearlyVoltR */
    double BJTinvRollOffF;  /* inverse of BJTrollOffF */
    double BJTinvRollOffR;  /* inverse of BJTrollOffR */
    double BJTcollectorConduct; /* collector conductance */
    double BJTemitterConduct;   /* emitter conductance */
    double BJTtransitTimeVBCFactor; /* */
    double BJTexcessPhaseFactor;
    double BJTf2;
    double BJTf3;
    double BJTf6;
    double BJTf7;

    unsigned BJTtnomGiven : 1;
    unsigned BJTsatCurGiven : 1;
    unsigned BJTbetaFGiven : 1;
    unsigned BJTemissionCoeffFGiven : 1;
    unsigned BJTearlyVoltFGiven : 1;
    unsigned BJTrollOffFGiven : 1;
    unsigned BJTleakBEcurrentGiven : 1;
    unsigned BJTc2Given : 1;
    unsigned BJTleakBEemissionCoeffGiven : 1;
    unsigned BJTbetaRGiven : 1;
    unsigned BJTemissionCoeffRGiven : 1;
    unsigned BJTearlyVoltRGiven : 1;
    unsigned BJTrollOffRGiven : 1;
    unsigned BJTleakBCcurrentGiven : 1;
    unsigned BJTc4Given : 1;
    unsigned BJTleakBCemissionCoeffGiven : 1;
    unsigned BJTbaseResistGiven : 1;
    unsigned BJTbaseCurrentHalfResistGiven : 1;
    unsigned BJTminBaseResistGiven : 1;
    unsigned BJTemitterResistGiven : 1;
    unsigned BJTcollectorResistGiven : 1;
    unsigned BJTdepletionCapBEGiven : 1;
    unsigned BJTpotentialBEGiven : 1;
    unsigned BJTjunctionExpBEGiven : 1;
    unsigned BJTtransitTimeFGiven : 1;
    unsigned BJTtransitTimeBiasCoeffFGiven : 1;
    unsigned BJTtransitTimeFVBCGiven : 1;
    unsigned BJTtransitTimeHighCurrentFGiven : 1;
    unsigned BJTexcessPhaseGiven : 1;
    unsigned BJTdepletionCapBCGiven : 1;
    unsigned BJTpotentialBCGiven : 1;
    unsigned BJTjunctionExpBCGiven : 1;
    unsigned BJTbaseFractionBCcapGiven : 1;
    unsigned BJTtransitTimeRGiven : 1;
    unsigned BJTcapCSGiven : 1;
    unsigned BJTpotentialSubstrateGiven : 1;
    unsigned BJTexponentialSubstrateGiven : 1;
    unsigned BJTbetaExpGiven : 1;
    unsigned BJTenergyGapGiven : 1;
    unsigned BJTtempExpISGiven : 1;
    unsigned BJTdepletionCapCoeffGiven : 1;
    unsigned BJTfNcoefGiven : 1;
    unsigned BJTfNexpGiven :1;
} BJTmodel;

#ifndef NPN
#define NPN 1
#define PNP -1
#endif /*NPN*/

/* device parameters */
#define BJT_AREA 1
#define BJT_OFF 2
#define BJT_IC_VBE 3
#define BJT_IC_VCE 4
#define BJT_IC 5
#define BJT_AREA_SENS 6
#define BJT_TEMP 7
#define BJT_DTEMP 8
#define BJT_M 9
#define BJT_AREAB 10
#define BJT_AREAC 11

/* model parameters */
#define BJT_MOD_NPN 101
#define BJT_MOD_PNP 102
#define BJT_MOD_IS 103
#define BJT_MOD_BF 104
#define BJT_MOD_NF 105
#define BJT_MOD_VAF 106
#define BJT_MOD_IKF 107
#define BJT_MOD_ISE 108
#define BJT_MOD_C2 109 
#define BJT_MOD_NE 110
#define BJT_MOD_BR 111
#define BJT_MOD_NR 112
#define BJT_MOD_VAR 113
#define BJT_MOD_IKR 114
#define BJT_MOD_ISC 115
#define BJT_MOD_C4 116
#define BJT_MOD_NC 117
#define BJT_MOD_RB 118
#define BJT_MOD_IRB 119
#define BJT_MOD_RBM 120
#define BJT_MOD_RE 121
#define BJT_MOD_RC 122
#define BJT_MOD_CJE 123
#define BJT_MOD_VJE 124
#define BJT_MOD_MJE 125
#define BJT_MOD_TF 126
#define BJT_MOD_XTF 127
#define BJT_MOD_VTF 128
#define BJT_MOD_ITF 129
#define BJT_MOD_PTF 130
#define BJT_MOD_CJC 131
#define BJT_MOD_VJC 132
#define BJT_MOD_MJC 133
#define BJT_MOD_XCJC 134
#define BJT_MOD_TR 135
#define BJT_MOD_CJS 136
#define BJT_MOD_VJS 137
#define BJT_MOD_MJS 138
#define BJT_MOD_XTB 139
#define BJT_MOD_EG 140
#define BJT_MOD_XTI 141
#define BJT_MOD_FC 142
#define BJT_MOD_TNOM 143
#define BJT_MOD_AF 144
#define BJT_MOD_KF 145

/* device questions */
#define BJT_QUEST_FT             201
#define BJT_QUEST_COLNODE        202
#define BJT_QUEST_BASENODE       203
#define BJT_QUEST_EMITNODE       204
#define BJT_QUEST_SUBSTNODE      205
#define BJT_QUEST_COLPRIMENODE   206
#define BJT_QUEST_BASEPRIMENODE  207
#define BJT_QUEST_EMITPRIMENODE  208
#define BJT_QUEST_VBE            209
#define BJT_QUEST_VBC            210
#define BJT_QUEST_CC             211
#define BJT_QUEST_CB             212
#define BJT_QUEST_GPI            213
#define BJT_QUEST_GMU            214
#define BJT_QUEST_GM             215
#define BJT_QUEST_GO             216
#define BJT_QUEST_QBE            217
#define BJT_QUEST_CQBE           218
#define BJT_QUEST_QBC            219
#define BJT_QUEST_CQBC           220
#define BJT_QUEST_QCS            221
#define BJT_QUEST_CQCS           222
#define BJT_QUEST_QBX            223
#define BJT_QUEST_CQBX           224
#define BJT_QUEST_GX             225
#define BJT_QUEST_CEXBC          226
#define BJT_QUEST_GEQCB          227
#define BJT_QUEST_GCCS           228
#define BJT_QUEST_GEQBX          229
#define BJT_QUEST_SENS_REAL      230
#define BJT_QUEST_SENS_IMAG      231
#define BJT_QUEST_SENS_MAG       232
#define BJT_QUEST_SENS_PH        233
#define BJT_QUEST_SENS_CPLX      234
#define BJT_QUEST_SENS_DC        235
#define BJT_QUEST_CE             236
#define BJT_QUEST_CS             237
#define BJT_QUEST_POWER          238

#define BJT_QUEST_CPI            239
#define BJT_QUEST_CMU            240
#define BJT_QUEST_CBX            241
#define BJT_QUEST_CCS            242

/* model questions */
#define BJT_MOD_INVEARLYF             301
#define BJT_MOD_INVEARLYR             302
#define BJT_MOD_INVROLLOFFF           303
#define BJT_MOD_INVROLLOFFR           304
#define BJT_MOD_COLCONDUCT            305
#define BJT_MOD_EMITTERCONDUCT        306
#define BJT_MOD_TRANSVBCFACT          307
#define BJT_MOD_EXCESSPHASEFACTOR     308
#define BJT_MOD_TYPE		      309


#include "bjtext.h"
#endif /*BJT*/
