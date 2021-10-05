/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef BJT
#define BJT

#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* structures to describe Bipolar Junction Transistors */

/* indices to array of BJT noise sources */

enum {
    BJTRCNOIZ = 0,
    BJTRBNOIZ,
    BJT_RE_NOISE,
    BJTICNOIZ,
    BJTIBNOIZ,
    BJTFLNOIZ,
    BJTTOTNOIZ,
    /* finally, the number of noise sources */
    BJTNSRCS
};

/* data needed to describe a single instance */

typedef struct sBJTinstance {

    struct GENinstance gen;

#define BJTmodPtr(inst) ((struct sBJTmodel *)((inst)->gen.GENmodPtr))
#define BJTnextInstance(inst) ((struct sBJTinstance *)((inst)->gen.GENnextInstance))
#define BJTname gen.GENname
#define BJTstate gen.GENstate

    const int BJTcolNode;   /* number of collector node of bjt */
    const int BJTbaseNode;  /* number of base node of bjt */
    const int BJTemitNode;  /* number of emitter node of bjt */
    const int BJTsubstNode; /* number of substrate node of bjt */
    int BJTcollCXNode;      /* number of internal collector node of bjt */
    int BJTcolPrimeNode;    /* number of internal collector node of bjt */
    int BJTbasePrimeNode;   /* number of internal base node of bjt */
    int BJTemitPrimeNode;   /* number of internal emitter node of bjt */
    int BJTsubstConNode;   /* number of node which substrate is connected to */
                           /* Substrate connection is either base prime      *
                            * or collector prime depending on whether        *
                            * the device is VERTICAL or LATERAL              */
    double BJTarea;      /* (emitter) area factor for the bjt */
    double BJTareab;     /* base area factor for the bjt */
    double BJTareac;     /* collector area factor for the bjt */
    double BJTm;        /* parallel multiplier */
    double BJTicVBE;    /* initial condition voltage B-E*/
    double BJTicVCE;    /* initial condition voltage C-E*/
    double BJTtemp;     /* instance temperature */
    double BJTdtemp;     /* instance delta temperature from circuit */
    double BJTtSatCur;  /* temperature adjusted saturation current */
    double BJTBEtSatCur;  /* temperature adjusted saturation current */
    double BJTBCtSatCur;  /* temperature adjusted saturation current */
    double BJTtBetaF;   /* temperature adjusted forward beta */
    double BJTtBetaR;   /* temperature adjusted reverse beta */
    double BJTtBEleakCur;  /* temperature adjusted B-E leakage current */
    double BJTtBCleakCur;  /* temperature adjusted B-C leakage current */
    double BJTtBEcap;   /* temperature adjusted B-E capacitance */
    double BJTtBEpot;   /* temperature adjusted B-E potential */
    double BJTtBCcap;   /* temperature adjusted B-C capacitance */
    double BJTtBCpot;   /* temperature adjusted B-C potential */
    double BJTtSubcap;   /* temperature adjusted Substrate capacitance */
    double BJTtSubpot;   /* temperature adjusted Substrate potential */
    double BJTtDepCap;  /* temperature adjusted join point in diode curve */
    double BJTtf1;      /* temperature adjusted polynomial coefficient */
    double BJTtf4;      /* temperature adjusted polynomial coefficient */
    double BJTtf5;      /* temperature adjusted polynomial coefficient */
    double BJTtf2;      /* temperature adjusted polynomial coefficient */
    double BJTtf3;      /* temperature adjusted polynomial coefficient */
    double BJTtf6;      /* temperature adjusted polynomial coefficient */
    double BJTtf7;      /* temperature adjusted polynomial coefficient */
    double BJTtVcrit;   /* temperature adjusted critical voltage */
    double BJTtSubVcrit; /* temperature adjusted substrate critical voltage */
    double BJTtSubSatCur; /* temperature adjusted subst. saturation current */
    double BJTtcollectorConduct;   /* temperature adjusted */
    double BJTtemitterConduct;   /* temperature adjusted */
    double BJTtbaseResist;   /* temperature adjusted */
    double BJTtbaseCurrentHalfResist;   /* temperature adjusted */
    double BJTtminBaseResist;   /* temperature adjusted */
    double BJTtinvEarlyVoltF;   /* temperature adjusted */
    double BJTtinvEarlyVoltR;   /* temperature adjusted */
    double BJTtinvRollOffF;   /* temperature adjusted */
    double BJTtinvRollOffR;   /* temperature adjusted */
    double BJTtemissionCoeffF;   /* temperature adjusted NF */
    double BJTtemissionCoeffR;   /* temperature adjusted NR */
    double BJTtleakBEemissionCoeff;   /* temperature adjusted NE */
    double BJTtleakBCemissionCoeff;   /* temperature adjusted NC */
    double BJTttransitTimeHighCurrentF;   /* temperature adjusted */
    double BJTttransitTimeF;   /* temperature adjusted */
    double BJTttransitTimeR;   /* temperature adjusted */
    double BJTtjunctionExpBE;   /* temperature adjusted MJE */
    double BJTtjunctionExpBC;   /* temperature adjusted MJC */
    double BJTtjunctionExpSub;   /* temperature adjusted MJS */
    double BJTtemissionCoeffS;   /* temperature adjusted NS */
    double BJTtintCollResist;   /* temperature adjusted QS RO */
    double BJTtepiSatVoltage;   /* temperature adjusted QS VO */
    double BJTtepiDoping;   /* temperature adjusted QS GAMMA */

    double *BJTcollCollCXPtr;    /* pointer to sparse matrix at
                             * (collector,collector cx) */
    double *BJTbaseBasePrimePtr;    /* pointer to sparse matrix at
                             * (base,base prime) */
    double *BJTemitEmitPrimePtr;    /* pointer to sparse matrix at
                             * (emitter,emitter prime) */
    double *BJTcollCXCollPtr;    /* pointer to sparse matrix at
                             * (collector cx,collector) */
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
    double *BJTsubstConSubstPtr;    /* pointer to sparse matrix at
                             * (Substrate connection, substrate) */
    double *BJTsubstSubstConPtr;    /* pointer to sparse matrix at
                             * (substrate, Substrate connection) */
    double *BJTsubstConSubstConPtr; /* pointer to sparse matrix at
                             * (Substrate connection, Substrate connection) */
                            /* Substrate connection is either base prime *
                             * or collector prime depending on whether   *
                             * the device is VERTICAL or LATERAL         */
    double *BJTbaseColPrimePtr; /* pointer to sparse matrix at
                             * (base,collector prime) */
    double *BJTcolPrimeBasePtr; /* pointer to sparse matrix at
                             * (collector prime,base) */

    double *BJTcollCXcollCXPtr; /* pointer to sparse matrix at
                             * (collector cx,collector cx) */
    double *BJTcollCXBasePrimePtr; /* pointer to sparse matrix at
                             * (collector cx,base prime) */
    double *BJTbasePrimeCollCXPtr; /* pointer to sparse matrix at
                             * (base prime,collector cx) */
    double *BJTcolPrimeCollCXPtr;    /* pointer to sparse matrix at
                             * (collector prime,collector cx) */
    double *BJTcollCXColPrimePtr;    /* pointer to sparse matrix at
                             * (collector cx,base prime) */

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
    double BJTcapsub;
    double BJTcapbx;
    double BJTcapbcx;
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
#define BJTsenCsub BJTsens+45 /* stores the perturbed values of csub */
#define BJTdphibedp BJTsens+51
#define BJTdphibcdp BJTsens+52
#define BJTdphisubdp BJTsens+53
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
#define BJTvbcx BJTstate+2
#define BJTvrci BJTstate+3
#define BJTcc BJTstate+4
#define BJTcb BJTstate+5
#define BJTgpi BJTstate+6
#define BJTgmu BJTstate+7
#define BJTgm BJTstate+8
#define BJTgo BJTstate+9
#define BJTqbe BJTstate+10
#define BJTcqbe BJTstate+11
#define BJTqbc BJTstate+12
#define BJTcqbc BJTstate+13
#define BJTqsub BJTstate+14
#define BJTcqsub BJTstate+15
#define BJTqbx BJTstate+16
#define BJTcqbx BJTstate+17
#define BJTgx BJTstate+18
#define BJTcexbc BJTstate+19
#define BJTgeqcb BJTstate+20
#define BJTgcsub BJTstate+21
#define BJTgeqbx BJTstate+22
#define BJTvsub BJTstate+23
#define BJTcdsub BJTstate+24
#define BJTgdsub BJTstate+25
#define BJTirci BJTstate+26
#define BJTirci_Vrci BJTstate+27
#define BJTirci_Vbci BJTstate+28
#define BJTirci_Vbcx BJTstate+29
#define BJTqbcx BJTstate+30
#define BJTcqbcx BJTstate+31
#define BJTgbcx BJTstate+32

#define BJTnumStates 33

#define BJTsensxpbe BJTstate+24 /* charge sensitivities and their
                   derivatives. +25 for the derivatives -
                   pointer to the beginning of the array */
#define BJTsensxpbc BJTstate+26
#define BJTsensxpsub BJTstate+28
#define BJTsensxpbx BJTstate+30

#define BJTnumSenStates 8

/* per model data */
typedef struct sBJTmodel {          /* model structure for a bjt */

    struct GENmodel gen;

#define BJTmodType gen.GENmodType
#define BJTnextModel(inst) ((struct sBJTmodel *)((inst)->gen.GENnextModel))
#define BJTinstances(inst) ((BJTinstance *)((inst)->gen.GENinstances))
#define BJTmodName gen.GENmodName

    int BJTtype;
    int BJTsubs;

    double BJTtnom; /* nominal temperature */
    double BJTsatCur;   /* input - don't use */
    double BJTBEsatCur;
    double BJTBCsatCur;
    double BJTbetaF;    /* input - don't use */
    double BJTemissionCoeffF;
    double BJTearlyVoltF;
    double BJTrollOffF;
    double BJTleakBEcurrent;    /* input - don't use */
    double BJTleakBEemissionCoeff;
    double BJTbetaR;    /* input - don't use */
    double BJTemissionCoeffR;
    double BJTearlyVoltR;
    double BJTrollOffR;
    double BJTleakBCcurrent;    /* input - don't use */
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
    double BJTcapSub;
    double BJTpotentialSubstrate;
    double BJTexponentialSubstrate;
    double BJTbetaExp;
    double BJTenergyGap;
    double BJTtempExpIS;
    double BJTdepletionCapCoeff;
    double BJTfNcoef;
    double BJTfNexp;
    double BJTsubSatCur;   /* input - don't use */
    double BJTemissionCoeffS;
    double BJTintCollResist;
    double BJTepiSatVoltage;
    double BJTepiDoping;
    double BJTepiCharge;
    int    BJTtlev;
    int    BJTtlevc;
    double BJTtbf1;
    double BJTtbf2;
    double BJTtbr1;
    double BJTtbr2;
    double BJTtikf1;
    double BJTtikf2;
    double BJTtikr1;
    double BJTtikr2;
    double BJTtirb1;
    double BJTtirb2;
    double BJTtnc1;
    double BJTtnc2;
    double BJTtne1;
    double BJTtne2;
    double BJTtnf1;
    double BJTtnf2;
    double BJTtnr1;
    double BJTtnr2;
    double BJTtrb1;
    double BJTtrb2;
    double BJTtrc1;
    double BJTtrc2;
    double BJTtre1;
    double BJTtre2;
    double BJTtrm1;
    double BJTtrm2;
    double BJTtvaf1;
    double BJTtvaf2;
    double BJTtvar1;
    double BJTtvar2;
    double BJTctc;
    double BJTcte;
    double BJTcts;
    double BJTtvjc;
    double BJTtvje;
    double BJTtvjs;
    double BJTtitf1;
    double BJTtitf2;
    double BJTttf1;
    double BJTttf2;
    double BJTttr1;
    double BJTttr2;
    double BJTtmje1;
    double BJTtmje2;
    double BJTtmjc1;
    double BJTtmjc2;
    double BJTtmjs1;
    double BJTtmjs2;
    double BJTtns1;
    double BJTtns2;
    double BJTnkf;
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
    double BJTtis1;
    double BJTtis2;
    double BJTtise1;
    double BJTtise2;
    double BJTtisc1;
    double BJTtisc2;
    double BJTtiss1;
    double BJTtiss2;
    int    BJTquasimod;
    double BJTenergyGapQS;
    double BJTtempExpRCI;
    double BJTtempExpVO;
    double BJTvbeMax; /* maximum voltage over B-E junction */
    double BJTvbcMax; /* maximum voltage over B-C junction */
    double BJTvceMax; /* maximum voltage over C-E branch */
    double BJTicMax;  /* maximum collector current */
    double BJTibMax;  /* maximum base current */
    double BJTpdMax; /* maximum device power dissipation */
    double BJTteMax;  /* maximum device temperature */
    double BJTrth0;   /* thermal resistance juntion to ambient */

    unsigned BJTsubsGiven : 1;
    unsigned BJTtnomGiven : 1;
    unsigned BJTsatCurGiven : 1;
    unsigned BJTBEsatCurGiven : 1;
    unsigned BJTBCsatCurGiven : 1;
    unsigned BJTbetaFGiven : 1;
    unsigned BJTemissionCoeffFGiven : 1;
    unsigned BJTearlyVoltFGiven : 1;
    unsigned BJTrollOffFGiven : 1;
    unsigned BJTleakBEcurrentGiven : 1;
    unsigned BJTleakBEemissionCoeffGiven : 1;
    unsigned BJTbetaRGiven : 1;
    unsigned BJTemissionCoeffRGiven : 1;
    unsigned BJTearlyVoltRGiven : 1;
    unsigned BJTrollOffRGiven : 1;
    unsigned BJTleakBCcurrentGiven : 1;
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
    unsigned BJTcapSubGiven : 1;
    unsigned BJTpotentialSubstrateGiven : 1;
    unsigned BJTexponentialSubstrateGiven : 1;
    unsigned BJTbetaExpGiven : 1;
    unsigned BJTenergyGapGiven : 1;
    unsigned BJTtempExpISGiven : 1;
    unsigned BJTdepletionCapCoeffGiven : 1;
    unsigned BJTfNcoefGiven : 1;
    unsigned BJTfNexpGiven :1;
    unsigned BJTsubSatCurGiven : 1;
    unsigned BJTemissionCoeffSGiven : 1;
    unsigned BJTintCollResistGiven : 1;
    unsigned BJTepiSatVoltageGiven : 1;
    unsigned BJTepiDopingGiven : 1;
    unsigned BJTepiChargeGiven : 1;
    unsigned BJTtlevGiven : 1;
    unsigned BJTtlevcGiven : 1;
    unsigned BJTtbf1Given : 1;
    unsigned BJTtbf2Given : 1;
    unsigned BJTtbr1Given : 1;
    unsigned BJTtbr2Given : 1;
    unsigned BJTtikf1Given : 1;
    unsigned BJTtikf2Given : 1;
    unsigned BJTtikr1Given : 1;
    unsigned BJTtikr2Given : 1;
    unsigned BJTtirb1Given : 1;
    unsigned BJTtirb2Given : 1;
    unsigned BJTtnc1Given : 1;
    unsigned BJTtnc2Given : 1;
    unsigned BJTtne1Given : 1;
    unsigned BJTtne2Given : 1;
    unsigned BJTtnf1Given : 1;
    unsigned BJTtnf2Given : 1;
    unsigned BJTtnr1Given : 1;
    unsigned BJTtnr2Given : 1;
    unsigned BJTtrb1Given : 1;
    unsigned BJTtrb2Given : 1;
    unsigned BJTtrc1Given : 1;
    unsigned BJTtrc2Given : 1;
    unsigned BJTtre1Given : 1;
    unsigned BJTtre2Given : 1;
    unsigned BJTtrm1Given : 1;
    unsigned BJTtrm2Given : 1;
    unsigned BJTtvaf1Given : 1;
    unsigned BJTtvaf2Given : 1;
    unsigned BJTtvar1Given : 1;
    unsigned BJTtvar2Given : 1;
    unsigned BJTctcGiven : 1;
    unsigned BJTcteGiven : 1;
    unsigned BJTctsGiven : 1;
    unsigned BJTtvjcGiven : 1;
    unsigned BJTtvjeGiven : 1;
    unsigned BJTtvjsGiven : 1;
    unsigned BJTtitf1Given : 1;
    unsigned BJTtitf2Given : 1;
    unsigned BJTttf1Given : 1;
    unsigned BJTttf2Given : 1;
    unsigned BJTttr1Given : 1;
    unsigned BJTttr2Given : 1;
    unsigned BJTtmje1Given : 1;
    unsigned BJTtmje2Given : 1;
    unsigned BJTtmjc1Given : 1;
    unsigned BJTtmjc2Given : 1;
    unsigned BJTtmjs1Given : 1;
    unsigned BJTtmjs2Given : 1;
    unsigned BJTtns1Given : 1;
    unsigned BJTtns2Given : 1;
    unsigned BJTnkfGiven : 1;
    unsigned BJTtis1Given : 1;
    unsigned BJTtis2Given : 1;
    unsigned BJTtise1Given : 1;
    unsigned BJTtise2Given : 1;
    unsigned BJTtisc1Given : 1;
    unsigned BJTtisc2Given : 1;
    unsigned BJTtiss1Given : 1;
    unsigned BJTtiss2Given : 1;
    unsigned BJTquasimodGiven : 1;
    unsigned BJTenergyGapQSGiven : 1;
    unsigned BJTtempExpRCIGiven : 1;
    unsigned BJTtempExpVOGiven : 1;
    unsigned BJTvbeMaxGiven : 1;
    unsigned BJTvbcMaxGiven : 1;
    unsigned BJTvceMaxGiven : 1;
    unsigned BJTpdMaxGiven : 1;
    unsigned BJTicMaxGiven : 1;
    unsigned BJTibMaxGiven : 1;
    unsigned BJTteMaxGiven : 1;
    unsigned BJTrth0Given : 1;
} BJTmodel;

#ifndef NPN
#define NPN 1
#define PNP -1
#endif /*NPN*/

/* 
 *  BJT defaults to vertical for both NPN and
 *  PNP devices. 
 */
#ifndef VERTICAL
#define VERTICAL 1
#define LATERAL -1
#endif /* VERTICAL */

/* device parameters */
enum {
    BJT_AREA = 1,
    BJT_OFF,
    BJT_IC_VBE,
    BJT_IC_VCE,
    BJT_IC,
    BJT_AREA_SENS,
    BJT_TEMP,
    BJT_DTEMP,
    BJT_M,
    BJT_AREAB,
    BJT_AREAC,
};

/* model parameters */
enum {
    BJT_MOD_NPN = 101,
    BJT_MOD_PNP,
    BJT_MOD_IS,
    BJT_MOD_IBE,
    BJT_MOD_IBC,
    BJT_MOD_BF,
    BJT_MOD_NF,
    BJT_MOD_VAF,
    BJT_MOD_IKF,
    BJT_MOD_ISE,
    BJT_MOD_NE,
    BJT_MOD_BR,
    BJT_MOD_NR,
    BJT_MOD_VAR,
    BJT_MOD_IKR,
    BJT_MOD_ISC,
    BJT_MOD_NC,
    BJT_MOD_RB,
    BJT_MOD_IRB,
    BJT_MOD_RBM,
    BJT_MOD_RE,
    BJT_MOD_RC,
    BJT_MOD_CJE,
    BJT_MOD_VJE,
    BJT_MOD_MJE,
    BJT_MOD_TF,
    BJT_MOD_XTF,
    BJT_MOD_VTF,
    BJT_MOD_ITF,
    BJT_MOD_PTF,
    BJT_MOD_CJC,
    BJT_MOD_VJC,
    BJT_MOD_MJC,
    BJT_MOD_XCJC,
    BJT_MOD_TR,
    BJT_MOD_CJS,
    BJT_MOD_VJS,
    BJT_MOD_MJS,
    BJT_MOD_XTB,
    BJT_MOD_EG,
    BJT_MOD_XTI,
    BJT_MOD_FC,
    BJT_MOD_AF,
    BJT_MOD_KF,
    BJT_MOD_ISS,
    BJT_MOD_NS,
    BJT_MOD_RCO,
    BJT_MOD_VO,
    BJT_MOD_GAMMA,
    BJT_MOD_QCO,
    BJT_MOD_TNOM,
    BJT_MOD_TLEV,
    BJT_MOD_TLEVC,
    BJT_MOD_TBF1,
    BJT_MOD_TBF2,
    BJT_MOD_TBR1,
    BJT_MOD_TBR2,
    BJT_MOD_TIKF1,
    BJT_MOD_TIKF2,
    BJT_MOD_TIKR1,
    BJT_MOD_TIKR2,
    BJT_MOD_TIRB1,
    BJT_MOD_TIRB2,
    BJT_MOD_TNC1,
    BJT_MOD_TNC2,
    BJT_MOD_TNE1,
    BJT_MOD_TNE2,
    BJT_MOD_TNF1,
    BJT_MOD_TNF2,
    BJT_MOD_TNR1,
    BJT_MOD_TNR2,
    BJT_MOD_TRB1,
    BJT_MOD_TRB2,
    BJT_MOD_TRC1,
    BJT_MOD_TRC2,
    BJT_MOD_TRE1,
    BJT_MOD_TRE2,
    BJT_MOD_TRM1,
    BJT_MOD_TRM2,
    BJT_MOD_TVAF1,
    BJT_MOD_TVAF2,
    BJT_MOD_TVAR1,
    BJT_MOD_TVAR2,
    BJT_MOD_CTC,
    BJT_MOD_CTE,
    BJT_MOD_CTS,
    BJT_MOD_TVJC,
    BJT_MOD_TVJE,
    BJT_MOD_TVJS,
    BJT_MOD_TITF1,
    BJT_MOD_TITF2,
    BJT_MOD_TTF1,
    BJT_MOD_TTF2,
    BJT_MOD_TTR1,
    BJT_MOD_TTR2,
    BJT_MOD_TMJE1,
    BJT_MOD_TMJE2,
    BJT_MOD_TMJC1,
    BJT_MOD_TMJC2,
    BJT_MOD_TMJS1,
    BJT_MOD_TMJS2,
    BJT_MOD_TNS1,
    BJT_MOD_TNS2,
    BJT_MOD_SUBS,
    BJT_MOD_NKF,
    BJT_MOD_TIS1,
    BJT_MOD_TIS2,
    BJT_MOD_TISE1,
    BJT_MOD_TISE2,
    BJT_MOD_TISC1,
    BJT_MOD_TISC2,
    BJT_MOD_TISS1,
    BJT_MOD_TISS2,
    BJT_MOD_QUASIMOD,
    BJT_MOD_EGQS,
    BJT_MOD_XRCI,
    BJT_MOD_XD,
    BJT_MOD_VBE_MAX,
    BJT_MOD_VBC_MAX,
    BJT_MOD_VCE_MAX,
    BJT_MOD_PD_MAX,
    BJT_MOD_IC_MAX,
    BJT_MOD_IB_MAX,
    BJT_MOD_TE_MAX,
    BJT_MOD_RTH0,
};

/* device questions */
enum {
    BJT_QUEST_FT = 211,
    BJT_QUEST_COLNODE,
    BJT_QUEST_BASENODE,
    BJT_QUEST_EMITNODE,
    BJT_QUEST_SUBSTNODE,
    BJT_QUEST_COLLCXNODE,
    BJT_QUEST_COLPRIMENODE,
    BJT_QUEST_BASEPRIMENODE,
    BJT_QUEST_EMITPRIMENODE,
    BJT_QUEST_VBE,
    BJT_QUEST_VBC,
    BJT_QUEST_CC,
    BJT_QUEST_CB,
    BJT_QUEST_GPI,
    BJT_QUEST_GMU,
    BJT_QUEST_GM,
    BJT_QUEST_GO,
    BJT_QUEST_QBE,
    BJT_QUEST_CQBE,
    BJT_QUEST_QBC,
    BJT_QUEST_CQBC,
    BJT_QUEST_QSUB,
    BJT_QUEST_CQSUB,
    BJT_QUEST_QBX,
    BJT_QUEST_CQBX,
    BJT_QUEST_GX,
    BJT_QUEST_CEXBC,
    BJT_QUEST_GEQCB,
    BJT_QUEST_GCSUB,
    BJT_QUEST_GEQBX,
    BJT_QUEST_SENS_REAL,
    BJT_QUEST_SENS_IMAG,
    BJT_QUEST_SENS_MAG,
    BJT_QUEST_SENS_PH,
    BJT_QUEST_SENS_CPLX,
    BJT_QUEST_SENS_DC,
    BJT_QUEST_CE,
    BJT_QUEST_CS,
    BJT_QUEST_POWER,
    BJT_QUEST_CPI,
    BJT_QUEST_CMU,
    BJT_QUEST_CBX,
    BJT_QUEST_CSUB,
    BJT_QUEST_GDSUB,
};

/* model questions */
enum {
    BJT_MOD_INVEARLYF = 301,
    BJT_MOD_INVEARLYR,
    BJT_MOD_INVROLLOFFF,
    BJT_MOD_INVROLLOFFR,
    BJT_MOD_COLCONDUCT,
    BJT_MOD_EMITTERCONDUCT,
    BJT_MOD_TRANSVBCFACT,
    BJT_MOD_EXCESSPHASEFACTOR,
    BJT_MOD_TYPE,
    BJT_MOD_QUEST_SUBS,
};

#include "bjtext.h"
#endif /*BJT*/
