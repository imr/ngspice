/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

#ifndef VBIC
#define VBIC

#include "cktdefs.h"
#include "ifsim.h"
#include "gendefs.h"
#include "complex.h"
#include "noisedef.h"

/* structures to describe Bipolar Junction Transistors */

/* data needed to describe a single instance */

typedef struct sVBICinstance {
    struct sVBICmodel *VBICmodPtr;    /* backpointer to model */
    struct sVBICinstance *VBICnextInstance;   /* pointer to next instance of
                                                 current model*/
    IFuid VBICname;  /* pointer to character string naming this instance */
    int VBICowner;   /* number of owner process */
    int VBICstate;   /* pointer to start of state vector for vbic */

    int VBICcollNode;   /* number of collector node of vbic */
    int VBICbaseNode;   /* number of base node of vbic */
    int VBICemitNode;   /* number of emitter node of vbic */
    int VBICsubsNode;   /* number of substrate node of vbic */
    int VBICcollCXNode; /* number of internal collector node of vbic */
    int VBICcollCINode; /* number of internal collector node of vbic */
    int VBICbaseBXNode; /* number of internal base node of vbic */
    int VBICbaseBINode; /* number of internal base node of vbic */
    int VBICemitEINode; /* number of internal emitter node of vbic */
    int VBICbaseBPNode; /* number of internal base node of vbic */
    int VBICsubsSINode; /* number of internal substrate node */

    double VBICarea;     /* area factor for the vbic */
    double VBICicVBE;    /* initial condition voltage B-E*/
    double VBICicVCE;    /* initial condition voltage C-E*/
    double VBICtemp;     /* instance temperature */
    double VBICdtemp;    /* instance delta temperature */
    double VBICm;        /* multiply factor for the vbic */

    double VBICtVcrit;
    double VBICttnom;    /* temperature adjusted model parameters per instance */
    double VBICtextCollResist;
    double VBICtintCollResist;
    double VBICtepiSatVoltage;
    double VBICtepiDoping;
    double VBICtextBaseResist;
    double VBICtintBaseResist;
    double VBICtemitterResist;
    double VBICtsubstrateResist;
    double VBICtparBaseResist;
    double VBICtsatCur;
    double VBICtemissionCoeffF;
    double VBICtemissionCoeffR;
    double VBICtdepletionCapBE;
    double VBICtpotentialBE;
    double VBICtdepletionCapBC;
    double VBICtextCapBC;
    double VBICtpotentialBC;
    double VBICtextCapSC;
    double VBICtpotentialSC;
    double VBICtidealSatCurBE;
    double VBICtnidealSatCurBE;
    double VBICtidealSatCurBC;
    double VBICtnidealSatCurBC;
    double VBICtavalanchePar2BC;
    double VBICtparasitSatCur;
    double VBICtidealParasitSatCurBE;
    double VBICtnidealParasitSatCurBE;
    double VBICtidealParasitSatCurBC;
    double VBICtnidealParasitSatCurBC;
    double VBICtrollOffF;
    double VBICtsepISRR;
    double VBICtvbbe;
    double VBICtnbbe;

    double *VBICcollCollPtr;  /* pointer to sparse matrix at
                             * (collector,collector) */
    double *VBICbaseBasePtr; /* pointer to sparse matrix at
                             * (base,base) */
    double *VBICemitEmitPtr; /* pointer to sparse matrix at
                             * (emitter,emitter) */
    double *VBICsubsSubsPtr;   /* pointer to sparse matrix at
                             * (substrate,substrate) */
    double *VBICcollCXCollCXPtr; /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *VBICcollCICollCIPtr; /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *VBICbaseBXBaseBXPtr;   /* pointer to sparse matrix at
                             * (base prime,base prime) */
    double *VBICbaseBIBaseBIPtr; /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *VBICbaseBPBaseBPPtr; /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *VBICemitEIEmitEIPtr;   /* pointer to sparse matrix at
                             * (emitter prime,emitter prime) */
    double *VBICsubsSISubsSIPtr;    /* pointer to sparse matrix at
                             * (substrate prime, substrate prime) */

    double *VBICbaseEmitPtr; /* pointer to sparse matrix at
                             * (base,emit) */
    double *VBICemitBasePtr; /* pointer to sparse matrix at
                             * (emit,base) */
    double *VBICbaseCollPtr; /* pointer to sparse matrix at
                             * (base,coll) */
    double *VBICcollBasePtr; /* pointer to sparse matrix at
                             * (coll,base) */
    double *VBICcollCollCXPtr;    /* pointer to sparse matrix at
                             * (collector,collector prime) */
    double *VBICbaseBaseBXPtr;    /* pointer to sparse matrix at
                             * (base,base prime) */
    double *VBICemitEmitEIPtr;    /* pointer to sparse matrix at
                             * (emitter,emitter prime) */
    double *VBICsubsSubsSIPtr;    /* pointer to sparse matrix at
                             * (substrate, Substrate connection) */
    double *VBICcollCXCollCIPtr;    /* pointer to sparse matrix at
                             * (collector prime,base prime) */
    double *VBICcollCXBaseBXPtr;    /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *VBICcollCXBaseBIPtr;    /* pointer to sparse matrix at
                             * (collector prime,collector prime) */
    double *VBICcollCXBaseBPPtr;    /* pointer to sparse matrix at
                             * (collector prime,base prime) */
    double *VBICcollCIBaseBIPtr; /* pointer to sparse matrix at
                             * (collector prime,base) */
    double *VBICcollCIEmitEIPtr;    /* pointer to sparse matrix at
                             * (collector prime,emitter prime) */
    double *VBICbaseBXBaseBIPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICbaseBXEmitEIPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICbaseBXBaseBPPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICbaseBXSubsSIPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICbaseBIEmitEIPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICbaseBPSubsSIPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */

    double *VBICcollCXCollPtr;  /* pointer to sparse matrix at
                             * (collector prime,collector) */
    double *VBICbaseBXBasePtr;    /* pointer to sparse matrix at
                             * (base prime,base ) */
    double *VBICemitEIEmitPtr;    /* pointer to sparse matrix at
                             * (emitter prime,emitter) */
    double *VBICsubsSISubsPtr;    /* pointer to sparse matrix at
                             * (Substrate connection, substrate) */
    double *VBICcollCICollCXPtr;    /* pointer to sparse matrix at
                             * (collector prime,base prime) */
    double *VBICbaseBICollCXPtr;    /* pointer to sparse matrix at
                             * (base prime,collector prime) */
    double *VBICbaseBPCollCXPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICbaseBXCollCIPtr; /* pointer to sparse matrix at
                             * (base,collector prime) */
    double *VBICbaseBICollCIPtr; /* pointer to sparse matrix at
                             * (base,collector prime) */
    double *VBICemitEICollCIPtr;    /* pointer to sparse matrix at
                             * (emitter prime,collector prime) */
    double *VBICbaseBPCollCIPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICsubsSICollCIPtr;   /* pointer to sparse matrix at
                             * (substrate,collector prime) */
    double *VBICbaseBIBaseBXPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICemitEIBaseBXPtr;   /* pointer to sparse matrix at
                             * (emitter prime,base prime) */
    double *VBICbaseBPBaseBXPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICsubsSIBaseBXPtr;   /* pointer to sparse matrix at
                             * (substrate,substrate) */
    double *VBICemitEIBaseBIPtr;   /* pointer to sparse matrix at
                             * (emitter prime,base prime) */
    double *VBICbaseBPBaseBIPtr;   /* pointer to sparse matrix at
                             * (base primt,emitter prime) */
    double *VBICsubsSIBaseBIPtr;   /* pointer to sparse matrix at
                             * (substrate,base prime) */
    double *VBICsubsSIBaseBPPtr;   /* pointer to sparse matrix at
                             * (substrate,substrate) */

    unsigned VBICareaGiven   :1; /* flag to indicate area was specified */
    unsigned VBICoff         :1; /* 'off' flag for vbic */
    unsigned VBICicVBEGiven  :1; /* flag to indicate VBE init. cond. given */
    unsigned VBICicVCEGiven  :1; /* flag to indicate VCE init. cond. given */
    unsigned VBICtempGiven   :1; /* temperature given for vbic instance*/
    unsigned VBICdtempGiven  :1; /* delta temperature given for vbic instance*/
    unsigned VBICmGiven      :1; /* flag to indicate multiplier was specified */
    unsigned VBICsenPertFlag :1; /* indictes whether the the parameter of
                                    the particular instance is to be perturbed */

    int  VBICsenParmNo;   /* parameter # for sensitivity use;
                             set equal to  0 if not a design parameter */
    double VBICcapbe;
    double VBICcapbex;
    double VBICcapbc;
    double VBICcapbcx;
    double VBICcapbep;
    double VBICcapbcp;
    double *VBICsens;

#define VBICsenGpi VBICsens /* stores the perturbed values of gpi */
#define VBICsenGmu VBICsens+5 /* stores the perturbed values of gmu */
#define VBICsenGm VBICsens+10 /* stores the perturbed values of gm */
#define VBICsenGo VBICsens+15 /* stores the perturbed values of go */
#define VBICsenGx VBICsens+20 /* stores the perturbed values of gx */
#define VBICsenCpi VBICsens+25 /* stores the perturbed values of cpi */
#define VBICsenCmu VBICsens+30 /* stores the perturbed values of cmu */
#define VBICsenCbx VBICsens+35 /* stores the perturbed values of cbx */
#define VBICsenCmcb VBICsens+40 /* stores the perturbed values of cmcb */
#define VBICsenCsub VBICsens+45 /* stores the perturbed values of csub */


/* indices to array of VBIC noise sources */

#define VBICRCNOIZ       0
#define VBICRCINOIZ      1
#define VBICRBNOIZ       2
#define VBICRBINOIZ      3
#define VBICRENOIZ       4
#define VBICRBPNOIZ      5
#define VBICICNOIZ       6
#define VBICIBNOIZ       7
#define VBICIBEPNOIZ     8
#define VBICFLBENOIZ     9
#define VBICFLBEPNOIZ   10
#define VBICRSNOIZ      11
#define VBICICCPNOIZ    12
#define VBICTOTNOIZ     13

#define VBICNSRCS       14     /* the number of VBIC noise sources */

#ifndef NONOISE
      double VBICnVar[NSTATVARS][VBICNSRCS];
#else /*NONOISE*/
      double **VBICnVar;
#endif /*NONOISE*/
/* the above to avoid allocating memory when it is not needed */

} VBICinstance ;

/* entries in the state vector for vbic: */

#define VBICvbei VBICstate
#define VBICvbex VBICstate+1
#define VBICvbci VBICstate+2
#define VBICvbcx VBICstate+3
#define VBICvbep VBICstate+4
#define VBICvrci VBICstate+5
#define VBICvrbi VBICstate+6
#define VBICvrbp VBICstate+7
#define VBICvbcp VBICstate+8

#define VBICibe VBICstate+9
#define VBICibe_Vbei VBICstate+10

#define VBICibex VBICstate+11
#define VBICibex_Vbex VBICstate+12

#define VBICitzf VBICstate+13
#define VBICitzf_Vbei VBICstate+14
#define VBICitzf_Vbci VBICstate+15

#define VBICitzr VBICstate+16
#define VBICitzr_Vbci VBICstate+17
#define VBICitzr_Vbei VBICstate+18

#define VBICibc VBICstate+19
#define VBICibc_Vbci VBICstate+20
#define VBICibc_Vbei VBICstate+21

#define VBICibep VBICstate+22
#define VBICibep_Vbep VBICstate+23

#define VBICirci VBICstate+24
#define VBICirci_Vrci VBICstate+25
#define VBICirci_Vbci VBICstate+26
#define VBICirci_Vbcx VBICstate+27

#define VBICirbi VBICstate+28
#define VBICirbi_Vrbi VBICstate+29
#define VBICirbi_Vbei VBICstate+30
#define VBICirbi_Vbci VBICstate+31

#define VBICirbp VBICstate+32
#define VBICirbp_Vrbp VBICstate+33
#define VBICirbp_Vbep VBICstate+34
#define VBICirbp_Vbci VBICstate+35


#define VBICqbe VBICstate+36
#define VBICcqbe VBICstate+37
#define VBICcqbeci VBICstate+38

#define VBICqbex VBICstate+39
#define VBICcqbex VBICstate+40

#define VBICqbc VBICstate+41
#define VBICcqbc VBICstate+42

#define VBICqbcx VBICstate+43
#define VBICcqbcx VBICstate+44

#define VBICqbep VBICstate+45
#define VBICcqbep VBICstate+46
#define VBICcqbepci VBICstate+47

#define VBICqbeo VBICstate+48
#define VBICcqbeo VBICstate+49
#define VBICgqbeo VBICstate+50

#define VBICqbco VBICstate+51
#define VBICcqbco VBICstate+52
#define VBICgqbco VBICstate+53

#define VBICibcp VBICstate+54
#define VBICibcp_Vbcp VBICstate+55

#define VBICiccp VBICstate+56
#define VBICiccp_Vbep VBICstate+57
#define VBICiccp_Vbci VBICstate+58
#define VBICiccp_Vbcp VBICstate+59

#define VBICqbcp VBICstate+60
#define VBICcqbcp VBICstate+61

#define VBICnumStates 62

#define VBICsensxpbe VBICstate+64 /* charge sensitivities and their
                   derivatives. +65 for the derivatives -
                   pointer to the beginning of the array */
#define VBICsensxpbex VBICstate+66
#define VBICsensxpbc VBICstate+68
#define VBICsensxpbcx VBICstate+70
#define VBICsensxpbep VBICstate+72

#define VBICnumSenStates 10


/* per model data */
typedef struct sVBICmodel {           /* model structure for a vbic */
    int VBICmodType;                  /* type index of this device type */
    struct sVBICmodel *VBICnextModel; /* pointer to next possible model in 
                                         linked list */
    VBICinstance * VBICinstances;     /* pointer to list of instances that have
                                         this model */
    IFuid VBICmodName;                /* pointer to character string naming 
                                         this model */
    int VBICtype;

    double VBICtnom;
    double VBICextCollResist;
    double VBICintCollResist;
    double VBICepiSatVoltage;
    double VBICepiDoping;
    double VBIChighCurFac;
    double VBICextBaseResist;
    double VBICintBaseResist;
    double VBICemitterResist;
    double VBICsubstrateResist;
    double VBICparBaseResist;
    double VBICsatCur;
    double VBICemissionCoeffF;
    double VBICemissionCoeffR;
    double VBICdeplCapLimitF;
    double VBICextOverlapCapBE;
    double VBICdepletionCapBE;
    double VBICpotentialBE;
    double VBICjunctionExpBE;
    double VBICsmoothCapBE;
    double VBICextOverlapCapBC;
    double VBICdepletionCapBC;
    double VBICepiCharge;
    double VBICextCapBC;
    double VBICpotentialBC;
    double VBICjunctionExpBC;
    double VBICsmoothCapBC;
    double VBICextCapSC;
    double VBICpotentialSC;
    double VBICjunctionExpSC;
    double VBICsmoothCapSC;
    double VBICidealSatCurBE;
    double VBICportionIBEI;
    double VBICidealEmissCoeffBE;
    double VBICnidealSatCurBE;
    double VBICnidealEmissCoeffBE;
    double VBICidealSatCurBC;
    double VBICidealEmissCoeffBC;
    double VBICnidealSatCurBC;
    double VBICnidealEmissCoeffBC;
    double VBICavalanchePar1BC;
    double VBICavalanchePar2BC;
    double VBICparasitSatCur;
    double VBICportionICCP;
    double VBICparasitFwdEmissCoeff;
    double VBICidealParasitSatCurBE;
    double VBICnidealParasitSatCurBE;
    double VBICidealParasitSatCurBC;
    double VBICidealParasitEmissCoeffBC;
    double VBICnidealParasitSatCurBC;
    double VBICnidealParasitEmissCoeffBC;
    double VBICearlyVoltF;
    double VBICearlyVoltR;
    double VBICrollOffF;
    double VBICrollOffR;
    double VBICparRollOff;
    double VBICtransitTimeF;
    double VBICvarTransitTimeF;
    double VBICtransitTimeBiasCoeffF;
    double VBICtransitTimeFVBC;
    double VBICtransitTimeHighCurrentF;
    double VBICtransitTimeR;
    double VBICdelayTimeF;
    double VBICfNcoef;
    double VBICfNexpA;
    double VBICfNexpB;
    double VBICtempExpRE;
    double VBICtempExpRBI;
    double VBICtempExpRCI;
    double VBICtempExpRS;
    double VBICtempExpVO;
    double VBICactivEnergyEA;
    double VBICactivEnergyEAIE;
    double VBICactivEnergyEAIC;
    double VBICactivEnergyEAIS;
    double VBICactivEnergyEANE;
    double VBICactivEnergyEANC;
    double VBICactivEnergyEANS;
    double VBICtempExpIS;
    double VBICtempExpII;
    double VBICtempExpIN;
    double VBICtempExpNF;
    double VBICtempExpAVC;
    double VBICthermalResist;
    double VBICthermalCapacitance;
    double VBICpunchThroughVoltageBC;
    double VBICdeplCapCoeff1;
    double VBICfixedCapacitanceCS;
    double VBICsgpQBselector;
    double VBIChighCurrentBetaRolloff;
    double VBICtempExpIKF;
    double VBICtempExpRCX;
    double VBICtempExpRBX;
    double VBICtempExpRBP;
    double VBICsepISRR;
    double VBICtempExpXISR;
    double VBICdear;
    double VBICeap;
    double VBICvbbe;
    double VBICnbbe;
    double VBICibbe;
    double VBICtvbbe1;
    double VBICtvbbe2;
    double VBICtnbbe;
    double VBICebbe;
    double VBIClocTempDiff;
    double VBICrevVersion;
    double VBICrefVersion;

    double VBICcollectorConduct; /* collector conductance */
    double VBICbaseConduct;      /* base conductance */
    double VBICemitterConduct;   /* emitter conductance */
    double VBICsubstrateConduct; /* substrate conductance */

    unsigned VBICtnomGiven : 1;
    unsigned VBICextCollResistGiven : 1;
    unsigned VBICintCollResistGiven : 1;
    unsigned VBICepiSatVoltageGiven : 1;
    unsigned VBICepiDopingGiven : 1;
    unsigned VBIChighCurFacGiven : 1;
    unsigned VBICextBaseResistGiven : 1;
    unsigned VBICintBaseResistGiven : 1;
    unsigned VBICemitterResistGiven : 1;
    unsigned VBICsubstrateResistGiven : 1;
    unsigned VBICparBaseResistGiven : 1;
    unsigned VBICsatCurGiven : 1;
    unsigned VBICemissionCoeffFGiven : 1;
    unsigned VBICemissionCoeffRGiven : 1;
    unsigned VBICdeplCapLimitFGiven : 1;
    unsigned VBICextOverlapCapBEGiven : 1;
    unsigned VBICdepletionCapBEGiven : 1;
    unsigned VBICpotentialBEGiven : 1;
    unsigned VBICjunctionExpBEGiven : 1;
    unsigned VBICsmoothCapBEGiven : 1;
    unsigned VBICextOverlapCapBCGiven : 1;
    unsigned VBICdepletionCapBCGiven : 1;
    unsigned VBICepiChargeGiven : 1;
    unsigned VBICextCapBCGiven : 1;
    unsigned VBICpotentialBCGiven : 1;
    unsigned VBICjunctionExpBCGiven : 1;
    unsigned VBICsmoothCapBCGiven : 1;
    unsigned VBICextCapSCGiven : 1;
    unsigned VBICpotentialSCGiven : 1;
    unsigned VBICjunctionExpSCGiven : 1;
    unsigned VBICsmoothCapSCGiven : 1;
    unsigned VBICidealSatCurBEGiven : 1;
    unsigned VBICportionIBEIGiven : 1;
    unsigned VBICidealEmissCoeffBEGiven : 1;
    unsigned VBICnidealSatCurBEGiven : 1;
    unsigned VBICnidealEmissCoeffBEGiven : 1;
    unsigned VBICidealSatCurBCGiven : 1;
    unsigned VBICidealEmissCoeffBCGiven : 1;
    unsigned VBICnidealSatCurBCGiven : 1;
    unsigned VBICnidealEmissCoeffBCGiven : 1;
    unsigned VBICavalanchePar1BCGiven : 1;
    unsigned VBICavalanchePar2BCGiven : 1;
    unsigned VBICparasitSatCurGiven : 1;
    unsigned VBICportionICCPGiven : 1;
    unsigned VBICparasitFwdEmissCoeffGiven : 1;
    unsigned VBICidealParasitSatCurBEGiven : 1;
    unsigned VBICnidealParasitSatCurBEGiven : 1;
    unsigned VBICidealParasitSatCurBCGiven : 1;
    unsigned VBICidealParasitEmissCoeffBCGiven : 1;
    unsigned VBICnidealParasitSatCurBCGiven : 1;
    unsigned VBICnidealParasitEmissCoeffBCGiven : 1;
    unsigned VBICearlyVoltFGiven : 1;
    unsigned VBICearlyVoltRGiven : 1;
    unsigned VBICrollOffFGiven : 1;
    unsigned VBICrollOffRGiven : 1;
    unsigned VBICparRollOffGiven : 1;
    unsigned VBICtransitTimeFGiven : 1;
    unsigned VBICvarTransitTimeFGiven : 1;
    unsigned VBICtransitTimeBiasCoeffFGiven : 1;
    unsigned VBICtransitTimeFVBCGiven : 1;
    unsigned VBICtransitTimeHighCurrentFGiven : 1;
    unsigned VBICtransitTimeRGiven : 1;
    unsigned VBICdelayTimeFGiven : 1;
    unsigned VBICfNcoefGiven : 1;
    unsigned VBICfNexpAGiven : 1;
    unsigned VBICfNexpBGiven : 1;
    unsigned VBICtempExpREGiven : 1;
    unsigned VBICtempExpRBIGiven : 1;
    unsigned VBICtempExpRCIGiven : 1;
    unsigned VBICtempExpRSGiven : 1;
    unsigned VBICtempExpVOGiven : 1;
    unsigned VBICactivEnergyEAGiven : 1;
    unsigned VBICactivEnergyEAIEGiven : 1;
    unsigned VBICactivEnergyEAICGiven : 1;
    unsigned VBICactivEnergyEAISGiven : 1;
    unsigned VBICactivEnergyEANEGiven : 1;
    unsigned VBICactivEnergyEANCGiven : 1;
    unsigned VBICactivEnergyEANSGiven : 1;
    unsigned VBICtempExpISGiven : 1;
    unsigned VBICtempExpIIGiven : 1;
    unsigned VBICtempExpINGiven : 1;
    unsigned VBICtempExpNFGiven : 1;
    unsigned VBICtempExpAVCGiven : 1;
    unsigned VBICthermalResistGiven : 1;
    unsigned VBICthermalCapacitanceGiven : 1;
    unsigned VBICpunchThroughVoltageBCGiven : 1;
    unsigned VBICdeplCapCoeff1Given : 1;
    unsigned VBICfixedCapacitanceCSGiven : 1;
    unsigned VBICsgpQBselectorGiven : 1;
    unsigned VBIChighCurrentBetaRolloffGiven : 1;
    unsigned VBICtempExpIKFGiven : 1;
    unsigned VBICtempExpRCXGiven : 1;
    unsigned VBICtempExpRBXGiven : 1;
    unsigned VBICtempExpRBPGiven : 1;
    unsigned VBICsepISRRGiven : 1;
    unsigned VBICtempExpXISRGiven : 1;
    unsigned VBICdearGiven : 1;
    unsigned VBICeapGiven : 1;
    unsigned VBICvbbeGiven : 1;
    unsigned VBICnbbeGiven : 1;
    unsigned VBICibbeGiven : 1;
    unsigned VBICtvbbe1Given : 1;
    unsigned VBICtvbbe2Given : 1;
    unsigned VBICtnbbeGiven : 1;
    unsigned VBICebbeGiven : 1;
    unsigned VBIClocTempDiffGiven : 1;
    unsigned VBICrevVersionGiven : 1;
    unsigned VBICrefVersionGiven : 1;
} VBICmodel;

#ifndef NPN
#define NPN 1
#define PNP -1
#endif /*NPN*/

/* device parameters */
#define VBIC_AREA 1
#define VBIC_OFF 2
#define VBIC_IC 3
#define VBIC_IC_VBE 4
#define VBIC_IC_VCE 5
#define VBIC_TEMP  6
#define VBIC_DTEMP 7
#define VBIC_M 8

/* model parameters */
#define VBIC_MOD_NPN    101 
#define VBIC_MOD_PNP    102 
#define VBIC_MOD_TNOM   103 
#define VBIC_MOD_RCX    104 
#define VBIC_MOD_RCI    105 
#define VBIC_MOD_VO     106 
#define VBIC_MOD_GAMM   107 
#define VBIC_MOD_HRCF   108 
#define VBIC_MOD_RBX    109 
#define VBIC_MOD_RBI    110 
#define VBIC_MOD_RE     111 
#define VBIC_MOD_RS     112 
#define VBIC_MOD_RBP    113 
#define VBIC_MOD_IS     114 
#define VBIC_MOD_NF     115 
#define VBIC_MOD_NR     116 
#define VBIC_MOD_FC     117 
#define VBIC_MOD_CBEO   118 
#define VBIC_MOD_CJE    119 
#define VBIC_MOD_PE     120 
#define VBIC_MOD_ME     121 
#define VBIC_MOD_AJE    122 
#define VBIC_MOD_CBCO   123 
#define VBIC_MOD_CJC    124 
#define VBIC_MOD_QCO    125 
#define VBIC_MOD_CJEP   126 
#define VBIC_MOD_PC     127 
#define VBIC_MOD_MC     128 
#define VBIC_MOD_AJC    129 
#define VBIC_MOD_CJCP   130 
#define VBIC_MOD_PS     131 
#define VBIC_MOD_MS     132 
#define VBIC_MOD_AJS    133 
#define VBIC_MOD_IBEI   134 
#define VBIC_MOD_WBE    135 
#define VBIC_MOD_NEI    136 
#define VBIC_MOD_IBEN   137 
#define VBIC_MOD_NEN    138 
#define VBIC_MOD_IBCI   139 
#define VBIC_MOD_NCI    140 
#define VBIC_MOD_IBCN   141 
#define VBIC_MOD_NCN    142 
#define VBIC_MOD_AVC1   143 
#define VBIC_MOD_AVC2   144 
#define VBIC_MOD_ISP    145 
#define VBIC_MOD_WSP    146 
#define VBIC_MOD_NFP    147 
#define VBIC_MOD_IBEIP  148 
#define VBIC_MOD_IBENP  149 
#define VBIC_MOD_IBCIP  150 
#define VBIC_MOD_NCIP   151 
#define VBIC_MOD_IBCNP  152 
#define VBIC_MOD_NCNP   153 
#define VBIC_MOD_VEF    154 
#define VBIC_MOD_VER    155 
#define VBIC_MOD_IKF    156 
#define VBIC_MOD_IKR    157 
#define VBIC_MOD_IKP    158   
#define VBIC_MOD_TF     159 
#define VBIC_MOD_QTF    160 
#define VBIC_MOD_XTF    161 
#define VBIC_MOD_VTF    162 
#define VBIC_MOD_ITF    163 
#define VBIC_MOD_TR     164 
#define VBIC_MOD_TD     165 
#define VBIC_MOD_KFN    166 
#define VBIC_MOD_AFN    167 
#define VBIC_MOD_BFN    168 
#define VBIC_MOD_XRE    169 
#define VBIC_MOD_XRBI   170 
#define VBIC_MOD_XRCI   171 
#define VBIC_MOD_XRS    172 
#define VBIC_MOD_XVO    173 
#define VBIC_MOD_EA     174 
#define VBIC_MOD_EAIE   175 
#define VBIC_MOD_EAIC   176 
#define VBIC_MOD_EAIS   177 
#define VBIC_MOD_EANE   178 
#define VBIC_MOD_EANC   179 
#define VBIC_MOD_EANS   180 
#define VBIC_MOD_XIS    181 
#define VBIC_MOD_XII    182 
#define VBIC_MOD_XIN    183 
#define VBIC_MOD_TNF    184 
#define VBIC_MOD_TAVC   185 
#define VBIC_MOD_RTH    186 
#define VBIC_MOD_CTH    187 
#define VBIC_MOD_VRT    188 
#define VBIC_MOD_ART    189 
#define VBIC_MOD_CCSO   190 
#define VBIC_MOD_QBM    191 
#define VBIC_MOD_NKF    192 
#define VBIC_MOD_XIKF   193 
#define VBIC_MOD_XRCX   194 
#define VBIC_MOD_XRBX   195 
#define VBIC_MOD_XRBP   196 
#define VBIC_MOD_ISRR   197 
#define VBIC_MOD_XISR   198 
#define VBIC_MOD_DEAR   199 
#define VBIC_MOD_EAP    200 
#define VBIC_MOD_VBBE   201 
#define VBIC_MOD_NBBE   202
#define VBIC_MOD_IBBE   203
#define VBIC_MOD_TVBBE1 204 
#define VBIC_MOD_TVBBE2 205 
#define VBIC_MOD_TNBBE  206 
#define VBIC_MOD_EBBE   207 
#define VBIC_MOD_DTEMP  208 
#define VBIC_MOD_VERS   209
#define VBIC_MOD_VREF   210

                              
/* device questions */        
#define VBIC_QUEST_FT             211
#define VBIC_QUEST_COLLNODE       212
#define VBIC_QUEST_BASENODE       213
#define VBIC_QUEST_EMITNODE       214
#define VBIC_QUEST_SUBSNODE       215
#define VBIC_QUEST_COLLCXNODE     216
#define VBIC_QUEST_COLLCINODE     217
#define VBIC_QUEST_BASEBXNODE     218
#define VBIC_QUEST_BASEBINODE     219
#define VBIC_QUEST_BASEBPNODE     220
#define VBIC_QUEST_EMITEINODE     221
#define VBIC_QUEST_SUBSSINODE     222
#define VBIC_QUEST_VBE            223
#define VBIC_QUEST_VBC            224
#define VBIC_QUEST_CC             225
#define VBIC_QUEST_CB             226
#define VBIC_QUEST_CE             227
#define VBIC_QUEST_CS             228
#define VBIC_QUEST_GM             229
#define VBIC_QUEST_GO             230
#define VBIC_QUEST_GPI            231
#define VBIC_QUEST_GMU            232
#define VBIC_QUEST_GX             233
#define VBIC_QUEST_QBE            234
#define VBIC_QUEST_CQBE           235
#define VBIC_QUEST_QBC            236
#define VBIC_QUEST_CQBC           237
#define VBIC_QUEST_QBX            238
#define VBIC_QUEST_CQBX           239
#define VBIC_QUEST_QBCP           240
#define VBIC_QUEST_CQBCP          241
#define VBIC_QUEST_CEXBC          242
#define VBIC_QUEST_GEQCB          243
#define VBIC_QUEST_GCSUB          244
#define VBIC_QUEST_GDSUB          245
#define VBIC_QUEST_GEQBX          246
#define VBIC_QUEST_CBE            247
#define VBIC_QUEST_CBEX           248
#define VBIC_QUEST_CBC            249
#define VBIC_QUEST_CBCX           250
#define VBIC_QUEST_CBEP           251
#define VBIC_QUEST_CBCP           252
#define VBIC_QUEST_SENS_REAL      253
#define VBIC_QUEST_SENS_IMAG      254
#define VBIC_QUEST_SENS_MAG       255
#define VBIC_QUEST_SENS_PH        256
#define VBIC_QUEST_SENS_CPLX      257
#define VBIC_QUEST_SENS_DC        258
#define VBIC_QUEST_POWER          259

/* model questions */
#define VBIC_MOD_COLLCONDUCT           301
#define VBIC_MOD_BASECONDUCT           302
#define VBIC_MOD_EMITTERCONDUCT        303
#define VBIC_MOD_SUBSTRATECONDUCT      304
#define VBIC_MOD_TYPE                  305

#include "vbicext.h"
#endif /*VBIC*/
