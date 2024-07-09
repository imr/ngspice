/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

#ifndef VBIC
#define VBIC

#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/gendefs.h"
#include "ngspice/complex.h"
#include "ngspice/noisedef.h"

/* structures to describe Bipolar Junction Transistors */

/* indices to array of VBIC noise sources */

enum {
    VBICRCNOIZ = 0,
    VBICRCINOIZ,
    VBICRBNOIZ,
    VBICRBINOIZ,
    VBICRENOIZ,
    VBICRBPNOIZ,
    VBICRSNOIZ,
    VBICICNOIZ,
    VBICIBNOIZ,
    VBICIBEPNOIZ,
    VBICICCPNOIZ,
    VBICFLBENOIZ,
    VBICFLBEPNOIZ,
    VBICTOTNOIZ,
    /* finally, the number of noise sources */
    VBICNSRCS
};

/* data needed to describe a single instance */

typedef struct sVBICinstance {

    struct GENinstance gen;

#define VBICmodPtr(inst) ((struct sVBICmodel *)((inst)->gen.GENmodPtr))
#define VBICnextInstance(inst) ((struct sVBICinstance *)((inst)->gen.GENnextInstance))
#define VBICname gen.GENname
#define VBICstate gen.GENstate

    const int VBICcollNode;   /* number of collector node of vbic */
    const int VBICbaseNode;   /* number of base node of vbic */
    const int VBICemitNode;   /* number of emitter node of vbic */
    const int VBICsubsNode;   /* number of substrate node of vbic */
    const int VBICtempNode;  /* number of the temperature node of the vbic */
    int VBICcollCXNode; /* number of internal collector node of vbic */
    int VBICcollCINode; /* number of internal collector node of vbic */
    int VBICbaseBXNode; /* number of internal base node of vbic */
    int VBICbaseBINode; /* number of internal base node of vbic */
    int VBICemitEINode; /* number of internal emitter node of vbic */
    int VBICbaseBPNode; /* number of internal base node of vbic */
    int VBICsubsSINode; /* number of internal substrate node */
    int VBICxf1Node;    /* number of internal excess phase 1 node itf */
    int VBICxf2Node;    /* number of internal excess phase 2 node itf */
    int VBICbrEq;    /* number of the branch equation added for current */

    double VBICarea;     /* area factor for the vbic */
    double VBICicVBE;    /* initial condition voltage B-E*/
    double VBICicVCE;    /* initial condition voltage C-E*/
    double VBICtemp;     /* instance temperature */
    double VBICdtemp;    /* instance delta temperature */
    double VBICm;        /* multiply factor for the vbic */

    double VBICtVcrit;
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
    double *VBICemitEIXfPtr;   /* pointer to sparse matrix at
                             * (emitter prime,xf) */
    double *VBICbaseBIXfPtr;   /* pointer to sparse matrix at
                             * (base prime,xf) */

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

    /* self heating */
    double *VBICcollTempPtr;
    double *VBICbaseTempPtr;
    double *VBICemitTempPtr;
    double *VBICsubsTempPtr;
    double *VBICcollCItempPtr;
    double *VBICcollCXtempPtr;
    double *VBICbaseBItempPtr;
    double *VBICbaseBXtempPtr;
    double *VBICbaseBPtempPtr;
    double *VBICemitEItempPtr;
    double *VBICsubsSItempPtr;
    double *VBICtempCollPtr;
    double *VBICtempCollCIPtr;
    double *VBICtempCollCXPtr;
    double *VBICtempBasePtr;
    double *VBICtempBaseBIPtr;
    double *VBICtempBaseBXPtr;
    double *VBICtempBaseBPPtr;
    double *VBICtempEmitPtr;
    double *VBICtempEmitEIPtr;
    double *VBICtempSubsPtr;
    double *VBICtempSubsSIPtr;
    double *VBICtempTempPtr;

    /* excess phase */
    double *VBICtempXf2Ptr;
    double *VBICxf1TempPtr;

    double *VBICxf1Xf1Ptr;
    double *VBICxf1Xf2Ptr;
    double *VBICxf1CollCIPtr;
    double *VBICxf1BaseBIPtr;
    double *VBICxf1EmitEIPtr;

    double *VBICxf2Xf2Ptr;
    double *VBICxf2Xf1Ptr;
    double *VBICcollCIXf2Ptr;
    double *VBICbaseBIXf2Ptr;
    double *VBICemitEIXf2Ptr;

    double *VBICxf1IbrPtr;
    double *VBICxf2IbrPtr;
    double *VBICibrXf1Ptr;
    double *VBICibrXf2Ptr;
    double *VBICibrIbrPtr;

    unsigned VBICareaGiven   :1; /* flag to indicate area was specified */
    unsigned VBICoff         :1; /* 'off' flag for vbic */
    unsigned VBICicVBEGiven  :1; /* flag to indicate VBE init. cond. given */
    unsigned VBICicVCEGiven  :1; /* flag to indicate VCE init. cond. given */
    unsigned VBICtempGiven   :1; /* temperature given for vbic instance*/
    unsigned VBICdtempGiven  :1; /* delta temperature given for vbic instance*/
    unsigned VBICmGiven      :1; /* flag to indicate multiplier was specified */

    double VBICcapbe;
    double VBICcapbex;
    double VBICcapbc;
    double VBICcapbcx;
    double VBICcapbep;
    double VBICcapbcp;
    double VBICcapcth;

    double VBICcapqbeth;
    double VBICcapqbexth;
    double VBICcapqbcth;
    double VBICcapqbcxth;
    double VBICcapqbepth;
    double VBICcapqbcpth;

    double VBICibe_Vrth;
    double VBICibex_Vrth;
    double VBICitzf_vrth;
    double VBICitzr_Vrth;
    double VBICibc_Vrth;
    double VBICibep_Vrth;
    double VBICircx_Vrth;
    double VBICirci_Vrth;
    double VBICirbx_Vrth;
    double VBICirbi_Vrth;
    double VBICire_Vrth;
    double VBICirbp_Vrth;
    double VBICibcp_Vrth;
    double VBICiccp_Vrth;
    double VBICirs_Vrth;
    double VBICirth_Vrth;
    double VBICith_Vrth;

    double VBICith_Vbei;
    double VBICith_Vbci;
    double VBICith_Vcei;
    double VBICith_Vbex;
    double VBICith_Vbep;
    double VBICith_Vbcp;
    double VBICqf_Vxf;
    double VBICith_Vcep;
    double VBICith_Vrci;
    double VBICith_Vbcx;
    double VBICith_Vrbi;
    double VBICith_Vrbp;
    double VBICith_Vrcx;
    double VBICith_Vrbx;
    double VBICith_Vre;
    double VBICith_Vrs;

    double VBICindInduct;

    int VBIC_selfheat;    /* self-heating enabled  */
    int VBIC_excessPhase; /* excess phase enabled  */

#ifndef NONOISE
      double VBICnVar[NSTATVARS][VBICNSRCS];
#else /*NONOISE*/
      double **VBICnVar;
#endif /*NONOISE*/
/* the above to avoid allocating memory when it is not needed */

#ifdef KLU
    BindElement *VBICcollCollBinding ;
    BindElement *VBICbaseBaseBinding ;
    BindElement *VBICemitEmitBinding ;
    BindElement *VBICsubsSubsBinding ;
    BindElement *VBICcollCXCollCXBinding ;
    BindElement *VBICcollCICollCIBinding ;
    BindElement *VBICbaseBXBaseBXBinding ;
    BindElement *VBICbaseBIBaseBIBinding ;
    BindElement *VBICemitEIEmitEIBinding ;
    BindElement *VBICbaseBPBaseBPBinding ;
    BindElement *VBICsubsSISubsSIBinding ;
    BindElement *VBICbaseEmitBinding ;
    BindElement *VBICemitBaseBinding ;
    BindElement *VBICbaseCollBinding ;
    BindElement *VBICcollBaseBinding ;
    BindElement *VBICcollCollCXBinding ;
    BindElement *VBICbaseBaseBXBinding ;
    BindElement *VBICemitEmitEIBinding ;
    BindElement *VBICsubsSubsSIBinding ;
    BindElement *VBICcollCXCollCIBinding ;
    BindElement *VBICcollCXBaseBXBinding ;
    BindElement *VBICcollCXBaseBIBinding ;
    BindElement *VBICcollCXBaseBPBinding ;
    BindElement *VBICcollCIBaseBIBinding ;
    BindElement *VBICcollCIEmitEIBinding ;
    BindElement *VBICbaseBXBaseBIBinding ;
    BindElement *VBICbaseBXEmitEIBinding ;
    BindElement *VBICbaseBXBaseBPBinding ;
    BindElement *VBICbaseBXSubsSIBinding ;
    BindElement *VBICbaseBIEmitEIBinding ;
    BindElement *VBICbaseBPSubsSIBinding ;
    BindElement *VBICcollCXCollBinding ;
    BindElement *VBICbaseBXBaseBinding ;
    BindElement *VBICemitEIEmitBinding ;
    BindElement *VBICsubsSISubsBinding ;
    BindElement *VBICcollCICollCXBinding ;
    BindElement *VBICbaseBICollCXBinding ;
    BindElement *VBICbaseBPCollCXBinding ;
    BindElement *VBICbaseBXCollCIBinding ;
    BindElement *VBICbaseBICollCIBinding ;
    BindElement *VBICemitEICollCIBinding ;
    BindElement *VBICbaseBPCollCIBinding ;
    BindElement *VBICbaseBIBaseBXBinding ;
    BindElement *VBICemitEIBaseBXBinding ;
    BindElement *VBICbaseBPBaseBXBinding ;
    BindElement *VBICsubsSIBaseBXBinding ;
    BindElement *VBICemitEIBaseBIBinding ;
    BindElement *VBICbaseBPBaseBIBinding ;
    BindElement *VBICsubsSICollCIBinding ;
    BindElement *VBICsubsSIBaseBIBinding ;
    BindElement *VBICsubsSIBaseBPBinding ;
    BindElement *VBICcollTempBinding ;
    BindElement *VBICbaseTempBinding ;
    BindElement *VBICemitTempBinding ;
    BindElement *VBICsubsTempBinding ;
    BindElement *VBICcollCItempBinding ;
    BindElement *VBICcollCXtempBinding ;
    BindElement *VBICbaseBItempBinding ;
    BindElement *VBICbaseBXtempBinding ;
    BindElement *VBICbaseBPtempBinding ;
    BindElement *VBICemitEItempBinding ;
    BindElement *VBICsubsSItempBinding ;
    BindElement *VBICtempCollBinding ;
    BindElement *VBICtempCollCIBinding ;
    BindElement *VBICtempCollCXBinding ;
    BindElement *VBICtempBaseBIBinding ;
    BindElement *VBICtempBaseBinding ;
    BindElement *VBICtempBaseBXBinding ;
    BindElement *VBICtempBaseBPBinding ;
    BindElement *VBICtempEmitBinding ;
    BindElement *VBICtempEmitEIBinding ;
    BindElement *VBICtempSubsBinding ;
    BindElement *VBICtempSubsSIBinding ;
    BindElement *VBICtempTempBinding ;
    BindElement *VBICtempXf2Binding ;
    BindElement *VBICxf1TempBinding ;
    BindElement *VBICxf1Xf1Binding ;
    BindElement *VBICxf1Xf2Binding ;
    BindElement *VBICxf1CollCIBinding ;
    BindElement *VBICxf1BaseBIBinding ;
    BindElement *VBICxf1EmitEIBinding ;
    BindElement *VBICxf2Xf2Binding ;
    BindElement *VBICxf2Xf1Binding ;
    BindElement *VBICcollCIXf2Binding ;
    BindElement *VBICbaseBIXf2Binding ;
    BindElement *VBICemitEIXf2Binding ;
    BindElement *VBICxf1IbrBinding ;
    BindElement *VBICxf2IbrBinding ;
    BindElement *VBICibrXf2Binding ;
    BindElement *VBICibrXf1Binding ;
    BindElement *VBICibrIbrBinding ;
#endif

} VBICinstance ;

/* entries in the state vector for vbic: */

#define VBICvbei  VBICstate
#define VBICvbex  VBICstate+1
#define VBICvbci  VBICstate+2
#define VBICvbcx  VBICstate+3
#define VBICvbep  VBICstate+4
#define VBICvrci  VBICstate+5
#define VBICvrbi  VBICstate+6
#define VBICvrbp  VBICstate+7
#define VBICvbcp  VBICstate+8

#define VBICibe       VBICstate+9
#define VBICibe_Vbei  VBICstate+10

#define VBICibex      VBICstate+11
#define VBICibex_Vbex VBICstate+12

#define VBICitzf      VBICstate+13
#define VBICitzf_Vbei VBICstate+14
#define VBICitzf_Vbci VBICstate+15
#define VBICitzf_Vrth VBICstate+16

#define VBICitzr      VBICstate+17
#define VBICitzr_Vbci VBICstate+18
#define VBICitzr_Vbei VBICstate+19

#define VBICibc       VBICstate+20
#define VBICibc_Vbci  VBICstate+21
#define VBICibc_Vbei  VBICstate+22

#define VBICibep      VBICstate+23
#define VBICibep_Vbep VBICstate+24

#define VBICirci      VBICstate+25
#define VBICirci_Vrci VBICstate+26
#define VBICirci_Vbci VBICstate+27
#define VBICirci_Vbcx VBICstate+28

#define VBICirbi      VBICstate+29
#define VBICirbi_Vrbi VBICstate+30
#define VBICirbi_Vbei VBICstate+31
#define VBICirbi_Vbci VBICstate+32

#define VBICirbp      VBICstate+33
#define VBICirbp_Vrbp VBICstate+34
#define VBICirbp_Vbep VBICstate+35
#define VBICirbp_Vbci VBICstate+36


#define VBICqbe    VBICstate+37
#define VBICcqbe   VBICstate+38
#define VBICcqbeci VBICstate+39

#define VBICqbex   VBICstate+40
#define VBICcqbex  VBICstate+41

#define VBICqbc    VBICstate+42
#define VBICcqbc   VBICstate+43

#define VBICqbcx   VBICstate+44
#define VBICcqbcx  VBICstate+45

#define VBICqbep    VBICstate+46
#define VBICcqbep   VBICstate+47
#define VBICcqbepci VBICstate+48

#define VBICqbeo  VBICstate+49
#define VBICcqbeo VBICstate+50
#define VBICgqbeo VBICstate+51

#define VBICqbco  VBICstate+52
#define VBICcqbco VBICstate+53
#define VBICgqbco VBICstate+54

#define VBICibcp      VBICstate+55
#define VBICibcp_Vbcp VBICstate+56

#define VBICiccp      VBICstate+57
#define VBICiccp_Vbep VBICstate+58
#define VBICiccp_Vbci VBICstate+59
#define VBICiccp_Vbcp VBICstate+60

#define VBICqbcp      VBICstate+61
#define VBICcqbcp     VBICstate+62

#define VBICircx_Vrcx VBICstate+63
#define VBICirbx_Vrbx VBICstate+64
#define VBICirs_Vrs   VBICstate+65
#define VBICire_Vre   VBICstate+66

#define VBICqcth      VBICstate+67 /* thermal capacitor charge */
#define VBICcqcth     VBICstate+68 /* thermal capacitor current */

#define VBICvrth      VBICstate+69
#define VBICicth_Vrth VBICstate+70

#define VBICqcxf      VBICstate+71
#define VBICcqcxf     VBICstate+72
#define VBICgqcxf     VBICstate+73

#define VBICibc_Vrxf  VBICstate+74

#define VBICixzf      VBICstate+75
#define VBICixzf_Vbei VBICstate+76
#define VBICixzf_Vbci VBICstate+77
#define VBICixzf_Vrth VBICstate+78

#define VBICixxf      VBICstate+79
#define VBICixxf_Vrxf VBICstate+80

#define VBICitxf      VBICstate+81
#define VBICitxf_Vrxf VBICstate+82
#define VBICith_Vrxf  VBICstate+83

#define VBICindFlux   VBICstate+84
#define VBICindVolt   VBICstate+85

#define VBICnumStates 86

/* per model data */
typedef struct sVBICmodel {           /* model structure for a vbic */

    struct GENmodel gen;

#define VBICmodType gen.GENmodType
#define VBICnextModel(inst) ((struct sVBICmodel *)((inst)->gen.GENnextModel))
#define VBICinstances(inst) ((VBICinstance *)((inst)->gen.GENinstances))
#define VBICmodName gen.GENmodName

    int VBICtype;
    int VBICselft;
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

    double VBICvbeMax; /* maximum voltage over B-E junction */
    double VBICvbcMax; /* maximum voltage over B-C junction */
    double VBICvceMax; /* maximum voltage over C-E branch */
    double VBICvsubMax; /* maximum voltage over C-substrate branch */
    double VBICvbcfwdMax; /* maximum forward voltage over B-C junction */
    double VBICvbefwdMax; /* maximum forward voltage over C-E branch */
    double VBICvsubfwdMax; /* maximum forward voltage over C-substrate branch */

    unsigned VBICselftGiven : 1;
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
    unsigned VBICtempExpRBGiven : 1;
    unsigned VBICtempExpRBIGiven : 1;
    unsigned VBICtempExpRCGiven : 1;
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
    unsigned VBICvbeMaxGiven : 1;
    unsigned VBICvbcMaxGiven : 1;
    unsigned VBICvceMaxGiven : 1;
    unsigned VBICvsubMaxGiven : 1;
    unsigned VBICvbcfwdMaxGiven : 1;
    unsigned VBICvbefwdMaxGiven : 1;
    unsigned VBICvsubfwdMaxGiven : 1;
} VBICmodel;

#ifndef NPN
#define NPN 1
#define PNP -1
#endif /*NPN*/

/* device parameters */
enum {
    VBIC_AREA = 1,
    VBIC_OFF,
    VBIC_IC,
    VBIC_IC_VBE,
    VBIC_IC_VCE,
    VBIC_TEMP,
    VBIC_DTEMP,
    VBIC_M,
};

/* model parameters */
enum {
    VBIC_MOD_NPN = 101,
    VBIC_MOD_PNP,
    VBIC_MOD_TNOM,
    VBIC_MOD_RCX,
    VBIC_MOD_RCI,
    VBIC_MOD_VO,
    VBIC_MOD_GAMM,
    VBIC_MOD_HRCF,
    VBIC_MOD_RBX,
    VBIC_MOD_RBI,
    VBIC_MOD_RE,
    VBIC_MOD_RS,
    VBIC_MOD_RBP,
    VBIC_MOD_IS,
    VBIC_MOD_NF,
    VBIC_MOD_NR,
    VBIC_MOD_FC,
    VBIC_MOD_CBEO,
    VBIC_MOD_CJE,
    VBIC_MOD_PE,
    VBIC_MOD_ME,
    VBIC_MOD_AJE,
    VBIC_MOD_CBCO,
    VBIC_MOD_CJC,
    VBIC_MOD_QCO,
    VBIC_MOD_CJEP,
    VBIC_MOD_PC,
    VBIC_MOD_MC,
    VBIC_MOD_AJC,
    VBIC_MOD_CJCP,
    VBIC_MOD_PS,
    VBIC_MOD_MS,
    VBIC_MOD_AJS,
    VBIC_MOD_IBEI,
    VBIC_MOD_WBE,
    VBIC_MOD_NEI,
    VBIC_MOD_IBEN,
    VBIC_MOD_NEN,
    VBIC_MOD_IBCI,
    VBIC_MOD_NCI,
    VBIC_MOD_IBCN,
    VBIC_MOD_NCN,
    VBIC_MOD_AVC1,
    VBIC_MOD_AVC2,
    VBIC_MOD_ISP,
    VBIC_MOD_WSP,
    VBIC_MOD_NFP,
    VBIC_MOD_IBEIP,
    VBIC_MOD_IBENP,
    VBIC_MOD_IBCIP,
    VBIC_MOD_NCIP,
    VBIC_MOD_IBCNP,
    VBIC_MOD_NCNP,
    VBIC_MOD_VEF,
    VBIC_MOD_VER,
    VBIC_MOD_IKF,
    VBIC_MOD_IKR,
    VBIC_MOD_IKP,
    VBIC_MOD_TF,
    VBIC_MOD_QTF,
    VBIC_MOD_XTF,
    VBIC_MOD_VTF,
    VBIC_MOD_ITF,
    VBIC_MOD_TR,
    VBIC_MOD_TD,
    VBIC_MOD_KFN,
    VBIC_MOD_AFN,
    VBIC_MOD_BFN,
    VBIC_MOD_XRE,
    VBIC_MOD_XRB,
    VBIC_MOD_XRBI,
    VBIC_MOD_XRC,
    VBIC_MOD_XRCI,
    VBIC_MOD_XRS,
    VBIC_MOD_XVO,
    VBIC_MOD_EA,
    VBIC_MOD_EAIE,
    VBIC_MOD_EAIC,
    VBIC_MOD_EAIS,
    VBIC_MOD_EANE,
    VBIC_MOD_EANC,
    VBIC_MOD_EANS,
    VBIC_MOD_XIS,
    VBIC_MOD_XII,
    VBIC_MOD_XIN,
    VBIC_MOD_TNF,
    VBIC_MOD_TAVC,
    VBIC_MOD_RTH,
    VBIC_MOD_CTH,
    VBIC_MOD_VRT,
    VBIC_MOD_ART,
    VBIC_MOD_CCSO,
    VBIC_MOD_QBM,
    VBIC_MOD_NKF,
    VBIC_MOD_XIKF,
    VBIC_MOD_XRCX,
    VBIC_MOD_XRBX,
    VBIC_MOD_XRBP,
    VBIC_MOD_ISRR,
    VBIC_MOD_XISR,
    VBIC_MOD_DEAR,
    VBIC_MOD_EAP,
    VBIC_MOD_VBBE,
    VBIC_MOD_NBBE,
    VBIC_MOD_IBBE,
    VBIC_MOD_TVBBE1,
    VBIC_MOD_TVBBE2,
    VBIC_MOD_TNBBE,
    VBIC_MOD_EBBE,
    VBIC_MOD_DTEMP,
    VBIC_MOD_VERS,
    VBIC_MOD_VREF,
    VBIC_MOD_VBE_MAX,
    VBIC_MOD_VBC_MAX,
    VBIC_MOD_VCE_MAX,
    VBIC_MOD_VSUB_MAX,
    VBIC_MOD_VBEFWD_MAX,
    VBIC_MOD_VBCFWD_MAX,
    VBIC_MOD_VSUBFWD_MAX,
    VBIC_MOD_SELFT,
};

/* device questions */
enum {
    VBIC_QUEST_FT = 221,
    VBIC_QUEST_COLLNODE,
    VBIC_QUEST_BASENODE,
    VBIC_QUEST_EMITNODE,
    VBIC_QUEST_SUBSNODE,
    VBIC_QUEST_COLLCXNODE,
    VBIC_QUEST_COLLCINODE,
    VBIC_QUEST_BASEBXNODE,
    VBIC_QUEST_BASEBINODE,
    VBIC_QUEST_BASEBPNODE,
    VBIC_QUEST_EMITEINODE,
    VBIC_QUEST_SUBSSINODE,
    VBIC_QUEST_VBE,
    VBIC_QUEST_VBC,
    VBIC_QUEST_CC,
    VBIC_QUEST_CB,
    VBIC_QUEST_CE,
    VBIC_QUEST_CS,
    VBIC_QUEST_GM,
    VBIC_QUEST_GO,
    VBIC_QUEST_GPI,
    VBIC_QUEST_GMU,
    VBIC_QUEST_GX,
    VBIC_QUEST_QBE,
    VBIC_QUEST_CQBE,
    VBIC_QUEST_QBC,
    VBIC_QUEST_CQBC,
    VBIC_QUEST_QBX,
    VBIC_QUEST_CQBX,
    VBIC_QUEST_QBCP,
    VBIC_QUEST_CQBCP,
    VBIC_QUEST_CEXBC,
    VBIC_QUEST_GEQCB,
    VBIC_QUEST_GCSUB,
    VBIC_QUEST_GDSUB,
    VBIC_QUEST_GEQBX,
    VBIC_QUEST_CBE,
    VBIC_QUEST_CBEX,
    VBIC_QUEST_CBC,
    VBIC_QUEST_CBCX,
    VBIC_QUEST_CBEP,
    VBIC_QUEST_CBCP,
    VBIC_QUEST_POWER,
};

/* model questions */
enum {
    VBIC_MOD_COLLCONDUCT = 301,
    VBIC_MOD_BASECONDUCT,
    VBIC_MOD_EMITTERCONDUCT,
    VBIC_MOD_SUBSTRATECONDUCT,
    VBIC_MOD_TYPE,
};

#include "vbicext.h"
#endif /*VBIC*/
