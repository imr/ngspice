/* Configuration file for ng-spice */
#include <config.h>

#include "conf.h"

/*
 * Analyses
 */
#define AN_op
#define AN_dc
#define AN_tf
#define AN_ac
#define AN_tran
#define AN_pz
#define AN_disto
#define AN_noise
#define AN_sense

/*
 * Devices
 */
#define DEV_asrc
#define DEV_bjt
#define DEV_bsim1
#define DEV_bsim2
#define DEV_bsim3
#define DEV_bsim4
#define DEV_bsim3v1
#define DEV_bsim3v2
#define DEV_cap
#define DEV_cccs
#define DEV_ccvs
#define DEV_csw
#define DEV_dio
#define DEV_ind
#define DEV_isrc
#define DEV_jfet
#define DEV_jfet2
#define DEV_ltra
#define DEV_mes
#define DEV_mos1
#define DEV_mos2
#define DEV_mos3
#define DEV_mos6
#define DEV_res
#define DEV_sw
#define DEV_tra
#define DEV_urc
#define DEV_vccs
#define DEV_vcvs
#define DEV_vsrc

#define DEVICES_USED "asrc bjt bsim1 bsim2 bsim3 bsim3v2 bsim3v1 cap cccs ccvs csw dio ind isrc jfet ltra mes mos1 mos2 mos3 mos6 res sw tra urc vccs vcvs vsrc"
#define ANALYSES_USED "op dc tf ac tran pz disto noise sense"

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice.h"


#define CONFIG



#include <stdio.h>
#include "noisedef.h"
#include "devdefs.h"
#include "suffix.h"

#include "asrc/asrcitf.h"
#include "bjt/bjtitf.h"
#include "cap/capitf.h"
#include "cccs/cccsitf.h"
#include "ccvs/ccvsitf.h"
#include "csw/cswitf.h"
#include "dio/dioitf.h"
#include "ind/inditf.h"
#include "isrc/isrcitf.h"
#include "mos1/mos1itf.h"
#include "mos6/mos6itf.h"
#include "res/resitf.h"
#include "sw/switf.h"
#include "vccs/vccsitf.h"
#include "vcvs/vcvsitf.h"
#include "vsrc/vsrcitf.h"
#include "bsim1/bsim1itf.h"
#include "bsim2/bsim2itf.h"
#include "bsim3/bsim3itf.h"
#include "bsim4/bsim4itf.h"
#include "bsim3v1/bsim3v1itf.h"
#include "bsim3v2/bsim3v2itf.h"
#include "mos2/mos2itf.h"
#include "mos3/mos3itf.h"
#include "jfet/jfetitf.h"
#include "jfet2/jfet2itf.h"
#include "mes/mesitf.h"
#include "ltra/ltraitf.h"
#include "tra/traitf.h"
#include "urc/urcitf.h"



extern SPICEanalysis OPTinfo;
extern SPICEanalysis ACinfo;
extern SPICEanalysis DCTinfo;
extern SPICEanalysis DCOinfo;
extern SPICEanalysis TRANinfo;
extern SPICEanalysis PZinfo;
extern SPICEanalysis TFinfo;
extern SPICEanalysis DISTOinfo;
extern SPICEanalysis NOISEinfo;
extern SPICEanalysis SENSinfo;


SPICEanalysis *analInfo[] = {
    &OPTinfo,
    &ACinfo,
    &DCTinfo,
    &DCOinfo,
    &TRANinfo,
    &PZinfo,
    &TFinfo,
    &DISTOinfo,
    &NOISEinfo,
    &SENSinfo,

};

int ANALmaxnum = sizeof(analInfo)/sizeof(SPICEanalysis*);
SPICEdev *DEVices[] = {

	/* URC must appear before the resistor, capacitor, and diode */
        &URCinfo,
        &ASRCinfo,
        &BJTinfo,
        &B1info,
        &B2info,
        &BSIM3info,
	&B4info,
	&BSIM3V2info,
	&BSIM3V1info,
        &CAPinfo,
        &CCCSinfo,
        &CCVSinfo,
        &CSWinfo,
        &DIOinfo,
        &INDinfo,
        &MUTinfo,
        &ISRCinfo,
        &JFETinfo,
        &JFET2info,
        &LTRAinfo,
        &MESinfo,
        &MOS1info,
        &MOS2info,
        &MOS3info,
        &MOS6info,
        &RESinfo,
        &SWinfo,
        &TRAinfo,
        &VCCSinfo,
        &VCVSinfo,
        &VSRCinfo,
};

/* my internal global constant for number of device types */
int DEVmaxnum = sizeof(DEVices)/sizeof(SPICEdev *);
/* XXX Should be -1 ? There is always an extra null element at the end ? */
static char * specSigList[] = {
    "time"
};

static IFparm nodeParms[] = {
    IP( "nodeset",PARM_NS ,IF_REAL,"suggested initial voltage"),
    IP( "ic",PARM_IC ,IF_REAL,"initial voltage"),
    IP( "type",PARM_NODETYPE ,IF_INTEGER,"output type of equation")
};

IFsimulator SIMinfo = {
    "ngspice",      /* name */
    "Circuit level simulation program",  /* more about me */
    Spice_Version,  /* version */

    CKTinit,        /* newCircuit function */
    CKTdestroy,     /* deleteCircuit function */

    CKTnewNode,     /* newNode function */
    CKTground,      /* groundNode function */
    CKTbindNode,    /* bindNode function */
    CKTfndNode,     /* findNode function */
    CKTinst2Node,   /* instToNode function */
    CKTsetNodPm,    /* setNodeParm function */
    CKTaskNodQst,   /* askNodeQuest function */
    CKTdltNod,      /* deleteNode function */

    CKTcrtElt,      /* newInstance function */
    CKTparam,       /* setInstanceParm function */
    CKTask,         /* askInstanceQuest function */
    CKTfndDev,      /* findInstance funciton */
    CKTdltInst,     /* deleteInstance function */

    CKTmodCrt,      /* newModel function */
    CKTmodParam,    /* setModelParm function */
    CKTmodAsk,      /* askModelQuest function */
    CKTfndMod,      /* findModel function */
    CKTdltMod,      /* deleteModel function */

    CKTnewTask,     /* newTask function */
    CKTnewAnal,     /* newAnalysis function */
    CKTsetAnalPm,   /* setAnalysisParm function */
    CKTaskAnalQ,    /* askAnalysisQuest function */
    CKTfndAnal,     /* findAnalysis function */
    CKTfndTask,     /* findTask function */
    CKTdelTask,     /* deleteTask function */

    CKTdoJob,       /* doAnalyses function */
    CKTtrouble,	    /* non-convergence message function */

    sizeof(DEVices)/sizeof(SPICEdev *),
    (IFdevice**)DEVices,

    sizeof(analInfo)/sizeof(SPICEanalysis *),
    (IFanalysis **)analInfo,

    sizeof(nodeParms)/sizeof(IFparm),
    nodeParms,

    sizeof(specSigList)/sizeof(char *),
    specSigList,
};
