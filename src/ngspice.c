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

#define ANALYSES_USED "op dc tf ac tran pz disto noise sense"

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice.h"


#define CONFIG



#include <stdio.h>
#include "devdefs.h"
#include "noisedef.h"
#include "suffix.h"


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
    "ngspice",			/* name */
    "Circuit level simulation program",	/* more about me */
    Spice_Version,		/* version */

    CKTinit,			/* newCircuit function */
    CKTdestroy,			/* deleteCircuit function */

    CKTnewNode,			/* newNode function */
    CKTground,			/* groundNode function */
    CKTbindNode,		/* bindNode function */
    CKTfndNode,			/* findNode function */
    CKTinst2Node,		/* instToNode function */
    CKTsetNodPm,		/* setNodeParm function */
    CKTaskNodQst,		/* askNodeQuest function */
    CKTdltNod,			/* deleteNode function */

    CKTcrtElt,			/* newInstance function */
    CKTparam,			/* setInstanceParm function */
    CKTask,			/* askInstanceQuest function */
    CKTfndDev,			/* findInstance funciton */
    CKTdltInst,			/* deleteInstance function */

    CKTmodCrt,			/* newModel function */
    CKTmodParam,		/* setModelParm function */
    CKTmodAsk,			/* askModelQuest function */
    CKTfndMod,			/* findModel function */
    CKTdltMod,			/* deleteModel function */

    CKTnewTask,			/* newTask function */
    CKTnewAnal,			/* newAnalysis function */
    CKTsetAnalPm,		/* setAnalysisParm function */
    CKTaskAnalQ,		/* askAnalysisQuest function */
    CKTfndAnal,			/* findAnalysis function */
    CKTfndTask,			/* findTask function */
    CKTdelTask,			/* deleteTask function */

    CKTdoJob,			/* doAnalyses function */
    CKTtrouble,			/* non-convergence message function */

    0,				/* Initialized in SIMinit() */
    NULL,			/* Initialized in SIMinit() */

    sizeof(analInfo)/sizeof(SPICEanalysis *),
    (IFanalysis **)analInfo,

    sizeof(nodeParms)/sizeof(IFparm),
    nodeParms,

    sizeof(specSigList)/sizeof(char *),
    specSigList,
};
