/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_res

#ifndef DEV_RES
#define DEV_RES

#include "resext.h"
extern IFparm RESpTable[ ];
extern IFparm RESmPTable[ ];
extern char *RESnames[ ];
extern int RESpTSize;
extern int RESmPTSize;
extern int RESnSize;
extern int RESiSize;
extern int RESmSize;

SPICEdev RESinfo = {
    {
        "Resistor",
        "Simple linear resistor",

        &RESnSize,
        &RESnSize,
        RESnames,

        &RESpTSize,
        RESpTable,

        &RESmPTSize,
        RESmPTable,
	0
    },

    RESparam,
    RESmParam,
    RESload,
    RESsetup,
    NULL,
    RESsetup,
    REStemp,
    NULL,
    NULL,
/* serban - added ac support */
    RESacload,  /* ac load and normal load are identical */
    NULL,
    RESdestroy,
#ifdef DELETES
    RESmDelete,
    RESdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    RESask,
    RESmodAsk,
#ifdef AN_pz
    RESpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
#ifdef AN_sense2
    RESsSetup,
    RESsLoad,
    NULL,
    RESsAcLoad,
    RESsPrint,
    NULL,
#else /* AN_sense2 */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense2 */
    NULL, /* Disto */
#ifdef AN_noise
    RESnoise,
#else	/* AN_noise */
    NULL,
#endif	/* AN_noise */

    &RESiSize,
    &RESmSize

};

#endif
#endif
