/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_vsrc

#ifndef DEV_VSRC
#define DEV_VSRC

#include "vsrcext.h"
extern IFparm VSRCpTable[ ];
extern char *VSRCnames[ ];
extern int VSRCpTSize;
extern int VSRCnSize;
extern int VSRCiSize;
extern int VSRCmSize;

SPICEdev VSRCinfo = {
    {
        "Vsource", 
        "Independent voltage source",

        &VSRCnSize,
        &VSRCnSize,
        VSRCnames,

        &VSRCpTSize,
        VSRCpTable,

        0,
        NULL,
	DEV_DEFAULT
    },

    VSRCparam,
    NULL,
    VSRCload,
    VSRCsetup,
    VSRCunsetup,
#ifdef AN_pz
    VSRCpzSetup,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    VSRCtemp,
    NULL,
    VSRCfindBr,
    VSRCacLoad,
    VSRCaccept,
    VSRCdestroy,
#ifdef DELETES
    VSRCmDelete,
    VSRCdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    VSRCask,
    NULL,
    VSRCpzLoad,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL, /* DISTO */
    NULL, /* NOISE */

    &VSRCiSize,
    &VSRCmSize
};


#endif
#endif
