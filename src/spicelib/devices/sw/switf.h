/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_sw

#ifndef DEV_SW
#define DEV_SW

#include "swext.h"
extern IFparm SWpTable[ ];
extern IFparm SWmPTable[ ];
extern char *SWnames[ ];
extern int SWpTSize;
extern int SWmPTSize;
extern int SWnSize;
extern int SWiSize;
extern int SWmSize;

SPICEdev SWinfo = {
    {
        "Switch",
        "Ideal voltage controlled switch",

        &SWnSize,
        &SWnSize,
        SWnames,

        &SWpTSize,
        SWpTable,

        &SWmPTSize,
        SWmPTable,
	0
    },

    SWparam,
    SWmParam,
    SWload,
    SWsetup,
    NULL,
    SWsetup,
    NULL,
    NULL,
    NULL,
    SWacLoad,   
    NULL,
    SWdestroy,
#ifdef DELETES
    SWmDelete,
    SWdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    SWask,
    SWmAsk,
#ifdef AN_pz
    SWpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL, /* DISTO */
#ifdef AN_noise
    SWnoise,
#else	/* AN_noise */
    NULL,
#endif	/* AN_noise */

    &SWiSize,
    &SWmSize

};

#endif
#endif
