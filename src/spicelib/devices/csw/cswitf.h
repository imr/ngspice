/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_csw

#ifndef DEV_CSW
#define DEV_CSW

#include "cswext.h"
extern IFparm CSWpTable[ ];
extern IFparm CSWmPTable[ ];
extern char *CSWnames[ ];
extern int CSWpTSize;
extern int CSWmPTSize;
extern int CSWnSize;
extern int CSWiSize;
extern int CSWmSize;

SPICEdev CSWinfo = {
    {   "CSwitch",
        "Current controlled ideal switch",

        &CSWnSize,
        &CSWnSize,
        CSWnames,

        &CSWpTSize,
        CSWpTable,

        &CSWmPTSize,
        CSWmPTable,
	0
    },

    CSWparam,
    CSWmParam,
    CSWload,
    CSWsetup,
    NULL,
    CSWsetup,
    NULL,
    NULL,
    NULL,
    CSWacLoad,
    NULL,
    CSWdestroy,
#ifdef DELETES
    CSWmDelete,
    CSWdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    CSWask,
    CSWmAsk,
#ifdef AN_pz
    CSWpzLoad,
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
    NULL,	/* DISTO */
#ifdef AN_noise
    CSWnoise,
#else	/* AN_noise */
    NULL,
#endif	/* AN_noise */

    &CSWiSize,
    &CSWmSize

};


#endif
#endif
