/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_urc

#ifndef DEV_URC
#define DEV_URC

#include "urcext.h"
extern IFparm URCpTable[ ];
extern IFparm URCmPTable[ ];
extern char *URCnames[ ];
extern int URCpTSize;
extern int URCmPTSize;
extern int URCnSize;
extern int URCiSize;
extern int URCmSize;

SPICEdev URCinfo = {
    {
        "URC",      /* MUST precede both resistors and capacitors */
        "Uniform R.C. line",

        &URCnSize,
        &URCnSize,
        URCnames,

        &URCpTSize,
        URCpTable,

        &URCmPTSize,
        URCmPTable,
	0
    },

    URCparam,
    URCmParam,
    NULL,
    URCsetup,
    URCunsetup,
    URCsetup,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    URCdestroy,
#ifdef DELETES
    URCmDelete,
    URCdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    URCask,
    URCmAsk,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,	/* DISTO */
    NULL,	/* NOISE */

    &URCiSize,
    &URCmSize

};


#endif
#endif
