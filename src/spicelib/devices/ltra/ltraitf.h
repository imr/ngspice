/**********
Copyright 1990 Regents of the University of California.  All rights
reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/
#ifdef DEV_ltra

#ifndef DEV_LTRA
#define DEV_LTRA

#include "ltraext.h"
extern IFparm LTRApTable[ ];
extern IFparm LTRAmPTable[ ];
extern char *LTRAnames[ ];
extern int LTRApTSize;
extern int LTRAmPTSize;
extern int LTRAnSize;
extern int LTRAiSize;
extern int LTRAmSize;

SPICEdev LTRAinfo = {
    { "LTRA",
        "Lossy transmission line",

        &LTRAnSize,
	&LTRAnSize,
	LTRAnames,

        &LTRApTSize,
        LTRApTable,

        &LTRAmPTSize,
        LTRAmPTable,
	0
    },

    LTRAparam,
    LTRAmParam,
    LTRAload,
    LTRAsetup,
    LTRAunsetup,
    LTRAsetup,
    LTRAtemp,
    LTRAtrunc,
    NULL,
    LTRAacLoad /*LTRAacLoad*/,
    LTRAaccept,
    LTRAdestroy,
#ifdef DELETES
    LTRAmDelete,
    LTRAdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL, 	/* getic */
    LTRAask,
    LTRAmAsk, 	/* */
    NULL,	/* pzLoad */
    NULL,	/* convTest */
    NULL,	/* sSetup */
    NULL,	/* sLoad */
    NULL,	/* sUpdate */
    NULL,	/* sAcLoad */
    NULL,	/* sPrint */
    NULL,	/* */
    NULL,	/* disto */
    NULL,	/* noise */

    &LTRAiSize,
    &LTRAmSize

};

#endif

#endif
