/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_tra

#ifndef DEV_TRA
#define DEV_TRA

#include "traext.h"
extern IFparm TRApTable[ ];
extern char *TRAnames[ ];
extern int TRApTSize;
extern int TRAnSize;
extern int TRAiSize;
extern int TRAmSize;

SPICEdev TRAinfo = {
    {
        "Tranline",
        "Lossless transmission line",

        &TRAnSize,
        &TRAnSize,
        TRAnames,

        &TRApTSize,
        TRApTable,

        0,
        NULL,
	0
    },

    TRAparam,
    NULL,
    TRAload,
    TRAsetup,
    TRAunsetup,
    TRAsetup,
    TRAtemp,
    TRAtrunc,
    NULL,
    TRAacLoad,
    TRAaccept,
    TRAdestroy,
#ifdef DELETES
    TRAmDelete,
    TRAdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    TRAask,
    NULL,
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

    &TRAiSize,
    &TRAmSize

};


#endif
#endif
