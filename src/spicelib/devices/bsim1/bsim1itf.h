/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_bsim1

#ifndef DEV_BSIM1
#define DEV_BSIM1

#include "bsim1ext.h"
extern IFparm B1pTable[ ];
extern IFparm B1mPTable[ ];
extern char *B1names[ ];
extern int B1pTSize;
extern int B1mPTSize;
extern int B1nSize;
extern int B1iSize;
extern int B1mSize;

SPICEdev B1info = {
    {   "BSIM1",
        "Berkeley Short Channel IGFET Model",

        &B1nSize,
        &B1nSize,
        B1names,

        &B1pTSize,
        B1pTable,

        &B1mPTSize,
        B1mPTable,
	DEV_DEFAULT
    },

    B1param,
    B1mParam,
    B1load,
    B1setup,
    B1unsetup,
    B1setup,
    B1temp,
    B1trunc,
    NULL,
    B1acLoad,
    NULL,
    B1destroy,
#ifdef DELETES
    B1mDelete,
    B1delete, 
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    B1getic,
    B1ask,
    B1mAsk,
#ifdef AN_pz
    B1pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    B1convTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#ifdef AN_disto
    B1disto,
#else
    NULL,
#endif
    NULL,	/* NOISE */

    &B1iSize,
    &B1mSize

};

#endif
#endif
