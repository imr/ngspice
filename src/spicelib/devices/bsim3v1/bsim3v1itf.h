/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
File: bsim3v1itf.h
**********/
#ifdef DEV_bsim3v1

#ifndef DEV_BSIM3V1
#define DEV_BSIM3V1

#include "bsim3v1ext.h"

extern IFparm BSIM3V1pTable[ ];
extern IFparm BSIM3V1mPTable[ ];
extern char *BSIM3V1names[ ];
extern int BSIM3V1pTSize;
extern int BSIM3V1mPTSize;
extern int BSIM3V1nSize;
extern int BSIM3V1iSize;
extern int BSIM3V1mSize;

SPICEdev BSIM3V1info = {
    {   "BSIM3V1",
        "Berkeley Short Channel IGFET Model Version-3 (3v3.1)",

        &BSIM3V1nSize,
        &BSIM3V1nSize,
        BSIM3V1names,

        &BSIM3V1pTSize,
        BSIM3V1pTable,

        &BSIM3V1mPTSize,
        BSIM3V1mPTable,
	DEV_DEFAULT,

    },

    BSIM3V1param,
    BSIM3V1mParam,
    BSIM3V1load,
    BSIM3V1setup,
    NULL,
    BSIM3V1setup,
    BSIM3V1temp,
    BSIM3V1trunc,
    NULL,
    BSIM3V1acLoad,
    NULL,
    BSIM3V1destroy,
#ifdef DELETES
    BSIM3V1mDelete,
    BSIM3V1delete, 
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    BSIM3V1getic,
    BSIM3V1ask,
    BSIM3V1mAsk,
#ifdef AN_pz
    BSIM3V1pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    BSIM3V1convTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,

#ifdef AN_noise
    BSIM3V1noise,
#else   /* AN_noise */
    NULL,
#endif  /* AN_noise */

    &BSIM3V1iSize,
    &BSIM3V1mSize

};

#endif
#endif

