/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
File: bsim3itf.h
**********/
#ifdef DEV_bsim3

#ifndef DEV_BSIM3
#define DEV_BSIM3

#include "bsim3ext.h"

extern IFparm BSIM3pTable[ ];
extern IFparm BSIM3mPTable[ ];
extern char *BSIM3names[ ];
extern int BSIM3pTSize;
extern int BSIM3mPTSize;
extern int BSIM3nSize;
extern int BSIM3iSize;
extern int BSIM3mSize;

SPICEdev BSIM3info = {
    {   "BSIM3",
        "Berkeley Short Channel IGFET Model Version-3",

        &BSIM3nSize,
        &BSIM3nSize,
        BSIM3names,

        &BSIM3pTSize,
        BSIM3pTable,

        &BSIM3mPTSize,
        BSIM3mPTable,
        DEV_DEFAULT
    },

    BSIM3param,
    BSIM3mParam,
    BSIM3load,
    BSIM3setup,
    BSIM3unsetup,
    BSIM3setup,
    BSIM3temp,
    BSIM3trunc,
    NULL,
    BSIM3acLoad,
    NULL,
    BSIM3destroy,
#ifdef DELETES
    BSIM3mDelete,
    BSIM3delete, 
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    BSIM3getic,
    BSIM3ask,
    BSIM3mAsk,
#ifdef AN_pz
    BSIM3pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    BSIM3convTest,
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
    BSIM3noise,
#else   /* AN_noise */
    NULL,
#endif  /* AN_noise */

    &BSIM3iSize,
    &BSIM3mSize

};

#endif
#endif
