/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1991 JianHui Huang and Min-Chie Jeng.
File: bsim3v2itf.h
**********/
#ifdef DEV_bsim3v2

#ifndef DEV_BSIM3V2
#define DEV_BSIM3V2

#include "bsim3v2ext.h"

extern IFparm BSIM3V2pTable[ ];
extern IFparm BSIM3V2mPTable[ ];
extern char *BSIM3V2names[ ];
extern int BSIM3V2pTSize;
extern int BSIM3V2mPTSize;
extern int BSIM3V2nSize;
extern int BSIM3V2iSize;
extern int BSIM3V2mSize;

SPICEdev BSIM3V2info = {
    {   "BSIM3V2",
        "Berkeley Short Channel IGFET Model Version-3 (3v3.2)",

        &BSIM3V2nSize,
        &BSIM3V2nSize,
        BSIM3V2names,

        &BSIM3V2pTSize,
        BSIM3V2pTable,

        &BSIM3V2mPTSize,
        BSIM3V2mPTable,
		DEV_DEFAULT
    },

    BSIM3V2param,
    BSIM3V2mParam,
    BSIM3V2load,
    BSIM3V2setup,
	BSIM3V2unsetup,
    BSIM3V2setup,
    BSIM3V2temp,
    BSIM3V2trunc,
    NULL,
    BSIM3V2acLoad,
    NULL,
    BSIM3V2destroy,
#ifdef DELETES
    BSIM3V2mDelete,
    BSIM3V2delete, 
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    BSIM3V2getic,
    BSIM3V2ask,
    BSIM3V2mAsk,
#ifdef AN_pz
    BSIM3V2pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    BSIM3V2convTest,
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
    BSIM3V2noise,
#else   /* AN_noise */
    NULL,
#endif  /* AN_noise */

    &BSIM3V2iSize,
    &BSIM3V2mSize

};

#endif
#endif

