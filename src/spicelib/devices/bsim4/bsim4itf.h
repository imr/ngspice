/**********
Copyright 2000 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
File: bsim4itf.h
**********/
#ifdef DEV_bsim4

#ifndef DEV_BSIM4
#define DEV_BSIM4

#include "bsim4ext.h"

extern IFparm BSIM4pTable[ ];
extern IFparm BSIM4mPTable[ ];
extern char *BSIM4names[ ];
extern int BSIM4pTSize;
extern int BSIM4mPTSize;
extern int BSIM4nSize;
extern int BSIM4iSize;
extern int BSIM4mSize;

SPICEdev B4info = {
    {   "BSIM4",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4nSize,
        &BSIM4nSize,
        BSIM4names,

        &BSIM4pTSize,
        BSIM4pTable,

        &BSIM4mPTSize,
        BSIM4mPTable,
        DEV_DEFAULT
    },
    BSIM4param,
    BSIM4mParam,
    BSIM4load,
    BSIM4setup,
    BSIM4unsetup,
    BSIM4setup,
    BSIM4temp,
    BSIM4trunc,
    NULL,
    BSIM4acLoad,
    NULL,
    BSIM4destroy,
#ifdef DELETES
    BSIM4mDelete,
    BSIM4delete, 
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    BSIM4getic,
    BSIM4ask,
    BSIM4mAsk,
#ifdef AN_pz
    BSIM4pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    BSIM4convTest,
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
    BSIM4noise,
#else   /* AN_noise */
    NULL,
#endif  /* AN_noise */

    &BSIM4iSize,
    &BSIM4mSize
};

#endif
#endif
