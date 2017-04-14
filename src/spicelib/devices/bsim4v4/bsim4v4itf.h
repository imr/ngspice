/**********
Copyright 2004 Regents of the University of California.  All rights reserved.
Author: 2000 Weidong Liu.
Author: 2001- Xuemei Xi
File: bsim4v4itf.h
**********/
#ifdef DEV_bsim4v4

#ifndef DEV_BSIM4v4
#define DEV_BSIM4v4

#include "bsim4v4ext.h"

extern IFparm BSIM4v4pTable[ ];
extern IFparm BSIM4v4mPTable[ ];
extern char *BSIM4v4names[ ];
extern int BSIM4v4pTSize;
extern int BSIM4v4mPTSize;
extern int BSIM4v4nSize;
extern int BSIM4v4iSize;
extern int BSIM4v4mSize;

SPICEdev B4info = {
    {   "BSIM4v4",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4v4nSize,
        &BSIM4v4nSize,
        BSIM4v4names,

        &BSIM4v4pTSize,
        BSIM4v4pTable,

        &BSIM4v4mPTSize,
        BSIM4v4mPTable,
        DEV_DEFAULT
    },
    BSIM4v4param,
    BSIM4v4mParam,
    BSIM4v4load,
    BSIM4v4setup,
    BSIM4v4unsetup,
    BSIM4v4setup,
    BSIM4v4temp,
    BSIM4v4trunc,
    NULL,
    BSIM4v4acLoad,
    NULL,
    BSIM4v4destroy,
#ifdef DELETES
    BSIM4v4mDelete,
    BSIM4v4delete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    BSIM4v4getic,
    BSIM4v4ask,
    BSIM4v4mAsk,
#ifdef AN_pz
    BSIM4v4pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    BSIM4v4convTest,
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
    BSIM4v4noise,
#else   /* AN_noise */
    NULL,
#endif  /* AN_noise */

    &BSIM4v4iSize,
    &BSIM4v4mSize
};

#endif
#endif
