/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_bsim2

#ifndef DEV_BSIM2
#define DEV_BSIM2

#include "bsim2ext.h"

extern IFparm B2pTable[ ];
extern IFparm B2mPTable[ ];
extern char *B2names[ ];
extern int B2pTSize;
extern int B2mPTSize;
extern int B2nSize;
extern int B2iSize;
extern int B2mSize;

SPICEdev B2info = {
    {   "BSIM2",
        "Berkeley Short Channel IGFET Model",

        &B2nSize,
        &B2nSize,
        B2names,

        &B2pTSize,
        B2pTable,

        &B2mPTSize,
        B2mPTable,
	DEV_DEFAULT
    },

    B2param,
    B2mParam,
    B2load,
    B2setup,
    B2unsetup,
    B2setup,
    B2temp,
    B2trunc,
    NULL,
    B2acLoad,
    NULL,
    B2destroy,
#ifdef DELETES
    B2mDelete,
    B2delete, 
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    B2getic,
    B2ask,
    B2mAsk,
#ifdef AN_pz
    B2pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    B2convTest,
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
    NULL,

    &B2iSize,
    &B2mSize

};

#endif
#endif
