/**********
Based on jfetitf.h
Copyright 1990 Regents of the University of California.  All rights reserved.

Modified to add PS model and new parameter definitions ( Anthony E. Parker )
   Copyright 1994  Macquarie University, Sydney Australia.
   pz and disto not supported
**********/
#ifdef DEV_jfet2

#ifndef DEV_JFET2
#define DEV_JFET2

#include "jfet2ext.h"
extern IFparm JFET2pTable[ ];
extern IFparm JFET2mPTable[ ];
extern char *JFET2names[ ];
extern int JFET2pTSize;
extern int JFET2mPTSize;
extern int JFET2nSize;
extern int JFET2iSize;
extern int JFET2mSize;

SPICEdev JFET2info = {
    {
        "JFET2",
        "Short channel field effect transistor",

        &JFET2nSize,
        &JFET2nSize,
        JFET2names,

        &JFET2pTSize,
        JFET2pTable,

        &JFET2mPTSize,
        JFET2mPTable,
	DEV_DEFAULT
    },

    JFET2param,
    JFET2mParam,
    JFET2load,
    JFET2setup,
    JFET2unsetup,
    JFET2setup,
    JFET2temp,
    JFET2trunc,
    NULL,
    JFET2acLoad,
    NULL,
    JFET2destroy,
#ifdef DELETES
    JFET2mDelete,
    JFET2delete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    JFET2getic,
    JFET2ask,
    JFET2mAsk,
    NULL, /* AN_pz */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL, /* AN_disto */
#ifdef AN_noise
    JFET2noise,
#else	/* AN_noise */
    NULL,
#endif	/* AN_noise */

    &JFET2iSize,
    &JFET2mSize

};


#endif
#endif
