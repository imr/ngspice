/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_mos1

#ifndef DEV_MOS1
#define DEV_MOS1

#include "mos1ext.h"
extern IFparm MOS1pTable[ ];
extern IFparm MOS1mPTable[ ];
extern char *MOS1names[ ];
extern int MOS1pTSize;
extern int MOS1mPTSize;
extern int MOS1nSize;
extern int MOS1iSize;
extern int MOS1mSize;

SPICEdev MOS1info = {
    {
        "Mos1",
        "Level 1 MOSfet model with Meyer capacitance model",

        &MOS1nSize,
        &MOS1nSize,
        MOS1names,

        &MOS1pTSize,
        MOS1pTable,

        &MOS1mPTSize,
        MOS1mPTable,
	DEV_DEFAULT
    },

    MOS1param,
    MOS1mParam,
    MOS1load,
    MOS1setup,
    MOS1unsetup,
    MOS1setup,
    MOS1temp,
    MOS1trunc,
    NULL,
    MOS1acLoad,
    NULL,
    MOS1destroy,
#ifdef DELETES
    MOS1mDelete,
    MOS1delete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    MOS1getic,
    MOS1ask,
    MOS1mAsk,
#ifdef AN_pz
    MOS1pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    MOS1convTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */

#ifdef AN_sense2
    MOS1sSetup,
    MOS1sLoad,
    MOS1sUpdate,
    MOS1sAcLoad,
    MOS1sPrint,
    NULL,
#else /* AN_sense2 */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense2 */
#ifdef AN_disto
    MOS1disto,
#else /* AN_disto */
    NULL,
#endif /* AN_disto */
#ifdef AN_noise
    MOS1noise,
#else /* AN_noise */
    NULL,
#endif /* AN_noise */

    &MOS1iSize,
    &MOS1mSize
};


#endif
#endif
