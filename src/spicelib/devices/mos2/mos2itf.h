/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_mos2

#ifndef DEV_MOS2
#define DEV_MOS2

#include "mos2ext.h"
extern IFparm MOS2pTable[ ];
extern IFparm MOS2mPTable[ ];
extern char *MOS2names[ ];
extern int MOS2pTSize;
extern int MOS2mPTSize;
extern int MOS2nSize;
extern int MOS2iSize;
extern int MOS2mSize;

SPICEdev MOS2info = {
    {
        "Mos2",
        "Level 2 MOSfet model with Meyer capacitance model",

        &MOS2nSize,
        &MOS2nSize,
        MOS2names,

        &MOS2pTSize,
        MOS2pTable,

        &MOS2mPTSize,
        MOS2mPTable,
	DEV_DEFAULT
    },

    MOS2param,
    MOS2mParam,
    MOS2load,
    MOS2setup,
    MOS2unsetup,
    MOS2setup,
    MOS2temp,
    MOS2trunc,
    NULL,
    MOS2acLoad,
    NULL,
    MOS2destroy,
#ifdef DELETES
    MOS2mDelete,
    MOS2delete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    MOS2getic,
    MOS2ask,
    MOS2mAsk,
#ifdef AN_pz
    MOS2pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    MOS2convTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */

#ifdef AN_sense2
    MOS2sSetup,
    MOS2sLoad,
    MOS2sUpdate,
    MOS2sAcLoad,
    MOS2sPrint,
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
    MOS2disto,
#else	/* AN_disto */
    NULL,
#endif	/* AN_disto */
#ifdef AN_noise
    MOS2noise,
#else	/* AN_noise */
    NULL,
#endif /* AN_noise */

    &MOS2iSize,
    &MOS2mSize
};


#endif
#endif
