/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_mos3

#ifndef DEV_MOS3
#define DEV_MOS3

#include "mos3ext.h"
extern IFparm MOS3pTable[ ];
extern IFparm MOS3mPTable[ ];
extern char *MOS3names[ ];
extern int MOS3pTSize;
extern int MOS3mPTSize;
extern int MOS3nSize;
extern int MOS3iSize;
extern int MOS3mSize;

SPICEdev MOS3info = {
    {
        "Mos3",
        "Level 3 MOSfet model with Meyer capacitance model",

        &MOS3nSize,
        &MOS3nSize,
        MOS3names,

        &MOS3pTSize,
        MOS3pTable,

        &MOS3mPTSize,
        MOS3mPTable,
	DEV_DEFAULT
    },

    MOS3param,
    MOS3mParam,
    MOS3load,
    MOS3setup,
    MOS3unsetup,
    MOS3setup,
    MOS3temp,
    MOS3trunc,
    NULL,
    MOS3acLoad,
    NULL,
    MOS3destroy,
#ifdef DELETES
    MOS3mDelete,
    MOS3delete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    MOS3getic,
    MOS3ask,
    MOS3mAsk,
#ifdef AN_pz
    MOS3pzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    MOS3convTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */

#ifdef AN_sense2
    MOS3sSetup,
    MOS3sLoad,
    MOS3sUpdate,
    MOS3sAcLoad,
    MOS3sPrint,
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
    MOS3disto,
#else	/* AN_disto */
    NULL,
#endif	/* AN_disto */
#ifdef AN_noise
    MOS3noise,
#else	/* AN_noise */
    NULL,
#endif	/* AN_noise */

    &MOS3iSize,
    &MOS3mSize

};

#endif
#endif
