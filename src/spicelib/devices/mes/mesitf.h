/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_mes

#ifndef DEV_MES
#define DEV_MES

#include "mesext.h"
extern IFparm MESpTable[ ];
extern IFparm MESmPTable[ ];
extern char *MESnames[ ];
extern int MESpTSize;
extern int MESmPTSize;
extern int MESnSize;
extern int MESiSize;
extern int MESmSize;

SPICEdev MESinfo = {
    {
        "MES",
        "GaAs MESFET model",

        &MESnSize,
        &MESnSize,
        MESnames,

        &MESpTSize,
        MESpTable,

        &MESmPTSize,
        MESmPTable,
	DEV_DEFAULT
    },

    MESparam,
    MESmParam,
    MESload,
    MESsetup,
    MESunsetup,
    MESsetup,
    MEStemp,
    MEStrunc,
    NULL,
    MESacLoad,
    NULL,
    MESdestroy,
#ifdef DELETES
    MESmDelete,
    MESdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    MESgetic,
    MESask,
    MESmAsk,
#ifdef AN_pz
    MESpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#ifdef AN_disto
    MESdisto,
#else	/* AN_disto */
    NULL,
#endif	/* AN_disto */
#ifdef AN_noise
    MESnoise,
#else	/* AN_noise */
    NULL,
#endif	/* AN_noise */

    &MESiSize,
    &MESmSize

};

#endif
#endif
