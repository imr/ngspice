/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_cccs

#ifndef DEV_CCCS
#define DEV_CCCS

#include "cccsext.h"
extern IFparm CCCSpTable[ ];
extern char *CCCSnames[ ];
extern int CCCSpTSize;
extern int CCCSnSize;
extern int CCCSiSize;
extern int CCCSmSize;

SPICEdev CCCSinfo = {
    {   "CCCS",
        "Current controlled current source",

        &CCCSnSize,
        &CCCSnSize,
        CCCSnames,

        &CCCSpTSize,
        CCCSpTable,

        0,
        NULL,
	DEV_DEFAULT
    },

    CCCSparam,
    NULL,
    CCCSload,
    CCCSsetup,
    NULL,
    CCCSsetup,
    NULL,
    NULL,
    NULL,
    CCCSload,
    NULL,
    CCCSdestroy,
#ifdef DELETES
    CCCSmDelete,
    CCCSdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    CCCSask,
    NULL,
#ifdef AN_pz
    CCCSpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
#ifdef AN_sense2
    CCCSsSetup,
    CCCSsLoad,
    NULL,
    CCCSsAcLoad,
    CCCSsPrint,
    NULL,
#else /* AN_sense2 */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense2 */
    NULL,	/* DISTO */
    NULL,	/* NOISE */

    &CCCSiSize,
    &CCCSmSize

};

#endif
#endif
