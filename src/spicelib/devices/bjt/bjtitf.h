/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_bjt

#ifndef DEV_BJT
#define DEV_BJT

#include "bjtext.h"
extern IFparm BJTpTable[ ];
extern IFparm BJTmPTable[ ];
extern char *BJTnames[ ];
extern int BJTpTSize;
extern int BJTmPTSize;
extern int BJTnSize;
extern int BJTiSize;
extern int BJTmSize;

SPICEdev BJTinfo = {
    {   "BJT",
        "Bipolar Junction Transistor",

        &BJTnSize,
        &BJTnSize,
        BJTnames,

        &BJTpTSize,
        BJTpTable,

        &BJTmPTSize,
        BJTmPTable,
	DEV_DEFAULT
    },

    BJTparam,
    BJTmParam,
    BJTload,
    BJTsetup,
    BJTunsetup,
    BJTsetup,
    BJTtemp,
    BJTtrunc,
    NULL,
    BJTacLoad,
    NULL,
    BJTdestroy,
#ifdef DELETES
    BJTmDelete,
    BJTdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    BJTgetic,
    BJTask,
    BJTmAsk,
#ifdef AN_pz
    BJTpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    BJTconvTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */

#ifdef AN_sense2
    BJTsSetup,
    BJTsLoad,
    BJTsUpdate,
    BJTsAcLoad,
    BJTsPrint,
#else /* AN_sense2 */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense2 */
    NULL,
#ifdef AN_disto
    BJTdisto,
#else /* AN_disto */
    NULL,
#endif /* AN_disto */
#ifdef AN_noise
    BJTnoise,
#else /* AN_noise */
    NULL,
#endif /* AN_noise */

    &BJTiSize,
    &BJTmSize

};

#endif
#endif
