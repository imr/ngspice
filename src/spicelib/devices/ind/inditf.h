/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_ind

#ifndef DEV_IND
#define DEV_IND

#define MUTUAL

#include "indext.h"
extern IFparm INDpTable[ ];
extern char *INDnames[ ];
extern int INDpTSize;
extern int INDnSize;
extern int INDiSize;
extern int INDmSize;

SPICEdev INDinfo = {
    {
        "Inductor",
        "Inductors",

        &INDnSize,
        &INDnSize,
        INDnames,

        &INDpTSize,
        INDpTable,

        0,
        NULL,
	0
    },

    INDparam,
    NULL,
    INDload,
    INDsetup,
    INDunsetup,
    INDsetup,
    NULL,
    INDtrunc,
    NULL,
    INDacLoad,
    NULL,
    INDdestroy,
#ifdef DELETES
    INDmDelete,
    INDdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    INDask,
    NULL,
#ifdef AN_pz
    INDpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
#ifdef AN_sense2
    INDsSetup,
    INDsLoad,
    INDsUpdate,
    INDsAcLoad,
    INDsPrint,
    NULL,
#else /* AN_sense2 */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense */
    NULL,	/* DISTO */
    NULL,	/* NOISE */

    &INDiSize,
    &INDmSize

};

#ifdef MUTUAL
extern IFparm MUTpTable[ ];
extern int MUTpTSize;
extern int MUTiSize;
extern int MUTmSize;

SPICEdev MUTinfo = {
    {   
        "mutual",
        "Mutual inductors",
        0, /* term count */
        0, /* term count */
        NULL,

        &MUTpTSize,
        MUTpTable,

        0,
        NULL,
	0
    },

    MUTparam,
    NULL,
    NULL,/* load handled by INDload */
    MUTsetup,
    NULL,
    MUTsetup,
    NULL,
    NULL,
    NULL,
    MUTacLoad,
    NULL,
    MUTdestroy,
#ifdef DELETES
    MUTmDelete,
    MUTdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    MUTask,
    NULL,
#ifdef AN_pz
    MUTpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
#ifdef AN_sense
    MUTsSetup,
    NULL,
    NULL,
    NULL,
    MUTsPrint,
    NULL,
#else /* AN_sense */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense */
    NULL,	/* DISTO */
    NULL,	/* NOISE */

    &MUTiSize,
    &MUTmSize

};

#endif /*MUTUAL*/

#endif
#endif
