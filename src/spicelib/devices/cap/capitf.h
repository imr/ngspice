/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_cap

#ifndef DEV_CAP
#define DEV_CAP

#include "capext.h"
extern IFparm CAPpTable[ ];
extern IFparm CAPmPTable[ ];
extern char *CAPnames[ ];
extern int CAPpTSize;
extern int CAPmPTSize;
extern int CAPnSize;
extern int CAPiSize;
extern int CAPmSize;

SPICEdev CAPinfo = {
    {   "Capacitor",
        "Fixed capacitor",

        &CAPnSize,
        &CAPnSize,
        CAPnames,

        &CAPpTSize,
        CAPpTable,

        &CAPmPTSize,
        CAPmPTable,
	0
    },

    CAPparam,
    CAPmParam,
    CAPload,
    CAPsetup,
    NULL,
    CAPsetup,
    CAPtemp,
    CAPtrunc,
    NULL,
    CAPacLoad,
    NULL,
    CAPdestroy,
#ifdef DELETES
    CAPmDelete,
    CAPdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif  /* DELETES */
    CAPgetic,
    CAPask,
    CAPmAsk,
#ifdef AN_pz
    CAPpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
#ifdef AN_sense2
    CAPsSetup,
    CAPsLoad,
    CAPsUpdate,
    CAPsAcLoad,
    CAPsPrint,
#else /* AN_sense2 */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense2 */
    NULL,
    NULL,	/* DISTO */
    NULL,	/* NOISE */

    &CAPiSize,
    &CAPmSize


};


#endif
#endif
