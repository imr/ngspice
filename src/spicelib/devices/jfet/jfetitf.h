/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_jfet

#ifndef DEV_JFET
#define DEV_JFET

#include "jfetext.h"
extern IFparm JFETpTable[ ];
extern IFparm JFETmPTable[ ];
extern char *JFETnames[ ];
extern int JFETpTSize;
extern int JFETmPTSize;
extern int JFETnSize;
extern int JFETiSize;
extern int JFETmSize;

SPICEdev JFETinfo = {
    {
        "JFET",
        "Junction Field effect transistor",

        &JFETnSize,
        &JFETnSize,
        JFETnames,

        &JFETpTSize,
        JFETpTable,

        &JFETmPTSize,
        JFETmPTable,
	DEV_DEFAULT
    },

    JFETparam,
    JFETmParam,
    JFETload,
    JFETsetup,
    JFETunsetup,
    JFETsetup,
    JFETtemp,
    JFETtrunc,
    NULL,
    JFETacLoad,
    NULL,
    JFETdestroy,
#ifdef DELETES
    JFETmDelete,
    JFETdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    JFETgetic,
    JFETask,
    JFETmAsk,
#ifdef AN_pz
    JFETpzLoad,
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
    JFETdisto,
#else	/* AN_disto */
    NULL,
#endif	/* AN_disto */
#ifdef AN_noise
    JFETnoise,
#else	/* AN_noise */
    NULL,
#endif	/* AN_noise */

    &JFETiSize,
    &JFETmSize

};


#endif
#endif
