/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_dio

#ifndef DEV_DIO
#define DEV_DIO

#include "dioext.h"
extern IFparm DIOpTable[ ];
extern IFparm DIOmPTable[ ];
extern char *DIOnames[ ];
extern int DIOpTSize;
extern int DIOmPTSize;
extern int DIOnSize;
extern int DIOiSize;
extern int DIOmSize;

SPICEdev DIOinfo = {
    {
        "Diode",
        "Junction Diode model",

        &DIOnSize,
        &DIOnSize,
        DIOnames,

        &DIOpTSize,
        DIOpTable,

        &DIOmPTSize,
        DIOmPTable,
	DEV_DEFAULT
    },

    DIOparam,
    DIOmParam,
    DIOload,
    DIOsetup,
    DIOunsetup,
    DIOsetup,
    DIOtemp,
    DIOtrunc,
    NULL,
    DIOacLoad,
    NULL,
    DIOdestroy,
#ifdef DELETES
    DIOmDelete,
    DIOdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    DIOgetic,
    DIOask,
    DIOmAsk,
#ifdef AN_pz
    DIOpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
#ifdef NEWCONV
    DIOconvTest,
#else /* NEWCONV */
    NULL,
#endif /* NEWCONV */

#ifdef AN_sense2
    DIOsSetup,
    DIOsLoad,
    DIOsUpdate,
    DIOsAcLoad,
    DIOsPrint,
    NULL,
#else /* AN_sense */
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
#endif /* AN_sense */
#ifdef AN_disto
    DIOdisto,
#else /* AN_disto */
    NULL,
#endif /* AN_disto */
#ifdef AN_noise
    DIOnoise,
#else /* AN_noise */
    NULL,
#endif /* AN_noise */

    &DIOiSize,
    &DIOmSize


};


#endif
#endif
