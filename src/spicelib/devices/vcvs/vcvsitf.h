/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_vcvs

#ifndef DEV_VCVS
#define DEV_VCVS

#include "vcvsext.h"
extern IFparm VCVSpTable[ ];
extern char *VCVSnames[ ];
extern int VCVSpTSize;
extern int VCVSnSize;
extern int VCVSiSize;
extern int VCVSmSize;

SPICEdev VCVSinfo = {
    {
        "VCVS",
        "Voltage controlled voltage source",

        &VCVSnSize,
        &VCVSnSize,
        VCVSnames,

        &VCVSpTSize,
        VCVSpTable,

        0,
        NULL,
	DEV_DEFAULT
    },

    VCVSparam,
    NULL,
    VCVSload,
    VCVSsetup,
    VCVSunsetup,
    VCVSsetup,
    NULL,
    NULL,
    VCVSfindBr,
    VCVSload,   /* AC and normal loads are identical */
    NULL,
    VCVSdestroy,
#ifdef DELETES
    VCVSmDelete,
    VCVSdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    VCVSask,
    NULL,
#ifdef AN_pz
    VCVSpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
#ifdef AN_sense2
    VCVSsSetup,
    VCVSsLoad,
    NULL,
    VCVSsAcLoad,
    VCVSsPrint,
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

    &VCVSiSize,
    &VCVSmSize

};


#endif
#endif
