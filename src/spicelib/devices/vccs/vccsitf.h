/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
#ifdef DEV_vccs

#ifndef DEV_VCCS
#define DEV_VCCS

#include "vccsext.h"
extern IFparm VCCSpTable[ ];
extern char *VCCSnames[ ];
extern int VCCSpTSize;
extern int VCCSnSize;
extern int VCCSiSize;
extern int VCCSmSize;

SPICEdev VCCSinfo = {
    {
        "VCCS",
        "Voltage controlled current source",

        &VCCSnSize,
        &VCCSnSize,
        VCCSnames,

        &VCCSpTSize,
        VCCSpTable,

        0,
        NULL,
	DEV_DEFAULT
    },

    VCCSparam,
    NULL,
    VCCSload,
    VCCSsetup,
    NULL,
    VCCSsetup,
    NULL,
    NULL,
    NULL,
    VCCSload,   /* ac and normal loads are identical */
    NULL,
    VCCSdestroy,
#ifdef DELETES
    VCCSmDelete,
    VCCSdelete,
#else /* DELETES */
    NULL,
    NULL,
#endif /* DELETES */
    NULL,
    VCCSask,
    NULL,
#ifdef AN_pz
    VCCSpzLoad,
#else /* AN_pz */
    NULL,
#endif /* AN_pz */
    NULL,
#ifdef AN_sense2
    VCCSsSetup,
    VCCSsLoad,
    NULL,
    VCCSsAcLoad,
    VCCSsPrint,
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

    &VCCSiSize,
    &VCCSmSize


};


#endif
#endif
