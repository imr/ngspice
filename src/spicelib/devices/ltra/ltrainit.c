#include <config.h>

#include <devdefs.h>

#include "ltraitf.h"
#include "ltraext.h"
#include "ltrainit.h"


SPICEdev LTRAinfo = {
    {
	"LTRA",
        "Lossy transmission line",

        &LTRAnSize,
	&LTRAnSize,
	LTRAnames,

        &LTRApTSize,
        LTRApTable,

        &LTRAmPTSize,
        LTRAmPTable,

#ifdef XSPICE
/*----  Fixed by SDB 5.2.2003 to enable XSPICE/tclspice integration  -----*/
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */
/*---------------------------  End of SDB fix   -------------------------*/
#endif

	0
    },

    DEVparam      : LTRAparam,
    DEVmodParam   : LTRAmParam,
    DEVload       : LTRAload,
    DEVsetup      : LTRAsetup,
    DEVunsetup    : LTRAunsetup,
    DEVpzSetup    : LTRAsetup,
    DEVtemperature: LTRAtemp,
    DEVtrunc      : LTRAtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : LTRAacLoad /*LTRAacLoad*/,
    DEVaccept     : LTRAaccept,
    DEVdestroy    : LTRAdestroy,
    DEVmodDelete  : LTRAmDelete,
    DEVdelete     : LTRAdelete,
    DEVsetic      : NULL, 	/* getic */
    DEVask        : LTRAask,
    DEVmodAsk     : LTRAmAsk, 	/* */
    DEVpzLoad     : NULL,	/* pzLoad */
    DEVconvTest   : NULL,	/* convTest */
    DEVsenSetup   : NULL,	/* sSetup */
    DEVsenLoad    : NULL,	/* sLoad */
    DEVsenUpdate  : NULL,	/* sUpdate */
    DEVsenAcLoad  : NULL,	/* sAcLoad */
    DEVsenPrint   : NULL,	/* sPrint */
    DEVsenTrunc   : NULL,	/* */
    DEVdisto      : NULL,	/* disto */
    DEVnoise      : NULL,	/* noise */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &LTRAiSize,
    DEVmodSize    : &LTRAmSize

};


SPICEdev *
get_ltra_info(void)
{
    return &LTRAinfo;
}
