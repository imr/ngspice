#include <config.h>

#include <devdefs.h>

#include "traitf.h"
#include "traext.h"
#include "trainit.h"


SPICEdev TRAinfo = {
    {
        "Tranline",
        "Lossless transmission line",

        &TRAnSize,
        &TRAnSize,
        TRAnames,

        &TRApTSize,
        TRApTable,

        0,
        NULL,

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

    DEVparam      : TRAparam,
    DEVmodParam   : NULL,
    DEVload       : TRAload,
    DEVsetup      : TRAsetup,
    DEVunsetup    : TRAunsetup,
    DEVpzSetup    : TRAsetup,
    DEVtemperature: TRAtemp,
    DEVtrunc      : TRAtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : TRAacLoad,
    DEVaccept     : TRAaccept,
    DEVdestroy    : TRAdestroy,
    DEVmodDelete  : TRAmDelete,
    DEVdelete     : TRAdelete,
    DEVsetic      : NULL,
    DEVask        : TRAask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : NULL,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : NULL,	/* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &TRAiSize,
    DEVmodSize    : &TRAmSize

};


SPICEdev *
get_tra_info(void)
{
    return &TRAinfo;
}
