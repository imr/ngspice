#include <config.h>

#include <devdefs.h>

#include "cccsitf.h"
#include "cccsext.h"
#include "cccsinit.h"


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

	DEV_DEFAULT
    },

    DEVparam      : CCCSparam,
    DEVmodParam   : NULL,
    DEVload       : CCCSload,
    DEVsetup      : CCCSsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : CCCSsetup,
    DEVtemperature: NULL,
    DEVtrunc      : NULL,
    DEVfindBranch : NULL,
    DEVacLoad     : CCCSload,
    DEVaccept     : NULL,
    DEVdestroy    : CCCSdestroy,
    DEVmodDelete  : CCCSmDelete,
    DEVdelete     : CCCSdelete,
    DEVsetic      : NULL,
    DEVask        : CCCSask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : CCCSpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : CCCSsSetup,
    DEVsenLoad    : CCCSsLoad,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : CCCSsAcLoad,
    DEVsenPrint   : CCCSsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : NULL,	/* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif    
    DEVinstSize   : &CCCSiSize,
    DEVmodSize    : &CCCSmSize

};


SPICEdev *
get_cccs_info(void)
{
    return &CCCSinfo;
}
