#include <config.h>

#include <devdefs.h>

#include "ccvsitf.h"
#include "ccvsext.h"
#include "ccvsinit.h"


SPICEdev CCVSinfo = {
    {
        "CCVS",
        "Linear current controlled current source",

        &CCVSnSize,
        &CCVSnSize,
        CCVSnames,

        &CCVSpTSize,
        CCVSpTable,

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

    DEVparam      : CCVSparam,
    DEVmodParam   : NULL,
    DEVload       : CCVSload,
    DEVsetup      : CCVSsetup,
    DEVunsetup    : CCVSunsetup,
    DEVpzSetup    : CCVSsetup,
    DEVtemperature: NULL,
    DEVtrunc      : NULL,
    DEVfindBranch : CCVSfindBr,
    DEVacLoad     : CCVSload,   /* ac and normal load functions identical */
    DEVaccept     : NULL,
    DEVdestroy    : CCVSdestroy,
    DEVmodDelete  : CCVSmDelete,
    DEVdelete     : CCVSdelete,
    DEVsetic      : NULL,
    DEVask        : CCVSask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : CCVSpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : CCVSsSetup,
    DEVsenLoad    : CCVSsLoad,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : CCVSsAcLoad,
    DEVsenPrint   : CCVSsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : NULL,	/* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif
    DEVinstSize   : &CCVSiSize,
    DEVmodSize    : &CCVSmSize

};


SPICEdev *
get_ccvs_info(void)
{
    return &CCVSinfo;
}
