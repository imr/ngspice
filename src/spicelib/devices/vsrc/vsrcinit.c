#include <config.h>

#include <devdefs.h>

#include "vsrcitf.h"
#include "vsrcext.h"
#include "vsrcinit.h"


SPICEdev VSRCinfo = {
    {
        "Vsource", 
        "Independent voltage source",

        &VSRCnSize,
        &VSRCnSize,
        VSRCnames,

        &VSRCpTSize,
        VSRCpTable,

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

    DEVparam      : VSRCparam,
    DEVmodParam   : NULL,
    DEVload       : VSRCload,
    DEVsetup      : VSRCsetup,
    DEVunsetup    : VSRCunsetup,
    DEVpzSetup    : VSRCpzSetup,
    DEVtemperature: VSRCtemp,
    DEVtrunc      : NULL,
    DEVfindBranch : VSRCfindBr,
    DEVacLoad     : VSRCacLoad,
    DEVaccept     : VSRCaccept,
    DEVdestroy    : VSRCdestroy,
    DEVmodDelete  : VSRCmDelete,
    DEVdelete     : VSRCdelete,
    DEVsetic      : NULL,
    DEVask        : VSRCask,
    DEVmodAsk     : NULL,
    DEVpzLoad     : VSRCpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL, /* DISTO */
    DEVnoise      : NULL, /* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &VSRCiSize,
    DEVmodSize    : &VSRCmSize
};


SPICEdev *
get_vsrc_info(void)
{
    return &VSRCinfo;
}
