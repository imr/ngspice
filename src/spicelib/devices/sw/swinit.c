/* Modified: Alansfixes */
#include <config.h>

#include <devdefs.h>

#include "switf.h"
#include "swext.h"
#include "swinit.h"


SPICEdev SWinfo = {
    {
        "Switch",
        "Ideal voltage controlled switch",

        &SWnSize,
        &SWnSize,
        SWnames,

        &SWpTSize,
        SWpTable,

        &SWmPTSize,
        SWmPTable,

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

    DEVparam      : SWparam,
    DEVmodParam   : SWmParam,
    DEVload       : SWload,
    DEVsetup      : SWsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : SWsetup,
    DEVtemperature: NULL,
    DEVtrunc      : SWtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : SWacLoad,   
    DEVaccept     : NULL,
    DEVdestroy    : SWdestroy,
    DEVmodDelete  : SWmDelete,
    DEVdelete     : SWdelete,
    DEVsetic      : NULL,
    DEVask        : SWask,
    DEVmodAsk     : SWmAsk,
    DEVpzLoad     : SWpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL, /* DISTO */
    DEVnoise      : SWnoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif /* CIDER */                        
    DEVinstSize   : &SWiSize,
    DEVmodSize    : &SWmSize

};


SPICEdev *
get_sw_info(void)
{
    return &SWinfo;
}
