#include <config.h>

#include <devdefs.h>

#include "capitf.h"
#include "capext.h"
#include "capinit.h"


SPICEdev CAPinfo = {
    {   "Capacitor",
        "Fixed capacitor",

        &CAPnSize,
        &CAPnSize,
        CAPnames,

        &CAPpTSize,
        CAPpTable,

        &CAPmPTSize,
        CAPmPTable,

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

    DEVparam      : CAPparam,
    DEVmodParam   : CAPmParam,
    DEVload       : CAPload,
    DEVsetup      : CAPsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : CAPsetup,
    DEVtemperature: CAPtemp,
    DEVtrunc      : CAPtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : CAPacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : CAPdestroy,
    DEVmodDelete  : CAPmDelete,
    DEVdelete     : CAPdelete,
    DEVsetic      : CAPgetic,
    DEVask        : CAPask,
    DEVmodAsk     : CAPmAsk,
    DEVpzLoad     : CAPpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : CAPsSetup,
    DEVsenLoad    : CAPsLoad,
    DEVsenUpdate  : CAPsUpdate,
    DEVsenAcLoad  : CAPsAcLoad,
    DEVsenPrint   : CAPsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : NULL,	/* NOISE */
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif    
    DEVinstSize   : &CAPiSize,
    DEVmodSize    : &CAPmSize
};


SPICEdev *
get_cap_info(void)
{
    return &CAPinfo;
}
