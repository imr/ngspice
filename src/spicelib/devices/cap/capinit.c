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
                    
    DEVinstSize   : &CAPiSize,
    DEVmodSize    : &CAPmSize
};


SPICEdev *
get_cap_info(void)
{
    return &CAPinfo;
}
