#include <config.h>

#include <devdefs.h>

#include "isrcitf.h"
#include "isrcext.h"
#include "isrcinit.h"


SPICEdev ISRCinfo = {
    {
        "Isource",  
        "Independent current source",

        &ISRCnSize,
        &ISRCnSize,
        ISRCnames,

        &ISRCpTSize,
        ISRCpTable,

        0,
        NULL,
	DEV_DEFAULT
    },

    DEVparam      : ISRCparam,
    DEVmodParam   : NULL,
    DEVload       : ISRCload,
    DEVsetup      : NULL,
    DEVunsetup    : NULL,
    DEVpzSetup    : NULL,
    DEVtemperature: ISRCtemp,
    DEVtrunc      : NULL,
    DEVfindBranch : NULL,
    DEVacLoad     : ISRCacLoad,
    DEVaccept     : ISRCaccept,
    DEVdestroy    : ISRCdestroy,
    DEVmodDelete  : ISRCmDelete,
    DEVdelete     : ISRCdelete,
    DEVsetic      : NULL,
    DEVask        : ISRCask,
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
                    
    DEVinstSize   : &ISRCiSize,
    DEVmodSize    : &ISRCmSize
};


SPICEdev *
get_isrc_info(void)
{
    return &ISRCinfo;
}
