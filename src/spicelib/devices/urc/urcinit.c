#include <config.h>

#include <devdefs.h>

#include "urcitf.h"
#include "urcext.h"
#include "urcinit.h"


SPICEdev URCinfo = {
    {
        "URC",      /* MUST precede both resistors and capacitors */
        "Uniform R.C. line",

        &URCnSize,
        &URCnSize,
        URCnames,

        &URCpTSize,
        URCpTable,

        &URCmPTSize,
        URCmPTable,
	0
    },

    DEVparam      : URCparam,
    DEVmodParam   : URCmParam,
    DEVload       : NULL,
    DEVsetup      : URCsetup,
    DEVunsetup    : URCunsetup,
    DEVpzSetup    : URCsetup,
    DEVtemperature: NULL,
    DEVtrunc      : NULL,
    DEVfindBranch : NULL,
    DEVacLoad     : NULL,
    DEVaccept     : NULL,
    DEVdestroy    : URCdestroy,
    DEVmodDelete  : URCmDelete,
    DEVdelete     : URCdelete,
    DEVsetic      : NULL,
    DEVask        : URCask,
    DEVmodAsk     : URCmAsk,
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
                    
    DEVinstSize   : &URCiSize,
    DEVmodSize    : &URCmSize

};


SPICEdev *
get_urc_info(void)
{
    return &URCinfo;
}
