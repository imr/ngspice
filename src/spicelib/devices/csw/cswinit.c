/* Modified: 2000 AlansFixes */

#include <config.h>

#include <devdefs.h>

#include "cswitf.h"
#include "cswext.h"
#include "cswinit.h"


SPICEdev CSWinfo = {
    {
	"CSwitch",
        "Current controlled ideal switch",

        &CSWnSize,
        &CSWnSize,
        CSWnames,

        &CSWpTSize,
        CSWpTable,

        &CSWmPTSize,
        CSWmPTable,
	0
    },

    DEVparam      : CSWparam,
    DEVmodParam   : CSWmParam,
    DEVload       : CSWload,
    DEVsetup      : CSWsetup,
    DEVunsetup    : NULL,
    DEVpzSetup    : CSWsetup,
    DEVtemperature: NULL,
    DEVtrunc      :CSWtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : CSWacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : CSWdestroy,
    DEVmodDelete  : CSWmDelete,
    DEVdelete     : CSWdelete,
    DEVsetic      : NULL,
    DEVask        : CSWask,
    DEVmodAsk     : CSWmAsk,
    DEVpzLoad     : CSWpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,	/* DISTO */
    DEVnoise      : CSWnoise,
                    
    DEVinstSize   : &CSWiSize,
    DEVmodSize    : &CSWmSize

};


SPICEdev *
get_csw_info(void)
{
    return &CSWinfo;
}
