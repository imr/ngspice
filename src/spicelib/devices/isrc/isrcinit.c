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
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &ISRCiSize,
    DEVmodSize    : &ISRCmSize
};


SPICEdev *
get_isrc_info(void)
{
    return &ISRCinfo;
}
