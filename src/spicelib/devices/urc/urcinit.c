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
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &URCiSize,
    DEVmodSize    : &URCmSize

};


SPICEdev *
get_urc_info(void)
{
    return &URCinfo;
}
