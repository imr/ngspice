#include <config.h>

#include <devdefs.h>

#include "b3soiitf.h"
#include "b3soiext.h"
#include "b3soiinit.h"

SPICEdev B3SOIinfo = {
    {   "B3SOI",
        "Berkeley SOI MOSFET  model version 3.0",

        &B3SOInSize,
        &B3SOInSize,
        B3SOInames,

        &B3SOIpTSize,
        B3SOIpTable,

        &B3SOImPTSize,
        B3SOImPTable,
	
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

    DEVparam      : B3SOIparam,
    DEVmodParam   : B3SOImParam,
    DEVload       : B3SOIload,
    DEVsetup      : B3SOIsetup,
    DEVunsetup    : B3SOIunsetup,
    DEVpzSetup    : B3SOIsetup,
    DEVtemperature: B3SOItemp,
    DEVtrunc      : B3SOItrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : B3SOIacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : B3SOIdestroy,
    DEVmodDelete  : B3SOImDelete,
    DEVdelete     : B3SOIdelete, 
    DEVsetic      : B3SOIgetic,
    DEVask        : B3SOIask,
    DEVmodAsk     : B3SOImAsk,
    DEVpzLoad     : B3SOIpzLoad,
    DEVconvTest   : B3SOIconvTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : B3SOInoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif
    DEVinstSize   : &B3SOIiSize,
    DEVmodSize    : &B3SOImSize
};

SPICEdev *
get_b3soi_info (void)
{
  return &B3SOIinfo;
}


