#include <config.h>

#include <devdefs.h>

#include "b3soidditf.h"
#include "b3soiddext.h"
#include "b3soiddinit.h"

SPICEdev B3SOIDDinfo = {
    {   "B3SOIDD",
        "Berkeley SOI MOSFET (DD) model version 2.1",

        &B3SOIDDnSize,
        &B3SOIDDnSize,
        B3SOIDDnames,

        &B3SOIDDpTSize,
        B3SOIDDpTable,

        &B3SOIDDmPTSize,
        B3SOIDDmPTable,
	
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

    DEVparam      : B3SOIDDparam,
    DEVmodParam   : B3SOIDDmParam,
    DEVload       : B3SOIDDload,
    DEVsetup      : B3SOIDDsetup,
    DEVunsetup    : B3SOIDDunsetup,
    DEVpzSetup    : B3SOIDDsetup,
    DEVtemperature: B3SOIDDtemp,
    DEVtrunc      : B3SOIDDtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : B3SOIDDacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : B3SOIDDdestroy,
    DEVmodDelete  : B3SOIDDmDelete,
    DEVdelete     : B3SOIDDdelete, 
    DEVsetic      : B3SOIDDgetic,
    DEVask        : B3SOIDDask,
    DEVmodAsk     : B3SOIDDmAsk,
    DEVpzLoad     : B3SOIDDpzLoad,
    DEVconvTest   : B3SOIDDconvTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : B3SOIDDnoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif
    DEVinstSize   : &B3SOIDDiSize,
    DEVmodSize    : &B3SOIDDmSize
};

SPICEdev *
get_b3soidd_info (void)
{
  return &B3SOIDDinfo;
}


