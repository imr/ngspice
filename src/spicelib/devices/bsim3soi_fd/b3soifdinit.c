#include <config.h>

#include <devdefs.h>

#include "b3soifditf.h"
#include "b3soifdext.h"
#include "b3soifdinit.h"

SPICEdev B3SOIFDinfo = {
    {   "B3SOIFD",
        "Berkeley SOI MOSFET (FD) model version 2.1",

        &B3SOIFDnSize,
        &B3SOIFDnSize,
        B3SOIFDnames,

        &B3SOIFDpTSize,
        B3SOIFDpTable,

        &B3SOIFDmPTSize,
        B3SOIFDmPTable,
	
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

    DEVparam      : B3SOIFDparam,
    DEVmodParam   : B3SOIFDmParam,
    DEVload       : B3SOIFDload,
    DEVsetup      : B3SOIFDsetup,
    DEVunsetup    : B3SOIFDunsetup,
    DEVpzSetup    : B3SOIFDsetup,
    DEVtemperature: B3SOIFDtemp,
    DEVtrunc      : B3SOIFDtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : B3SOIFDacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : B3SOIFDdestroy,
    DEVmodDelete  : B3SOIFDmDelete,
    DEVdelete     : B3SOIFDdelete, 
    DEVsetic      : B3SOIFDgetic,
    DEVask        : B3SOIFDask,
    DEVmodAsk     : B3SOIFDmAsk,
    DEVpzLoad     : B3SOIFDpzLoad,
    DEVconvTest   : B3SOIFDconvTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : B3SOIFDnoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif    
    DEVinstSize:	&B3SOIFDiSize,
    DEVmodSize:	&B3SOIFDmSize
};






SPICEdev *
get_b3soifd_info (void)
{
  return &B3SOIFDinfo;
}
