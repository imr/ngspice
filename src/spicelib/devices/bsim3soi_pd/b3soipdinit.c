#include <config.h>

#include <devdefs.h>

#include "b3soipditf.h"
#include "b3soipdext.h"
#include "b3soipdinit.h"


SPICEdev B3SOIPDinfo = {
  {"B3SOIPD",
   "Berkeley SOI (PD) MOSFET model version 2.2.3",

   &B3SOIPDnSize,
   &B3SOIPDnSize,
   B3SOIPDnames,

   &B3SOIPDpTSize,
   B3SOIPDpTable,

   &B3SOIPDmPTSize,
   B3SOIPDmPTable,
   
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

    DEVparam:       B3SOIPDparam,
    DEVmodParam:    B3SOIPDmParam,
    DEVload:        B3SOIPDload,
    DEVsetup:       B3SOIPDsetup,
    DEVunsetup:     B3SOIPDunsetup,
    DEVpzSetup:     B3SOIPDsetup,
    DEVtemperature: B3SOIPDtemp,
    DEVtrunc:       B3SOIPDtrunc,
    DEVfindBranch:  NULL,
    DEVacLoad:      B3SOIPDacLoad,
    DEVaccept:      NULL,
    DEVdestroy:     B3SOIPDdestroy,
    DEVmodDelete:   B3SOIPDmDelete,
    DEVdelete:      B3SOIPDdelete,
    DEVsetic:       B3SOIPDgetic,
    DEVask:         B3SOIPDask,
    DEVmodAsk:      B3SOIPDmAsk,
    DEVpzLoad:      B3SOIPDpzLoad,
    DEVconvTest:    B3SOIPDconvTest,
    DEVsenSetup:    NULL,
    DEVsenLoad:     NULL,
    DEVsenUpdate:   NULL,
    DEVsenAcLoad:   NULL,
    DEVsenPrint:    NULL,
    DEVsenTrunc:    NULL,
    DEVdisto:       NULL,
    DEVnoise:       B3SOIPDnoise,
#ifdef CIDER
    DEVdump:        NULL,
    DEVacct:        NULL,
#endif
    DEVinstSize:   &B3SOIPDiSize,
    DEVmodSize:    &B3SOIPDmSize
};

SPICEdev *
get_b3soipd_info (void)
{
  return &B3SOIPDinfo;
}
