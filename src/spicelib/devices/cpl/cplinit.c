#include <config.h>

#include <devdefs.h>

#include "cplitf.h"
#include "cplext.h"
#include "cplinit.h"

SPICEdev CPLinfo = {
  {
    "CplLines",
    "Simple Coupled Multiconductor Lines",

    &CPLnSize,
    &CPLnSize,
    CPLnames,

    &CPLpTSize,
    CPLpTable,

    &CPLmPTSize,
    CPLmPTable,

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

 
  DEVparam       : CPLparam,
  DEVmodParam    : CPLmParam,
  DEVload        : CPLload,
  DEVsetup       : CPLsetup,
  DEVunsetup     : CPLunsetup,
  DEVpzSetup     : NULL,
  DEVtemperature : NULL,
  DEVtrunc       : NULL,
  DEVfindBranch  : NULL, /* CPLfindBranch, */
  DEVacLoad      : NULL, 
  DEVaccept      : NULL,
  DEVdestroy     : CPLdestroy,
  DEVmodDelete   : CPLmDelete,
  DEVdelete      : CPLdelete,
  DEVsetic       : NULL,
  DEVask         : CPLask,
  DEVmodAsk      : CPLmAsk,
  DEVpzLoad      : NULL,
  DEVconvTest    : NULL,
  DEVsenSetup    : NULL,
  DEVsenLoad     : NULL,
  DEVsenUpdate   : NULL,
  DEVsenAcLoad   : NULL,
  DEVsenPrint    : NULL,
  DEVsenTrunc    : NULL,
  DEVdisto       : NULL,
  DEVnoise       : NULL,
#ifdef CIDER
  DEVdump        : NULL,
  DEVacct        : NULL,
#endif   
  DEVinstSize    : &CPLiSize,
  DEVmodSize     : &CPLmSize

};

SPICEdev *
get_cpl_info(void)
{
  return &CPLinfo;
}
