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

  CPLparam,
  CPLmParam,
  CPLload,
  CPLsetup,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL, /* CPLfindBranch, */
  NULL, 
  NULL,
  CPLdestroy,
#ifdef DELETES
  CPLmDelete,
  CPLdelete,
#else /* DELETES */
  NULL,
  NULL,
#endif /* DELETES */
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,

  &CPLiSize,
  &CPLmSize

};

SPICEdev *
get_cpl_info(void)
{
  return &CPLinfo;
}
