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
