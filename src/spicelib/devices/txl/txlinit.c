/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/
#include <config.h>

#include <devdefs.h>

#include "txlitf.h"
#include "txlext.h"
#include "txlinit.h"


SPICEdev TXLinfo = {
  {
    "TransLine",
    "Simple Lossy Transmission Line",

    &TXLnSize,
    &TXLnSize,
    TXLnames,

    &TXLpTSize,
    TXLpTable,

    &TXLmPTSize,
    TXLmPTable,

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

  TXLparam,
  TXLmParam,
  TXLload,
  TXLsetup,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL, /* TXLfindBranch, */
  TXLload, /* ac load */
  NULL,
  TXLdestroy,
#ifdef DELETES
  TXLmDelete,
  TXLdelete,
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

  &TXLiSize,
  &TXLmSize

};

SPICEdev *
get_txl_info(void)
{
  return &TXLinfo;
}
