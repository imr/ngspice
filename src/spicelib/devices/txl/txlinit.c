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

  DEVparam      : TXLparam,
  DEVmodParam   : TXLmParam,
  DEVload       : TXLload,
  DEVsetup      : TXLsetup,
  DEVunsetup    : TXLunsetup,
  DEVpzSetup    : NULL,
  DEVtemperature: NULL,
  DEVtrunc      : NULL,
  DEVfindBranch : NULL, /* TXLfindBranch default: disabled */
  DEVacLoad     : TXLload, /* ac load */
  DEVaccept     : NULL, /* TXLaccept default: disabled */
  DEVdestroy    : TXLdestroy,
  DEVmodDelete  : TXLmDelete,
  DEVdelete     : TXLdelete,
  DEVsetic      : NULL,
  DEVask        : TXLask,
  DEVmodAsk     : TXLmodAsk,
  DEVpzLoad     : NULL,
  DEVconvTest   : NULL,
  DEVsenSetup   : NULL,
  DEVsenLoad    : NULL,
  DEVsenUpdate  : NULL,
  DEVsenAcLoad  : NULL,
  DEVsenPrint   : NULL,
  DEVsenTrunc   : NULL,
  DEVdisto      : NULL,
  DEVnoise      : NULL,
#ifdef CIDER
  DEVdump       : NULL,
  DEVacct       : NULL,  
#endif
  &TXLiSize,
  &TXLmSize

};

SPICEdev *
get_txl_info(void)
{
  return &TXLinfo;
}
