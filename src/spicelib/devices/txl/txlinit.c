/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/
#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "txlitf.h"
#include "txlext.h"
#include "txlinit.h"


SPICEdev TXLinfo = {
    .DEVpublic = {
	.name = "TransLine",
	.description = "Simple Lossy Transmission Line",
	.terms = &TXLnSize,
	.numNames = &TXLnSize,
	.termNames = TXLnames,
	.numInstanceParms = &TXLpTSize,
	.instanceParms = TXLpTable,
	.numModelParms = &TXLmPTSize,
	.modelParms = TXLmPTable,
	.flags = 0,

#ifdef XSPICE
	.cm_func = NULL,
	.num_conn = 0,
	.conn = NULL,
	.num_param = 0,
	.param = NULL,
	.num_inst_var = 0,
	.inst_var = NULL,
#endif
    },

    .DEVparam = TXLparam,
    .DEVmodParam = TXLmParam,
    .DEVload = TXLload,
    .DEVsetup = TXLsetup,
    .DEVunsetup = TXLunsetup,
    .DEVpzSetup = NULL,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = TXLload,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = TXLdevDelete,
    .DEVsetic = NULL,
    .DEVask = TXLask,
    .DEVmodAsk = TXLmodAsk,
    .DEVpzLoad = NULL,
    .DEVconvTest = NULL,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &TXLiSize,
    .DEVmodSize = &TXLmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_txl_info(void)
{
  return &TXLinfo;
}
