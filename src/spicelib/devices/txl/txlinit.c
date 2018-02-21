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
#ifdef XSPICE
	.cm_func = NULL,
	.num_conn = 0,
	.conn = NULL,
	.num_param = 0,
	.param = NULL,
	.num_inst_var = 0,
	.inst_var = NULL,
#endif
	.flags = 0,
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
    .DEVdestroy = TXLdestroy,
    .DEVmodDelete = TXLmDelete,
    .DEVdelete = TXLdelete,
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
#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
    .DEVinstSize = &TXLiSize,
    .DEVmodSize = &TXLmSize,
};

SPICEdev *
get_txl_info(void)
{
  return &TXLinfo;
}
