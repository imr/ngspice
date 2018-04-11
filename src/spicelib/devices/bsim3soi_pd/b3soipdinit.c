#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "b3soipditf.h"
#include "b3soipdinit.h"


SPICEdev B3SOIPDinfo = {
    .DEVpublic = {
	.name = "B3SOIPD",
	.description = "Berkeley SOI (PD) MOSFET model version 2.2.3",
	.terms = &B3SOIPDnSize,
	.numNames = &B3SOIPDnSize,
	.termNames = B3SOIPDnames,
	.numInstanceParms = &B3SOIPDpTSize,
	.instanceParms = B3SOIPDpTable,
	.numModelParms = &B3SOIPDmPTSize,
	.modelParms = B3SOIPDmPTable,
	.flags = DEV_DEFAULT,

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

    .DEVparam = B3SOIPDparam,
    .DEVmodParam = B3SOIPDmParam,
    .DEVload = B3SOIPDload,
    .DEVsetup = B3SOIPDsetup,
    .DEVunsetup = B3SOIPDunsetup,
    .DEVpzSetup = B3SOIPDsetup,
    .DEVtemperature = B3SOIPDtemp,
    .DEVtrunc = B3SOIPDtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = B3SOIPDacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = B3SOIPDgetic,
    .DEVask = B3SOIPDask,
    .DEVmodAsk = B3SOIPDmAsk,
    .DEVpzLoad = B3SOIPDpzLoad,
    .DEVconvTest = B3SOIPDconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = B3SOIPDnoise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &B3SOIPDiSize,
    .DEVmodSize = &B3SOIPDmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_b3soipd_info (void)
{
  return &B3SOIPDinfo;
}
