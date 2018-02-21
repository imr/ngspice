#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "cplitf.h"
#include "cplext.h"
#include "cplinit.h"

SPICEdev CPLinfo = {
    .DEVpublic = {
	.name = "CplLines",
	.description = "Simple Coupled Multiconductor Lines",
	.terms = &CPLnSize,
	.numNames = &CPLnSize,
	.termNames = CPLnames,
	.numInstanceParms = &CPLpTSize,
	.instanceParms = CPLpTable,
	.numModelParms = &CPLmPTSize,
	.modelParms = CPLmPTable,
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
    .DEVparam = CPLparam,
    .DEVmodParam = CPLmParam,
    .DEVload = CPLload,
    .DEVsetup = CPLsetup,
    .DEVunsetup = CPLunsetup,
    .DEVpzSetup = NULL,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = NULL,
    .DEVaccept = NULL,
    .DEVdestroy = CPLdestroy,
    .DEVmodDelete = CPLmDelete,
    .DEVdelete = CPLdelete,
    .DEVsetic = NULL,
    .DEVask = CPLask,
    .DEVmodAsk = CPLmAsk,
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
    .DEVinstSize = &CPLiSize,
    .DEVmodSize = &CPLmSize,
};

SPICEdev *
get_cpl_info(void)
{
  return &CPLinfo;
}
