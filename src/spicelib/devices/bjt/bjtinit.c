#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bjtitf.h"
#include "bjtext.h"
#include "bjtinit.h"


SPICEdev BJTinfo = {
    .DEVpublic = {
	.name = "BJT",
	.description = "Bipolar Junction Transistor",
	.terms = &BJTnSize,
	.numNames = &BJTnSize,
	.termNames = BJTnames,
	.numInstanceParms = &BJTpTSize,
	.instanceParms = BJTpTable,
	.numModelParms = &BJTmPTSize,
	.modelParms = BJTmPTable,
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

    .DEVparam = BJTparam,
    .DEVmodParam = BJTmParam,
    .DEVload = BJTload,
    .DEVsetup = BJTsetup,
    .DEVunsetup = BJTunsetup,
    .DEVpzSetup = BJTsetup,
    .DEVtemperature = BJTtemp,
    .DEVtrunc = BJTtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BJTacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = BJTdelete,
    .DEVsetic = BJTgetic,
    .DEVask = BJTask,
    .DEVmodAsk = BJTmAsk,
    .DEVpzLoad = BJTpzLoad,
    .DEVconvTest = BJTconvTest,
    .DEVsenSetup = BJTsSetup,
    .DEVsenLoad = BJTsLoad,
    .DEVsenUpdate = BJTsUpdate,
    .DEVsenAcLoad = BJTsAcLoad,
    .DEVsenPrint = BJTsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = BJTdisto,
    .DEVnoise = BJTnoise,
    .DEVsoaCheck = BJTsoaCheck,
    .DEVinstSize = &BJTiSize,
    .DEVmodSize = &BJTmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bjt_info(void)
{
    return &BJTinfo;
}
