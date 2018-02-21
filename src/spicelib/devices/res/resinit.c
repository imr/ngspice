#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "resitf.h"
#include "resext.h"
#include "resinit.h"


SPICEdev RESinfo = {
    .DEVpublic = {
        .name = "Resistor",
        .description = "Simple linear resistor",
        .terms = &RESnSize,
        .numNames = &RESnSize,
        .termNames = RESnames,
        .numInstanceParms = &RESpTSize,
        .instanceParms = RESpTable,
        .numModelParms = &RESmPTSize,
        .modelParms = RESmPTable,
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
    .DEVparam = RESparam,
    .DEVmodParam = RESmParam,
    .DEVload = RESload,
    .DEVsetup = RESsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = RESsetup,
    .DEVtemperature = REStemp,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = RESacload,
    .DEVaccept = NULL,
    .DEVdestroy = RESdestroy,
    .DEVmodDelete = RESmDelete,
    .DEVdelete = RESdelete,
    .DEVsetic = NULL,
    .DEVask = RESask,
    .DEVmodAsk = RESmodAsk,
    .DEVpzLoad = RESpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = RESsSetup,
    .DEVsenLoad = RESsLoad,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = RESsAcLoad,
    .DEVsenPrint = RESsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = RESnoise,
    .DEVsoaCheck = RESsoaCheck,
#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
    .DEVinstSize = &RESiSize,
    .DEVmodSize = &RESmSize,
};


SPICEdev *
get_res_info(void)
{
    return &RESinfo;
}
