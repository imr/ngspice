#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "urcitf.h"
#include "urcext.h"
#include "urcinit.h"


SPICEdev URCinfo = {
    .DEVpublic = {
        .name = "URC",
        .description = "Uniform R.C. line",
        .terms = &URCnSize,
        .numNames = &URCnSize,
        .termNames = URCnames,
        .numInstanceParms = &URCpTSize,
        .instanceParms = URCpTable,
        .numModelParms = &URCmPTSize,
        .modelParms = URCmPTable,
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

    .DEVparam = URCparam,
    .DEVmodParam = URCmParam,
    .DEVload = NULL,
    .DEVsetup = URCsetup,
    .DEVunsetup = URCunsetup,
    .DEVpzSetup = URCsetup,
    .DEVtemperature = NULL,
    .DEVtrunc = NULL,
    .DEVfindBranch = NULL,
    .DEVacLoad = NULL,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = NULL,
    .DEVask = URCask,
    .DEVmodAsk = URCmAsk,
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
    .DEVinstSize = &URCiSize,
    .DEVmodSize = &URCmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_urc_info(void)
{
    return &URCinfo;
}
