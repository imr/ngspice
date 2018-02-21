#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "nbjtitf.h"
#include "nbjtext.h"
#include "nbjtinit.h"


SPICEdev NBJTinfo = {
    .DEVpublic = {
        .name = "NBJT",
        .description = "1D Numerical Bipolar Junction Transistor model",
        .terms = &NBJTnSize,
        .numNames = &NBJTnSize,
        .termNames = NBJTnames,
        .numInstanceParms = &NBJTpTSize,
        .instanceParms = NBJTpTable,
        .numModelParms = &NBJTmPTSize,
        .modelParms = NBJTmPTable,
#ifdef XSPICE
        .cm_func = NULL,
        .num_conn = 0,
        .conn = NULL,
        .num_param = 0,
        .param = NULL,
        .num_inst_var = 0,
        .inst_var = NULL,
#endif
        .flags = DEV_DEFAULT,
    },
    .DEVparam = NBJTparam,
    .DEVmodParam = NBJTmParam,
    .DEVload = NBJTload,
    .DEVsetup = NBJTsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = NBJTsetup,
    .DEVtemperature = NBJTtemp,
    .DEVtrunc = NBJTtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = NBJTacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NBJTdestroy,
    .DEVmodDelete = NBJTmDelete,
    .DEVdelete = NBJTdelete,
    .DEVsetic = NULL,
    .DEVask = NBJTask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = NBJTpzLoad,
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
    .DEVdump = NBJTdump,
    .DEVacct = NBJTacct,
#endif
    .DEVinstSize = &NBJTiSize,
    .DEVmodSize = &NBJTmSize,
};


SPICEdev *
get_nbjt_info(void)
{
    return &NBJTinfo;
}
