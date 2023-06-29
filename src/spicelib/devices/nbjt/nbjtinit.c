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
    .DEVdestroy = NULL,
    .DEVmodDelete = NBJTmodDelete,
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
    .DEVinstSize = &NBJTiSize,
    .DEVmodSize = &NBJTmSize,

#ifdef CIDER
    .DEVdump = NBJTdump,
    .DEVacct = NBJTacct,
#endif
};


SPICEdev *
get_nbjt_info(void)
{
    return &NBJTinfo;
}
