#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "mesitf.h"
#include "mesext.h"
#include "mesinit.h"


SPICEdev MESinfo = {
    .DEVpublic = {
        .name = "MES",
        .description = "GaAs MESFET model",
        .terms = &MESnSize,
        .numNames = &MESnSize,
        .termNames = MESnames,
        .numInstanceParms = &MESpTSize,
        .instanceParms = MESpTable,
        .numModelParms = &MESmPTSize,
        .modelParms = MESmPTable,
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

    .DEVparam = MESparam,
    .DEVmodParam = MESmParam,
    .DEVload = MESload,
    .DEVsetup = MESsetup,
    .DEVunsetup = MESunsetup,
    .DEVpzSetup = MESsetup,
    .DEVtemperature = MEStemp,
    .DEVtrunc = MEStrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = MESacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = MESgetic,
    .DEVask = MESask,
    .DEVmodAsk = MESmAsk,
    .DEVpzLoad = MESpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = MESdisto,
    .DEVnoise = MESnoise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &MESiSize,
    .DEVmodSize = &MESmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_mes_info(void)
{
    return &MESinfo;
}
