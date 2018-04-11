#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "jfetitf.h"
#include "jfetext.h"
#include "jfetinit.h"


SPICEdev JFETinfo = {
    .DEVpublic = {
        .name = "JFET",
        .description = "Junction Field effect transistor",
        .terms = &JFETnSize,
        .numNames = &JFETnSize,
        .termNames = JFETnames,
        .numInstanceParms = &JFETpTSize,
        .instanceParms = JFETpTable,
        .numModelParms = &JFETmPTSize,
        .modelParms = JFETmPTable,
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

    .DEVparam = JFETparam,
    .DEVmodParam = JFETmParam,
    .DEVload = JFETload,
    .DEVsetup = JFETsetup,
    .DEVunsetup = JFETunsetup,
    .DEVpzSetup = JFETsetup,
    .DEVtemperature = JFETtemp,
    .DEVtrunc = JFETtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = JFETacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = JFETgetic,
    .DEVask = JFETask,
    .DEVmodAsk = JFETmAsk,
    .DEVpzLoad = JFETpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = JFETdisto,
    .DEVnoise = JFETnoise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &JFETiSize,
    .DEVmodSize = &JFETmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_jfet_info(void)
{
    return &JFETinfo;
}
