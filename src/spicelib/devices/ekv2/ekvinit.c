#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "ekvitf.h"
#include "ekvext.h"
#include "ekvinit.h"


SPICEdev EKVinfo = {
    .DEVpublic = {
        .name = "EKV",
        .description = "EPLF-EKV v2.6 MOSFET model",
        .terms = &EKVnSize,
        .numNames = &EKVnSize,
        .termNames = EKVnames,
        .numInstanceParms = &EKVpTSize,
        .instanceParms = EKVpTable,
        .numModelParms = &EKVmPTSize,
        .modelParms = EKVmPTable,
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

    .DEVparam = EKVparam,
    .DEVmodParam = EKVmParam,
    .DEVload = EKVload,
    .DEVsetup = EKVsetup,
    .DEVunsetup = EKVunsetup,
    .DEVpzSetup = EKVsetup,
    .DEVtemperature = EKVtemp,
    .DEVtrunc = EKVtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = EKVacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = EKVdelete,
    .DEVsetic = EKVgetic,
    .DEVask = EKVask,
    .DEVmodAsk = EKVmAsk,
    .DEVpzLoad = NULL,
    .DEVconvTest = EKVconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = EKVnoise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &EKViSize,
    .DEVmodSize = &EKVmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_ekv_info(void)
{
    return &EKVinfo;
}
