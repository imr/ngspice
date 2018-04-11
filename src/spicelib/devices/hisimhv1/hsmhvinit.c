#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hsmhvdef.h"
#include "hsmhvitf.h"
#include "hsmhvinit.h"


SPICEdev HSMHVinfo = {
    .DEVpublic = {
        .name = "HiSIMHV1",
        .description = "Hiroshima University STARC IGFET Model - HiSIM_HV v.1",
        .terms = &HSMHVnSize,
        .numNames = &HSMHVnSize,
        .termNames = HSMHVnames,
        .numInstanceParms = &HSMHVpTSize,
        .instanceParms = HSMHVpTable,
        .numModelParms = &HSMHVmPTSize,
        .modelParms = HSMHVmPTable,
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

    .DEVparam = HSMHVparam,
    .DEVmodParam = HSMHVmParam,
    .DEVload = HSMHVload,
    .DEVsetup = HSMHVsetup,
    .DEVunsetup = HSMHVunsetup,
    .DEVpzSetup = HSMHVsetup,
    .DEVtemperature = HSMHVtemp,
    .DEVtrunc = HSMHVtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = HSMHVacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = HSMHVgetic,
    .DEVask = HSMHVask,
    .DEVmodAsk = HSMHVmAsk,
    .DEVpzLoad = HSMHVpzLoad,
    .DEVconvTest = HSMHVconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = HSMHVnoise,
    .DEVsoaCheck = HSMHVsoaCheck,
    .DEVinstSize = &HSMHViSize,
    .DEVmodSize = &HSMHVmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_hsmhv_info(void)
{
    return &HSMHVinfo;
}
