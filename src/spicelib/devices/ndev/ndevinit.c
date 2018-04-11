#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "ndevitf.h"
#include "ndevext.h"
#include "ndevinit.h"


SPICEdev NDEVinfo = {
    .DEVpublic = {
        .name = "NDEV",
        .description = "Numerical Device",
        .terms = &NDEVnSize,
        .numNames = &NDEVnSize,
        .termNames = NDEVnames,
        .numInstanceParms = &NDEVpTSize,
        .instanceParms = NDEVpTable,
        .numModelParms = &NDEVmPTSize,
        .modelParms = NDEVmPTable,
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

    .DEVparam = NDEVparam,
    .DEVmodParam = NDEVmParam,
    .DEVload = NDEVload,
    .DEVsetup = NDEVsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = NDEVsetup,
    .DEVtemperature = NDEVtemp,
    .DEVtrunc = NDEVtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = NDEVacLoad,
    .DEVaccept = NDEVaccept,
    .DEVdestroy = NULL,
    .DEVmodDelete = NDEVmDelete,
    .DEVdelete = NULL,
    .DEVsetic = NDEVgetic,
    .DEVask = NDEVask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = NDEVpzLoad,
    .DEVconvTest = NDEVconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &NDEViSize,
    .DEVmodSize = &NDEVmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_ndev_info(void)
{
    return &NDEVinfo;
}
