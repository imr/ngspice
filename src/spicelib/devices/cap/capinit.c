#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "capitf.h"
#include "capext.h"
#include "capinit.h"


SPICEdev CAPinfo = {
    .DEVpublic = {
        .name = "Capacitor",
        .description = "Fixed capacitor",
        .terms = &CAPnSize,
        .numNames = &CAPnSize,
        .termNames = CAPnames,
        .numInstanceParms = &CAPpTSize,
        .instanceParms = CAPpTable,
        .numModelParms = &CAPmPTSize,
        .modelParms = CAPmPTable,
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
    .DEVparam = CAPparam,
    .DEVmodParam = CAPmParam,
    .DEVload = CAPload,
    .DEVsetup = CAPsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = CAPsetup,
    .DEVtemperature = CAPtemp,
    .DEVtrunc = CAPtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = CAPacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = CAPdestroy,
    .DEVmodDelete = CAPmDelete,
    .DEVdelete = CAPdelete,
    .DEVsetic = CAPgetic,
    .DEVask = CAPask,
    .DEVmodAsk = CAPmAsk,
    .DEVpzLoad = CAPpzLoad,
    .DEVconvTest = NULL,
    .DEVsenSetup = CAPsSetup,
    .DEVsenLoad = CAPsLoad,
    .DEVsenUpdate = CAPsUpdate,
    .DEVsenAcLoad = CAPsAcLoad,
    .DEVsenPrint = CAPsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = NULL,
    .DEVsoaCheck = CAPsoaCheck,
#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
    .DEVinstSize = &CAPiSize,
    .DEVmodSize = &CAPmSize,
};


SPICEdev *
get_cap_info(void)
{
    return &CAPinfo;
}
