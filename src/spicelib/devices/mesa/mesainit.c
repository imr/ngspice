#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "mesaitf.h"
#include "mesaext.h"
#include "mesainit.h"


SPICEdev MESAinfo = {
    .DEVpublic = {
        .name = "MESA",
        .description = "GaAs MESFET model",
        .terms = &MESAnSize,
        .numNames = &MESAnSize,
        .termNames = MESAnames,
        .numInstanceParms = &MESApTSize,
        .instanceParms = MESApTable,
        .numModelParms = &MESAmPTSize,
        .modelParms = MESAmPTable,
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
    .DEVparam = MESAparam,
    .DEVmodParam = MESAmParam,
    .DEVload = MESAload,
    .DEVsetup = MESAsetup,
    .DEVunsetup = MESAunsetup,
    .DEVpzSetup = MESAsetup,
    .DEVtemperature = MESAtemp,
    .DEVtrunc = MESAtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = MESAacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = MESAdestroy,
    .DEVmodDelete = MESAmDelete,
    .DEVdelete = MESAdelete,
    .DEVsetic = MESAgetic,
    .DEVask = MESAask,
    .DEVmodAsk = MESAmAsk,
    .DEVpzLoad = MESApzLoad,
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
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
    .DEVinstSize = &MESAiSize,
    .DEVmodSize = &MESAmSize,
};


SPICEdev *
get_mesa_info(void)
{
    return &MESAinfo;
}
