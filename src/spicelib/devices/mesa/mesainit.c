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
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
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
    .DEVinstSize = &MESAiSize,
    .DEVmodSize = &MESAmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_mesa_info(void)
{
    return &MESAinfo;
}
