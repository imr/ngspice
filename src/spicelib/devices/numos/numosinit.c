#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "numositf.h"
#include "numosext.h"
#include "numosinit.h"


SPICEdev NUMOSinfo = {
    .DEVpublic = {
        .name = "NUMOS",
        .description = "2D Numerical MOS Field Effect Transistor model",
        .terms = &NUMOSnSize,
        .numNames = &NUMOSnSize,
        .termNames = NUMOSnames,
        .numInstanceParms = &NUMOSpTSize,
        .instanceParms = NUMOSpTable,
        .numModelParms = &NUMOSmPTSize,
        .modelParms = NUMOSmPTable,
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

    .DEVparam = NUMOSparam,
    .DEVmodParam = NUMOSmParam,
    .DEVload = NUMOSload,
    .DEVsetup = NUMOSsetup,
    .DEVunsetup = NULL,
    .DEVpzSetup = NUMOSsetup,
    .DEVtemperature = NUMOStemp,
    .DEVtrunc = NUMOStrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = NUMOSacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NUMOSmodDelete,
    .DEVdelete = NUMOSdelete,
    .DEVsetic = NULL,
    .DEVask = NUMOSask,
    .DEVmodAsk = NULL,
    .DEVpzLoad = NUMOSpzLoad,
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
    .DEVinstSize = &NUMOSiSize,
    .DEVmodSize = &NUMOSmSize,

#ifdef CIDER
    .DEVdump = NUMOSdump,
    .DEVacct = NUMOSacct,
#endif
};


SPICEdev *
get_numos_info(void)
{
    return &NUMOSinfo;
}
