#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "vdmositf.h"
#include "vdmosext.h"
#include "vdmosinit.h"


SPICEdev VDMOSinfo = {
    .DEVpublic = {
        .name = "VDMOS",
        .description = "DMOS model based on Level 1 MOSFET model",
        .terms = &VDMOSnSize,
        .numNames = &VDMOSnSize,
        .termNames = VDMOSnames,
        .numInstanceParms = &VDMOSpTSize,
        .instanceParms = VDMOSpTable,
        .numModelParms = &VDMOSmPTSize,
        .modelParms = VDMOSmPTable,
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

    .DEVparam = VDMOSparam,
    .DEVmodParam = VDMOSmParam,
    .DEVload = VDMOSload,
    .DEVsetup = VDMOSsetup,
    .DEVunsetup = VDMOSunsetup,
    .DEVpzSetup = VDMOSsetup,
    .DEVtemperature = VDMOStemp,
    .DEVtrunc = VDMOStrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = VDMOSacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = VDMOSgetic,
    .DEVask = VDMOSask,
    .DEVmodAsk = VDMOSmAsk,
    .DEVpzLoad = VDMOSpzLoad,
    .DEVconvTest = VDMOSconvTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = VDMOSdisto,
    .DEVnoise = VDMOSnoise,
    .DEVsoaCheck = VDMOSsoaCheck,
    .DEVinstSize = &VDMOSiSize,
    .DEVmodSize = &VDMOSmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_vdmos_info(void)
{
    return &VDMOSinfo;
}
