#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hsmhv2def.h"
#include "hsmhv2itf.h"
#include "hsmhv2init.h"


SPICEdev HSMHV2info = {
    .DEVpublic = {
        .name = "HiSIMHV2",
        .description = "Hiroshima University STARC IGFET Model - HiSIM_HV v.2",
        .terms = &HSMHV2nSize,
        .numNames = &HSMHV2nSize,
        .termNames = HSMHV2names,
        .numInstanceParms = &HSMHV2pTSize,
        .instanceParms = HSMHV2pTable,
        .numModelParms = &HSMHV2mPTSize,
        .modelParms = HSMHV2mPTable,
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

    .DEVparam = HSMHV2param,
    .DEVmodParam = HSMHV2mParam,
    .DEVload = HSMHV2load,
    .DEVsetup = HSMHV2setup,
    .DEVunsetup = HSMHV2unsetup,
    .DEVpzSetup = HSMHV2setup,
    .DEVtemperature = HSMHV2temp,
    .DEVtrunc = HSMHV2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = HSMHV2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = HSMHV2getic,
    .DEVask = HSMHV2ask,
    .DEVmodAsk = HSMHV2mAsk,
    .DEVpzLoad = HSMHV2pzLoad,
    .DEVconvTest = HSMHV2convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = HSMHV2noise,
    .DEVsoaCheck = HSMHV2soaCheck,
    .DEVinstSize = &HSMHV2iSize,
    .DEVmodSize = &HSMHV2mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_hsmhv2_info(void)
{
    return &HSMHV2info;
}
