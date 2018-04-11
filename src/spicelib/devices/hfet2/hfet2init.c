#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hfet2itf.h"
#include "hfet2ext.h"
#include "hfet2init.h"


SPICEdev HFET2info = {
    .DEVpublic = {
        .name = "HFET2",
        .description = "HFET2 Model",
        .terms = &HFET2nSize,
        .numNames = &HFET2nSize,
        .termNames = HFET2names,
        .numInstanceParms = &HFET2pTSize,
        .instanceParms = HFET2pTable,
        .numModelParms = &HFET2mPTSize,
        .modelParms = HFET2mPTable,
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

    .DEVparam = HFET2param,
    .DEVmodParam = HFET2mParam,
    .DEVload = HFET2load,
    .DEVsetup = HFET2setup,
    .DEVunsetup = HFET2unsetup,
    .DEVpzSetup = HFET2setup,
    .DEVtemperature = HFET2temp,
    .DEVtrunc = HFET2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = HFET2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = HFET2getic,
    .DEVask = HFET2ask,
    .DEVmodAsk = HFET2mAsk,
    .DEVpzLoad = HFET2pzLoad,
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
    .DEVinstSize = &HFET2iSize,
    .DEVmodSize = &HFET2mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_hfet2_info(void)
{
    return &HFET2info;
}
