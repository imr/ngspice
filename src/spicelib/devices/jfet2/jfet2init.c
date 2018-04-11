#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "jfet2itf.h"
#include "jfet2ext.h"
#include "jfet2init.h"


SPICEdev JFET2info = {
    .DEVpublic = {
        .name = "JFET2",
        .description = "Short channel field effect transistor",
        .terms = &JFET2nSize,
        .numNames = &JFET2nSize,
        .termNames = JFET2names,
        .numInstanceParms = &JFET2pTSize,
        .instanceParms = JFET2pTable,
        .numModelParms = &JFET2mPTSize,
        .modelParms = JFET2mPTable,
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

    .DEVparam = JFET2param,
    .DEVmodParam = JFET2mParam,
    .DEVload = JFET2load,
    .DEVsetup = JFET2setup,
    .DEVunsetup = JFET2unsetup,
    .DEVpzSetup = JFET2setup,
    .DEVtemperature = JFET2temp,
    .DEVtrunc = JFET2trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = JFET2acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = JFET2getic,
    .DEVask = JFET2ask,
    .DEVmodAsk = JFET2mAsk,
    .DEVpzLoad = NULL,
    .DEVconvTest = NULL,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = JFET2noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &JFET2iSize,
    .DEVmodSize = &JFET2mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_jfet2_info(void)
{
    return &JFET2info;
}
