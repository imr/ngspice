#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim3itf.h"
#include "bsim3ext.h"
#include "bsim3init.h"


SPICEdev BSIM3info = {
    .DEVpublic = {
        .name = "BSIM3",
        .description = "Berkeley Short Channel IGFET Model Version-3",
        .terms = &BSIM3nSize,
        .numNames = &BSIM3nSize,
        .termNames = BSIM3names,
        .numInstanceParms = &BSIM3pTSize,
        .instanceParms = BSIM3pTable,
        .numModelParms = &BSIM3mPTSize,
        .modelParms = BSIM3mPTable,
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

    .DEVparam = BSIM3param,
    .DEVmodParam = BSIM3mParam,
    .DEVload = BSIM3load,
    .DEVsetup = BSIM3setup,
    .DEVunsetup = BSIM3unsetup,
    .DEVpzSetup = BSIM3setup,
    .DEVtemperature = BSIM3temp,
    .DEVtrunc = BSIM3trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM3acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM3mDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM3getic,
    .DEVask = BSIM3ask,
    .DEVmodAsk = BSIM3mAsk,
    .DEVpzLoad = BSIM3pzLoad,
    .DEVconvTest = BSIM3convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM3noise,
    .DEVsoaCheck = BSIM3soaCheck,
    .DEVinstSize = &BSIM3iSize,
    .DEVmodSize = &BSIM3mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim3_info(void)
{
    return &BSIM3info;
}


