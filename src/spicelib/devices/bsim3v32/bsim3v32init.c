#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim3v32itf.h"
#include "bsim3v32ext.h"
#include "bsim3v32init.h"


SPICEdev BSIM3v32info = {
    .DEVpublic = {
        .name = "BSIM3v32",
        .description = "Berkeley Short Channel IGFET Model Version-3",
        .terms = &BSIM3v32nSize,
        .numNames = &BSIM3v32nSize,
        .termNames = BSIM3v32names,
        .numInstanceParms = &BSIM3v32pTSize,
        .instanceParms = BSIM3v32pTable,
        .numModelParms = &BSIM3v32mPTSize,
        .modelParms = BSIM3v32mPTable,
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

    .DEVparam = BSIM3v32param,
    .DEVmodParam = BSIM3v32mParam,
    .DEVload = BSIM3v32load,
    .DEVsetup = BSIM3v32setup,
    .DEVunsetup = BSIM3v32unsetup,
    .DEVpzSetup = BSIM3v32setup,
    .DEVtemperature = BSIM3v32temp,
    .DEVtrunc = BSIM3v32trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = BSIM3v32acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = BSIM3v32mDelete,
    .DEVdelete = NULL,
    .DEVsetic = BSIM3v32getic,
    .DEVask = BSIM3v32ask,
    .DEVmodAsk = BSIM3v32mAsk,
    .DEVpzLoad = BSIM3v32pzLoad,
    .DEVconvTest = BSIM3v32convTest,
    .DEVsenSetup = NULL,
    .DEVsenLoad = NULL,
    .DEVsenUpdate = NULL,
    .DEVsenAcLoad = NULL,
    .DEVsenPrint = NULL,
    .DEVsenTrunc = NULL,
    .DEVdisto = NULL,
    .DEVnoise = BSIM3v32noise,
    .DEVsoaCheck = BSIM3v32soaCheck,
    .DEVinstSize = &BSIM3v32iSize,
    .DEVmodSize = &BSIM3v32mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_bsim3v32_info(void)
{
    return &BSIM3v32info;
}
