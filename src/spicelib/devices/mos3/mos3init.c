#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "mos3itf.h"
#include "mos3ext.h"
#include "mos3init.h"


SPICEdev MOS3info = {
    .DEVpublic = {
        .name = "Mos3",
        .description = "Level 3 MOSfet model with Meyer capacitance model",
        .terms = &MOS3nSize,
        .numNames = &MOS3nSize,
        .termNames = MOS3names,
        .numInstanceParms = &MOS3pTSize,
        .instanceParms = MOS3pTable,
        .numModelParms = &MOS3mPTSize,
        .modelParms = MOS3mPTable,
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

    .DEVparam = MOS3param,
    .DEVmodParam = MOS3mParam,
    .DEVload = MOS3load,
    .DEVsetup = MOS3setup,
    .DEVunsetup = MOS3unsetup,
    .DEVpzSetup = MOS3setup,
    .DEVtemperature = MOS3temp,
    .DEVtrunc = MOS3trunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = MOS3acLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = MOS3delete,
    .DEVsetic = MOS3getic,
    .DEVask = MOS3ask,
    .DEVmodAsk = MOS3mAsk,
    .DEVpzLoad = MOS3pzLoad,
    .DEVconvTest = MOS3convTest,
    .DEVsenSetup = MOS3sSetup,
    .DEVsenLoad = MOS3sLoad,
    .DEVsenUpdate = MOS3sUpdate,
    .DEVsenAcLoad = MOS3sAcLoad,
    .DEVsenPrint = MOS3sPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = MOS3disto,
    .DEVnoise = MOS3noise,
    .DEVsoaCheck = NULL,
    .DEVinstSize = &MOS3iSize,
    .DEVmodSize = &MOS3mSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_mos3_info(void)
{
    return &MOS3info;
}
