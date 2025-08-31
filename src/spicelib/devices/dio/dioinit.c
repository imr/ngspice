#include "ngspice/config.h"

#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"

#include "diodefs.h"
#include "dioitf.h"
#include "dioinit.h"


SPICEdev DIOinfo = {
    .DEVpublic = {
        .name = "Diode",
        .description = "Junction Diode model",
        .terms = &DIOnSize,
        .numNames = &DIOnSize,
        .termNames = DIOnames,
        .numInstanceParms = &DIOpTSize,
        .instanceParms = DIOpTable,
        .numModelParms = &DIOmPTSize,
        .modelParms = DIOmPTable,
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

    .DEVparam = DIOparam,
    .DEVmodParam = DIOmParam,
    .DEVload = DIOload,
    .DEVsetup = DIOsetup,
    .DEVunsetup = DIOunsetup,
    .DEVpzSetup = DIOsetup,
    .DEVtemperature = DIOtemp,
    .DEVtrunc = DIOtrunc,
    .DEVfindBranch = NULL,
    .DEVacLoad = DIOacLoad,
    .DEVaccept = NULL,
    .DEVdestroy = NULL,
    .DEVmodDelete = NULL,
    .DEVdelete = NULL,
    .DEVsetic = DIOgetic,
    .DEVask = DIOask,
    .DEVmodAsk = DIOmAsk,
    .DEVpzLoad = DIOpzLoad,
    .DEVconvTest = DIOconvTest,
    .DEVsenSetup = DIOsSetup,
    .DEVsenLoad = DIOsLoad,
    .DEVsenUpdate = DIOsUpdate,
    .DEVsenAcLoad = DIOsAcLoad,
    .DEVsenPrint = DIOsPrint,
    .DEVsenTrunc = NULL,
    .DEVdisto = DIOdisto,
    .DEVnoise = DIOnoise,
    .DEVsoaCheck = DIOsoaCheck,
    .DEVinstSize = &DIOiSize,
    .DEVmodSize = &DIOmSize,

#ifdef CIDER
    .DEVdump = NULL,
    .DEVacct = NULL,
#endif
};


SPICEdev *
get_dio_info(void)
{
    return &DIOinfo;
}
