#include <config.h>

#include <devdefs.h>

#include "mesitf.h"
#include "mesext.h"
#include "mesinit.h"


SPICEdev MESinfo = {
    {
        "MES",
        "GaAs MESFET model",

        &MESnSize,
        &MESnSize,
        MESnames,

        &MESpTSize,
        MESpTable,

        &MESmPTSize,
        MESmPTable,
	DEV_DEFAULT
    },

    DEVparam      : MESparam,
    DEVmodParam   : MESmParam,
    DEVload       : MESload,
    DEVsetup      : MESsetup,
    DEVunsetup    : MESunsetup,
    DEVpzSetup    : MESsetup,
    DEVtemperature: MEStemp,
    DEVtrunc      : MEStrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : MESacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MESdestroy,
    DEVmodDelete  : MESmDelete,
    DEVdelete     : MESdelete,
    DEVsetic      : MESgetic,
    DEVask        : MESask,
    DEVmodAsk     : MESmAsk,
    DEVpzLoad     : MESpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : MESdisto,
    DEVnoise      : MESnoise,
                    
    DEVinstSize   : &MESiSize,
    DEVmodSize    : &MESmSize

};


SPICEdev *
get_mes_info(void)
{
    return &MESinfo;
}
