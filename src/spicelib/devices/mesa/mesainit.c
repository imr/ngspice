#include <config.h>

#include <devdefs.h>

#include "mesaitf.h"
#include "mesaext.h"
#include "mesainit.h"


SPICEdev MESAinfo = {
    {
        "MESA",
        "GaAs MESFET model",

        &MESAnSize,
        &MESAnSize,
        MESAnames,

        &MESApTSize,
        MESApTable,

        &MESAmPTSize,
        MESAmPTable,
	DEV_DEFAULT
    },

    DEVparam      : MESAparam,
    DEVmodParam   : MESAmParam,
    DEVload       : MESAload,
    DEVsetup      : MESAsetup,
    DEVunsetup    : MESAunsetup,
    DEVpzSetup    : MESAsetup,
    DEVtemperature: MESAtemp,
    DEVtrunc      : MESAtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : MESAacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MESAdestroy,
    DEVmodDelete  : MESAmDelete,
    DEVdelete     : MESAdelete,
    DEVsetic      : MESAgetic,
    DEVask        : MESAask,
    DEVmodAsk     : MESAmAsk,
    DEVpzLoad     : NULL,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : NULL,
                    
    DEVinstSize   : &MESAiSize,
    DEVmodSize    : &MESAmSize

};


SPICEdev *
get_mesa_info(void)
{
    return &MESAinfo;
}
