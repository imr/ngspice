#include <config.h>

#include <devdefs.h>

#include "bjtitf.h"
#include "bjtext.h"
#include "bjtinit.h"


SPICEdev BJTinfo = {
    {
	"BJT",
        "Bipolar Junction Transistor",

        &BJTnSize,
        &BJTnSize,
        BJTnames,

        &BJTpTSize,
        BJTpTable,

        &BJTmPTSize,
        BJTmPTable,
	DEV_DEFAULT
    },

    DEVparam      : BJTparam,
    DEVmodParam   : BJTmParam,
    DEVload       : BJTload,
    DEVsetup      : BJTsetup,
    DEVunsetup    : BJTunsetup,
    DEVpzSetup    : BJTsetup,
    DEVtemperature: BJTtemp,
    DEVtrunc      : BJTtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BJTacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BJTdestroy,
    DEVmodDelete  : BJTmDelete,
    DEVdelete     : BJTdelete,
    DEVsetic      : BJTgetic,
    DEVask        : BJTask,
    DEVmodAsk     : BJTmAsk,
    DEVpzLoad     : BJTpzLoad,
    DEVconvTest   : BJTconvTest,
    DEVsenSetup   : BJTsSetup,
    DEVsenLoad    : BJTsLoad,
    DEVsenUpdate  : BJTsUpdate,
    DEVsenAcLoad  : BJTsAcLoad,
    DEVsenPrint   : BJTsPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : BJTdisto,
    DEVnoise      : BJTnoise,
                    
    DEVinstSize   : &BJTiSize,
    DEVmodSize    : &BJTmSize

};


SPICEdev *
get_bjt_info(void)
{
    return &BJTinfo;
}
