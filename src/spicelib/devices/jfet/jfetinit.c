#include <config.h>

#include <devdefs.h>

#include "jfetitf.h"
#include "jfetext.h"
#include "jfetinit.h"


SPICEdev JFETinfo = {
    {
        "JFET",
        "Junction Field effect transistor",

        &JFETnSize,
        &JFETnSize,
        JFETnames,

        &JFETpTSize,
        JFETpTable,

        &JFETmPTSize,
        JFETmPTable,
	DEV_DEFAULT
    },

    DEVparam      : JFETparam,
    DEVmodParam   : JFETmParam,
    DEVload       : JFETload,
    DEVsetup      : JFETsetup,
    DEVunsetup    : JFETunsetup,
    DEVpzSetup    : JFETsetup,
    DEVtemperature: JFETtemp,
    DEVtrunc      : JFETtrunc,
    DEVfindBranch : NULL,
    DEVacLoad     : JFETacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : JFETdestroy,
    DEVmodDelete  : JFETmDelete,
    DEVdelete     : JFETdelete,
    DEVsetic      : JFETgetic,
    DEVask        : JFETask,
    DEVmodAsk     : JFETmAsk,
    DEVpzLoad     : JFETpzLoad,
    DEVconvTest   : NULL,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : JFETdisto,
    DEVnoise      : JFETnoise,
                    
    DEVinstSize   : &JFETiSize,
    DEVmodSize    : &JFETmSize

};


SPICEdev *
get_jfet_info(void)
{
    return &JFETinfo;
}
