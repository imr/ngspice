#include <config.h>

#include <devdefs.h>

#include "bsim3v2itf.h"
#include "bsim3v2ext.h"
#include "bsim3v2init.h"


SPICEdev BSIM3V2info = {
    {   "BSIM3V2",
        "Berkeley Short Channel IGFET Model Version-3 (3v3.2)",

        &BSIM3V2nSize,
        &BSIM3V2nSize,
        BSIM3V2names,

        &BSIM3V2pTSize,
        BSIM3V2pTable,

        &BSIM3V2mPTSize,
        BSIM3V2mPTable,
		DEV_DEFAULT
    },

    DEVparam      : BSIM3V2param,
    DEVmodParam   : BSIM3V2mParam,
    DEVload       : BSIM3V2load,
    DEVsetup      : BSIM3V2setup,
    DEVunsetup    : BSIM3V2unsetup,
    DEVpzSetup    : BSIM3V2setup,
    DEVtemperature: BSIM3V2temp,
    DEVtrunc      : BSIM3V2trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM3V2acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM3V2destroy,
    DEVmodDelete  : BSIM3V2mDelete,
    DEVdelete     : BSIM3V2delete, 
    DEVsetic      : BSIM3V2getic,
    DEVask        : BSIM3V2ask,
    DEVmodAsk     : BSIM3V2mAsk,
    DEVpzLoad     : BSIM3V2pzLoad,
    DEVconvTest   : BSIM3V2convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : BSIM3V2noise,
                    
    DEVinstSize   : &BSIM3V2iSize,
    DEVmodSize    : &BSIM3V2mSize

};


SPICEdev *
get_bsim3v2_info(void)
{
    return &BSIM3V2info;
}
