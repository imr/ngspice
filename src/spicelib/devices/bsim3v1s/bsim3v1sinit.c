#include <config.h>

#include <devdefs.h>

#include "bsim3v1sitf.h"
#include "bsim3v1sext.h"
#include "bsim3v1sinit.h"


SPICEdev BSIM3v1Sinfo = {
    {
        "BSIM3v1S",
        "Berkeley Short Channel IGFET Model Version-3 (3.1 Serban)",

        &BSIM3v1SnSize,
        &BSIM3v1SnSize,
        BSIM3v1Snames,

        &BSIM3v1SpTSize,
        BSIM3v1SpTable,

        &BSIM3v1SmPTSize,
        BSIM3v1SmPTable,

#ifdef XSPICE
        /*
         * OH what a hack this is!!! I have no idea what the proper values
         * should be so I am just going to zero it out! This is a heck of a
         * lot better than what existed perviously which was to convert
         * DEV_DEFAULT to a function pointer. Would have started executing
         * data at that point. Gotta love it!!!
         */
        NULL,

        0,
        NULL,

        0,
        NULL,

        0,
        NULL,
#endif

	DEV_DEFAULT,
    },

    DEVparam      : BSIM3v1Sparam,
    DEVmodParam   : BSIM3v1SmParam,
    DEVload       : BSIM3v1Sload,
    DEVsetup      : BSIM3v1Ssetup,
    DEVunsetup    : BSIM3v1Sunsetup,
    DEVpzSetup    : BSIM3v1Ssetup,
    DEVtemperature: BSIM3v1Stemp,
    DEVtrunc      : BSIM3v1Strunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM3v1SacLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM3v1Sdestroy,
    DEVmodDelete  : BSIM3v1SmDelete,
    DEVdelete     : BSIM3v1Sdelete, 
    DEVsetic      : BSIM3v1Sgetic,
    DEVask        : BSIM3v1Sask,
    DEVmodAsk     : BSIM3v1SmAsk,
    DEVpzLoad     : BSIM3v1SpzLoad,
    DEVconvTest   : BSIM3v1SconvTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : BSIM3v1Snoise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                    
    DEVinstSize   : &BSIM3v1SiSize,
    DEVmodSize    : &BSIM3v1SmSize

};


SPICEdev *
get_bsim3v1s_info(void)
{
    return &BSIM3v1Sinfo;
}
