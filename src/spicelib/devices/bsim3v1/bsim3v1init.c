#include <config.h>

#include <devdefs.h>

#include "bsim3v1itf.h"
#include "bsim3v1ext.h"
#include "bsim3v1init.h"


SPICEdev BSIM3V1info = {
    {
        "BSIM3V1",
        "Berkeley Short Channel IGFET Model Version-3 (3v3.1)",

        &BSIM3V1nSize,
        &BSIM3V1nSize,
        BSIM3V1names,

        &BSIM3V1pTSize,
        BSIM3V1pTable,

        &BSIM3V1mPTSize,
        BSIM3V1mPTable,

#ifdef XSPICE
/*----  Fixed by SDB 5.2.2003 to enable XSPICE/tclspice integration  -----*/
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */

        0,     /* This is a SPICE device, it has no MIF info data */
        NULL,  /* This is a SPICE device, it has no MIF info data */
/*---------------------------  End of SDB fix   -------------------------*/
#endif

	DEV_DEFAULT,

    },

    DEVparam      : BSIM3V1param,
    DEVmodParam   : BSIM3V1mParam,
    DEVload       : BSIM3V1load,
    DEVsetup      : BSIM3V1setup,
    DEVunsetup    : NULL,
    DEVpzSetup    : BSIM3V1setup,
    DEVtemperature: BSIM3V1temp,
    DEVtrunc      : BSIM3V1trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM3V1acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM3V1destroy,
    DEVmodDelete  : BSIM3V1mDelete,
    DEVdelete     : BSIM3V1delete, 
    DEVsetic      : BSIM3V1getic,
    DEVask        : BSIM3V1ask,
    DEVmodAsk     : BSIM3V1mAsk,
    DEVpzLoad     : BSIM3V1pzLoad,
    DEVconvTest   : BSIM3V1convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : BSIM3V1noise,
                    
    DEVinstSize   : &BSIM3V1iSize,
    DEVmodSize    : &BSIM3V1mSize

};


SPICEdev *
get_bsim3v1_info(void)
{
    return &BSIM3V1info;
}
