#include <config.h>

#include <devdefs.h>

#include "bsim4itf.h"
#include "bsim4ext.h"
#include "bsim4init.h"


SPICEdev BSIM4info = {
    {
        "BSIM4",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4nSize,
        &BSIM4nSize,
        BSIM4names,

        &BSIM4pTSize,
        BSIM4pTable,

        &BSIM4mPTSize,
        BSIM4mPTable,

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

        DEV_DEFAULT
    },

    DEVparam      : BSIM4param,
    DEVmodParam   : BSIM4mParam,
    DEVload       : BSIM4load,
    DEVsetup      : BSIM4setup,
    DEVunsetup    : BSIM4unsetup,
    DEVpzSetup    : BSIM4setup,
    DEVtemperature: BSIM4temp,
    DEVtrunc      : BSIM4trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM4acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM4destroy,
    DEVmodDelete  : BSIM4mDelete,
    DEVdelete     : BSIM4delete, 
    DEVsetic      : BSIM4getic,
    DEVask        : BSIM4ask,
    DEVmodAsk     : BSIM4mAsk,
    DEVpzLoad     : BSIM4pzLoad,
    DEVconvTest   : BSIM4convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : BSIM4noise,
                    
    DEVinstSize   : &BSIM4iSize,
    DEVmodSize    : &BSIM4mSize
};


SPICEdev *
get_bsim4_info(void)
{
    return &BSIM4info;
}
