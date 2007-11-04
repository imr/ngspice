#include <config.h>

#include <devdefs.h>

#include "bsim4v3itf.h"
#include "bsim4v3ext.h"
#include "bsim4v3init.h"


SPICEdev BSIM4v3info = {
    {
        "BSIM4v3",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4v3nSize,
        &BSIM4v3nSize,
        BSIM4v3names,

        &BSIM4v3pTSize,
        BSIM4v3pTable,

        &BSIM4v3mPTSize,
        BSIM4v3mPTable,

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

    DEVparam      : BSIM4v3param,
    DEVmodParam   : BSIM4v3mParam,
    DEVload       : BSIM4v3load,
    DEVsetup      : BSIM4v3setup,
    DEVunsetup    : BSIM4v3unsetup,
    DEVpzSetup    : BSIM4v3setup,
    DEVtemperature: BSIM4v3temp,
    DEVtrunc      : BSIM4v3trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : BSIM4v3acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : BSIM4v3destroy,
    DEVmodDelete  : BSIM4v3mDelete,
    DEVdelete     : BSIM4v3delete, 
    DEVsetic      : BSIM4v3getic,
    DEVask        : BSIM4v3ask,
    DEVmodAsk     : BSIM4v3mAsk,
    DEVpzLoad     : BSIM4v3pzLoad,
    DEVconvTest   : BSIM4v3convTest,
    DEVsenSetup   : NULL,
    DEVsenLoad    : NULL,
    DEVsenUpdate  : NULL,
    DEVsenAcLoad  : NULL,
    DEVsenPrint   : NULL,
    DEVsenTrunc   : NULL,
    DEVdisto      : NULL,
    DEVnoise      : BSIM4v3noise,
                    
    DEVinstSize   : &BSIM4v3iSize,
    DEVmodSize    : &BSIM4v3mSize
};


SPICEdev *
get_bsim4v3_info(void)
{
    return &BSIM4v3info;
}
