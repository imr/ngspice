#include <config.h>

#include <devdefs.h>

#include "mos3itf.h"
#include "mos3ext.h"
#include "mos3init.h"


SPICEdev MOS3info = {
    {
        "Mos3",
        "Level 3 MOSfet model with Meyer capacitance model",

        &MOS3nSize,
        &MOS3nSize,
        MOS3names,

        &MOS3pTSize,
        MOS3pTable,

        &MOS3mPTSize,
        MOS3mPTable,

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

    DEVparam      : MOS3param,
    DEVmodParam   : MOS3mParam,
    DEVload       : MOS3load,
    DEVsetup      : MOS3setup,
    DEVunsetup    : MOS3unsetup,
    DEVpzSetup    : MOS3setup,
    DEVtemperature: MOS3temp,
    DEVtrunc      : MOS3trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : MOS3acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MOS3destroy,
    DEVmodDelete  : MOS3mDelete,
    DEVdelete     : MOS3delete,
    DEVsetic      : MOS3getic,
    DEVask        : MOS3ask,
    DEVmodAsk     : MOS3mAsk,
    DEVpzLoad     : MOS3pzLoad,
    DEVconvTest   : MOS3convTest,
    DEVsenSetup   : MOS3sSetup,
    DEVsenLoad    : MOS3sLoad,
    DEVsenUpdate  : MOS3sUpdate,
    DEVsenAcLoad  : MOS3sAcLoad,
    DEVsenPrint   : MOS3sPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : MOS3disto,
    DEVnoise      : MOS3noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                         
    DEVinstSize   : &MOS3iSize,
    DEVmodSize    : &MOS3mSize

};


SPICEdev *
get_mos3_info(void)
{
    return &MOS3info;
}
