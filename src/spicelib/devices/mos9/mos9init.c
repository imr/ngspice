#include <config.h>

#include <devdefs.h>

#include "mos9itf.h"
#include "mos9ext.h"
#include "mos9init.h"


SPICEdev MOS9info = {
    {
        "Mos9",
        "Modified Level 3 MOSfet model",

        &MOS9nSize,
        &MOS9nSize,
        MOS9names,

        &MOS9pTSize,
        MOS9pTable,

        &MOS9mPTSize,
        MOS9mPTable,

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

    DEVparam      : MOS9param,
    DEVmodParam   : MOS9mParam,
    DEVload       : MOS9load,
    DEVsetup      : MOS9setup,
    DEVunsetup    : MOS9unsetup,
    DEVpzSetup    : MOS9setup,
    DEVtemperature: MOS9temp,
    DEVtrunc      : MOS9trunc,
    DEVfindBranch : NULL,
    DEVacLoad     : MOS9acLoad,
    DEVaccept     : NULL,
    DEVdestroy    : MOS9destroy,
    DEVmodDelete  : MOS9mDelete,
    DEVdelete     : MOS9delete,
    DEVsetic      : MOS9getic,
    DEVask        : MOS9ask,
    DEVmodAsk     : MOS9mAsk,
    DEVpzLoad     : MOS9pzLoad,
    DEVconvTest   : MOS9convTest,
    DEVsenSetup   : MOS9sSetup,
    DEVsenLoad    : MOS9sLoad,
    DEVsenUpdate  : MOS9sUpdate,
    DEVsenAcLoad  : MOS9sAcLoad,
    DEVsenPrint   : MOS9sPrint,
    DEVsenTrunc   : NULL,
    DEVdisto      : MOS9disto,
    DEVnoise      : MOS9noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif                        
    DEVinstSize   : &MOS9iSize,
    DEVmodSize    : &MOS9mSize

};


SPICEdev *
get_mos9_info(void)
{
    return &MOS9info;
}
