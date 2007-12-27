#include "config.h"

#include "devdefs.h"

#include "bsim4v2itf.h"
#include "bsim4v2ext.h"
#include "bsim4v2init.h"


SPICEdev BSIM4v2info = {
    {
        "BSIM4v2",
        "Berkeley Short Channel IGFET Model-4",

        &BSIM4v2nSize,
        &BSIM4v2nSize,
        BSIM4v2names,

        &BSIM4v2pTSize,
        BSIM4v2pTable,

        &BSIM4v2mPTSize,
        BSIM4v2mPTable,

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

 /* DEVparam      */ BSIM4v2param,
 /* DEVmodParam   */ BSIM4v2mParam,
 /* DEVload       */ BSIM4v2load,
 /* DEVsetup      */ BSIM4v2setup,
 /* DEVunsetup    */ BSIM4v2unsetup,
 /* DEVpzSetup    */ BSIM4v2setup,
 /* DEVtemperature*/ BSIM4v2temp,
 /* DEVtrunc      */ BSIM4v2trunc,
 /* DEVfindBranch */ NULL,
 /* DEVacLoad     */ BSIM4v2acLoad,
 /* DEVaccept     */ NULL,
 /* DEVdestroy    */ BSIM4v2destroy,
 /* DEVmodDelete  */ BSIM4v2mDelete,
 /* DEVdelete     */ BSIM4v2delete, 
 /* DEVsetic      */ BSIM4v2getic,
 /* DEVask        */ BSIM4v2ask,
 /* DEVmodAsk     */ BSIM4v2mAsk,
 /* DEVpzLoad     */ BSIM4v2pzLoad,
 /* DEVconvTest   */ BSIM4v2convTest,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL,
 /* DEVnoise      */ BSIM4v2noise,
                    
 /* DEVinstSize   */ &BSIM4v2iSize,
 /* DEVmodSize    */ &BSIM4v2mSize
};


SPICEdev *
get_bsim4v2_info(void)
{
    return &BSIM4v2info;
}
