#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "bsim3v32itf.h"
#include "bsim3v32ext.h"
#include "bsim3v32init.h"


SPICEdev BSIM3v32info = {
    {   "BSIM3v32",
        "Berkeley Short Channel IGFET Model Version-3",

        &BSIM3v32nSize,
        &BSIM3v32nSize,
        BSIM3v32names,

        &BSIM3v32pTSize,
        BSIM3v32pTable,

        &BSIM3v32mPTSize,
        BSIM3v32mPTable,

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

 /* DEVparam      */ BSIM3v32param,
 /* DEVmodParam   */ BSIM3v32mParam,
 /* DEVload       */ BSIM3v32load,
 /* DEVsetup      */ BSIM3v32setup,
 /* DEVunsetup    */ BSIM3v32unsetup,
 /* DEVpzSetup    */ BSIM3v32setup,
 /* DEVtemperature*/ BSIM3v32temp,
 /* DEVtrunc      */ BSIM3v32trunc,
 /* DEVfindBranch */ NULL,
 /* DEVacLoad     */ BSIM3v32acLoad,
 /* DEVaccept     */ NULL,
 /* DEVdestroy    */ BSIM3v32destroy,
 /* DEVmodDelete  */ BSIM3v32mDelete,
 /* DEVdelete     */ BSIM3v32delete, 
 /* DEVsetic      */ BSIM3v32getic,
 /* DEVask        */ BSIM3v32ask,
 /* DEVmodAsk     */ BSIM3v32mAsk,
 /* DEVpzLoad     */ BSIM3v32pzLoad,
 /* DEVconvTest   */ BSIM3v32convTest,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL,
 /* DEVnoise      */ BSIM3v32noise,
#ifdef CIDER
 /* DEVdump       */ NULL,
 /* DEVacct       */ NULL,
#endif                       
 /* DEVinstSize   */ &BSIM3v32iSize,
 /* DEVmodSize    */ &BSIM3v32mSize,
#ifdef KLU
 /* DEVbindCSC        */   BSIM3v32bindCSC,
 /* DEVbindCSCComplex */   BSIM3v32bindCSCComplex,
#endif

};


SPICEdev *
get_bsim3v32_info(void)
{
    return &BSIM3v32info;
}
