#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hsm2def.h"
#include "hsm2itf.h"
#include "hsm2init.h"

SPICEdev HSM2info = {
  {   "HiSIM2",
      "Hiroshima University STARC IGFET Model 2.7.0",
      
      &HSM2nSize,
      &HSM2nSize,
      HSM2names,

      &HSM2pTSize,
      HSM2pTable,
      
      &HSM2mPTSize,
      HSM2mPTable,

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

 /* DEVparam      */ HSM2param,
 /* DEVmodParam   */ HSM2mParam,
 /* DEVload       */ HSM2load,
 /* DEVsetup      */ HSM2setup,
 /* DEVunsetup    */ HSM2unsetup,
 /* DEVpzSetup    */ HSM2setup,
 /* DEVtemperature*/ HSM2temp,
 /* DEVtrunc      */ HSM2trunc,
 /* DEVfindBranch */ NULL,
 /* DEVacLoad     */ HSM2acLoad,
 /* DEVaccept     */ NULL,
 /* DEVdestroy    */ HSM2destroy,
 /* DEVmodDelete  */ HSM2mDelete,
 /* DEVdelete     */ HSM2delete, 
 /* DEVsetic      */ HSM2getic,
 /* DEVask        */ HSM2ask,
 /* DEVmodAsk     */ HSM2mAsk,
 /* DEVpzLoad     */ HSM2pzLoad,
 /* DEVconvTest   */ HSM2convTest,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL,
 /* DEVnoise      */ HSM2noise,
#ifdef CIDER
 /* DEVdump       */ NULL,
 /* DEVacct       */ NULL,
#endif
 /* DEVinstSize   */ &HSM2iSize,
 /* DEVmodSize    */ &HSM2mSize,
 /* DEVnodeIsNonLinear */ HSM2nodeIsNonLinear

};


SPICEdev *
get_hsm2_info(void)
{
    return &HSM2info;
}
