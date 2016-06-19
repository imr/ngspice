#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hsmhv2def.h"
#include "hsmhv2itf.h"
#include "hsmhv2init.h"

SPICEdev HSMHV2info = {
  {   "HiSIMHV2",
      "Hiroshima University STARC IGFET Model - HiSIM_HV v.2",
      
      &HSMHV2nSize,
      &HSMHV2nSize,
      HSMHV2names,

      &HSMHV2pTSize,
      HSMHV2pTable,
      
      &HSMHV2mPTSize,
      HSMHV2mPTable,

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

 /* DEVparam      */ HSMHV2param,
 /* DEVmodParam   */ HSMHV2mParam,
 /* DEVload       */ HSMHV2load,
 /* DEVsetup      */ HSMHV2setup,
 /* DEVunsetup    */ HSMHV2unsetup,
 /* DEVpzSetup    */ HSMHV2setup,
 /* DEVtemperature*/ HSMHV2temp,
 /* DEVtrunc      */ HSMHV2trunc,
 /* DEVfindBranch */ NULL,
 /* DEVacLoad     */ HSMHV2acLoad,
 /* DEVaccept     */ NULL,
 /* DEVdestroy    */ HSMHV2destroy,
 /* DEVmodDelete  */ HSMHV2mDelete,
 /* DEVdelete     */ HSMHV2delete, 
 /* DEVsetic      */ HSMHV2getic,
 /* DEVask        */ HSMHV2ask,
 /* DEVmodAsk     */ HSMHV2mAsk,
 /* DEVpzLoad     */ HSMHV2pzLoad,
 /* DEVconvTest   */ HSMHV2convTest,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL,
 /* DEVnoise      */ HSMHV2noise,
 /* DEVsoaCheck   */ HSMHV2soaCheck,
#ifdef CIDER
 /* DEVdump       */ NULL,
 /* DEVacct       */ NULL,
#endif
 /* DEVinstSize   */ &HSMHV2iSize,
 /* DEVmodSize    */ &HSMHV2mSize,

#ifdef KLU
 /* DEVbindCSC        */       HSMHV2bindCSC,
 /* DEVbindCSCComplex */       HSMHV2bindCSCComplex,
 /* DEVbindCSCComplexToReal */ HSMHV2bindCSCComplexToReal,
#endif

};


SPICEdev *
get_hsmhv2_info(void)
{
    return &HSMHV2info;
}
