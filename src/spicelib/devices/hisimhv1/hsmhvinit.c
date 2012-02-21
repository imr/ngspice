#include "ngspice/config.h"

#include "ngspice/devdefs.h"

#include "hsmhvdef.h"
#include "hsmhvitf.h"
#include "hsmhvinit.h"

SPICEdev HSMHVinfo = {
  {   "HiSIMHV",
      "Hiroshima University STARC IGFET Model - HiSIM_HV",
      
      &HSMHVnSize,
      &HSMHVnSize,
      HSMHVnames,

      &HSMHVpTSize,
      HSMHVpTable,
      
      &HSMHVmPTSize,
      HSMHVmPTable,

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

 /* DEVparam      */ HSMHVparam,
 /* DEVmodParam   */ HSMHVmParam,
 /* DEVload       */ HSMHVload,
 /* DEVsetup      */ HSMHVsetup,
 /* DEVunsetup    */ HSMHVunsetup,
 /* DEVpzSetup    */ HSMHVsetup,
 /* DEVtemperature*/ HSMHVtemp,
 /* DEVtrunc      */ HSMHVtrunc,
 /* DEVfindBranch */ NULL,
 /* DEVacLoad     */ HSMHVacLoad,
 /* DEVaccept     */ NULL,
 /* DEVdestroy    */ HSMHVdestroy,
 /* DEVmodDelete  */ HSMHVmDelete,
 /* DEVdelete     */ HSMHVdelete, 
 /* DEVsetic      */ HSMHVgetic,
 /* DEVask        */ HSMHVask,
 /* DEVmodAsk     */ HSMHVmAsk,
 /* DEVpzLoad     */ HSMHVpzLoad,
 /* DEVconvTest   */ HSMHVconvTest,
 /* DEVsenSetup   */ NULL,
 /* DEVsenLoad    */ NULL,
 /* DEVsenUpdate  */ NULL,
 /* DEVsenAcLoad  */ NULL,
 /* DEVsenPrint   */ NULL,
 /* DEVsenTrunc   */ NULL,
 /* DEVdisto      */ NULL,
 /* DEVnoise      */ HSMHVnoise,
#ifdef CIDER
 /* DEVdump       */ NULL,
 /* DEVacct       */ NULL,
#endif
 /* DEVinstSize   */ &HSMHViSize,
 /* DEVmodSize    */ &HSMHVmSize,
#ifdef KLU
 /* DEVbindCSC        */   HSMHVbindCSC,
 /* DEVbindCSCComplex */   HSMHVbindCSCComplex,
#endif

};


SPICEdev *
get_hsmhv_info(void)
{
    return &HSMHVinfo;
}
