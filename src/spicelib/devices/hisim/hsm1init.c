#include <config.h>

#include <devdefs.h>

#include "hsm1def.h"
#include "hsm1itf.h"
#include "hsm1ext.h"
#include "hsm1init.h"

SPICEdev HSM1info = {
  {   "HiSIM1",
      "Hiroshima University STARC IGFET Model 1.2.0",
      
      &HSM1nSize,
      &HSM1nSize,
      HSM1names,

      &HSM1pTSize,
      HSM1pTable,
      
      &HSM1mPTSize,
      HSM1mPTable,

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

  DEVparam      : HSM1param,
  DEVmodParam   : HSM1mParam,
  DEVload       : HSM1load,
  DEVsetup      : HSM1setup,
  DEVunsetup    : HSM1unsetup,
  DEVpzSetup    : HSM1setup,
  DEVtemperature: HSM1temp,
  DEVtrunc      : HSM1trunc,
  DEVfindBranch : NULL,
  DEVacLoad     : HSM1acLoad,
  DEVaccept     : NULL,
  DEVdestroy    : HSM1destroy,
  DEVmodDelete  : HSM1mDelete,
  DEVdelete     : HSM1delete, 
  DEVsetic      : HSM1getic,
  DEVask        : HSM1ask,
  DEVmodAsk     : HSM1mAsk,
  DEVpzLoad     : HSM1pzLoad,
  DEVconvTest   : HSM1convTest,
  DEVsenSetup   : NULL,
  DEVsenLoad    : NULL,
  DEVsenUpdate  : NULL,
  DEVsenAcLoad  : NULL,
  DEVsenPrint   : NULL,
  DEVsenTrunc   : NULL,
  DEVdisto      : NULL,

  DEVnoise      : HSM1noise,
#ifdef CIDER
    DEVdump       : NULL,
    DEVacct       : NULL,
#endif
  DEVinstSize   : &HSM1iSize,
  DEVmodSize    : &HSM1mSize

};


SPICEdev *
get_hsm1_info(void)
{
    return &HSM1info;
}
