/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2011 Hiroshima University & STARC

 VERSION : HiSIM_2.5.1 
 FILE : hsm2itf.h

 date : 2011.04.07

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#ifdef DEV_hisim2

#ifndef DEV_HISIM2
#define DEV_HISIM2

#include "hsm2ext.h"

extern IFparm HSM2pTable[ ];
extern IFparm HSM2mPTable[ ];
extern char *HSM2names[ ];
extern int HSM2pTSize;
extern int HSM2mPTSize;
extern int HSM2nSize;
extern int HSM2iSize;
extern int HSM2mSize;

SPICEdev HSM2info = {
  {   "HiSIM2",
      "Hiroshima University STARC IGFET Model 2.5.1",
      
      &HSM2nSize,
      &HSM2nSize,
      HSM2names,

      &HSM2pTSize,
      HSM2pTable,
      
      &HSM2mPTSize,
      HSM2mPTable,

  },

  HSM2param,
  HSM2mParam,
  HSM2load,
  HSM2setup,
  NULL,
  HSM2setup,
  HSM2temp,
  HSM2trunc,
  NULL,
  HSM2acLoad,
  NULL,
  HSM2destroy,
#ifdef DELETES
  HSM2mDelete,
  HSM2delete, 
#else /* DELETES */
  NULL,
  NULL,
#endif /* DELETES */
  HSM2getic,
  HSM2ask,
  HSM2mAsk,
#ifdef AN_pz
  HSM2pzLoad,
#else /* AN_pz */
  NULL,
#endif /* AN_pz */
#ifdef NEWCONV
  HSM2convTest,
#else /* NEWCONV */
  NULL,
#endif /* NEWCONV */
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,

#ifdef AN_noise
  HSM2noise,
#else /* AN_noise */
  NULL,
#endif /* AN_noise */

  &HSM2iSize,
  &HSM2mSize

};

#endif
#endif
