/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2014 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 2  SUBVERSION : 2  REVISION : 0 ) 
 Model Parameter 'VERSION' : 2.20
 FILE : hisimhv.h

 DATE : 2014.6.11

 released by
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

/**********************************************************************

The following source code, and all copyrights, trade secrets or other
intellectual property rights in and to the source code in its entirety,
is owned by the Hiroshima University and the STARC organization.

All users need to follow the "HISIM_HV Distribution Statement and
Copyright Notice" attached to HiSIM_HV model.

-----HISIM_HV Distribution Statement and Copyright Notice--------------

Software is distributed as is, completely without warranty or service
support. Hiroshima University or STARC and its employees are not liable
for the condition or performance of the software.

Hiroshima University and STARC own the copyright and grant users a perpetual,
irrevocable, worldwide, non-exclusive, royalty-free license with respect 
to the software as set forth below.   

Hiroshima University and STARC hereby disclaims all implied warranties.

Hiroshima University and STARC grant the users the right to modify, copy,
and redistribute the software and documentation, both within the user's
organization and externally, subject to the following restrictions

1. The users agree not to charge for Hiroshima University and STARC code
itself but may charge for additions, extensions, or support.

2. In any product based on the software, the users agree to acknowledge
Hiroshima University and STARC that developed the software. This
acknowledgment shall appear in the product documentation.

3. The users agree to reproduce any copyright notice which appears on
the software on any copy or modification of such made available
to others."

Toshimasa Asahara, President, Hiroshima University
Mitiko Miura-Mattausch, Professor, Hiroshima University
Katsuhiro Shimohigashi, President&CEO, STARC
June 2008 (revised October 2011) 
*************************************************************************/

#include "hsmhv2def.h"
#include "ngspice/cktdefs.h"

#ifndef _HiSIMHV2_H
#define _HiSIMHV2_H

/* return value */
#ifndef OK
#define HiSIM_OK        0
#define HiSIM_ERROR     1
#else
#define HiSIM_OK        OK
#define HiSIM_ERROR     E_PANIC
#endif

/* MOS type */
#ifndef NMOS
#define NMOS     1
#define PMOS    -1
#endif

/* device working mode */
#ifndef CMI_NORMAL_MODE
#define HiSIM_NORMAL_MODE    1
#define HiSIM_REVERSE_MODE  -1
#else
#define HiSIM_NORMAL_MODE  CMI_NORMAL_MODE
#define HiSIM_REVERSE_MODE CMI_REVERSE_MODE
#endif

/* others */
#ifndef NULL
#define NULL            0
#endif

#define HiSIM_FALSE     0
#define HiSIM_TRUE      1

#ifndef return_if_error
#define return_if_error(s) { int error = s; if(error) return(error); }
#endif

extern int HSMHV2evaluate
(
 double ivds,
 double ivgs,
 double ivbs,
 double ivdsi,
 double ivgsi,
 double ivbsi,
 double vbs_jct,
 double vbd_jct,
 double vsubs,
 double vddp,
 double deltemp,
 HSMHV2instance *here,
 HSMHV2model    *model,
 CKTcircuit   *ckt
 ) ;
extern int HSMHV2rdrift
( 
 double vddp,
 double ivds,
 double ivbs,
 double vsubs,
 double deltemp,
 HSMHV2instance *here,
 HSMHV2model    *model,
 CKTcircuit   *ckt
 ) ;
extern int HSMHV2dio
(
 double        vbs_jct,
 double        vbd_jct,
 double        deltemp,
 HSMHV2instance *here,
 HSMHV2model    *model,
 CKTcircuit   *ckt
 ) ;

#endif /* _HiSIMHV2_H */
