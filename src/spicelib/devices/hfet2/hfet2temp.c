
#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "hfet2defs.h"
#include "const.h"
#include "sperror.h"
#include "suffix.h"


int HFET2temp(inModel, ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{

  HFET2instance *here;
  HFET2model *model = (HFET2model*)inModel;

  for( ; model != NULL; model = model->HFET2nextModel ) {
    if(model->HFET2rd != 0)
      model->HFET2drainConduct = 1/model->HFET2rd;
    else
      model->HFET2drainConduct = 0;
    if(model->HFET2rs != 0)
      model->HFET2sourceConduct = 1/model->HFET2rs;
    else
      model->HFET2sourceConduct = 0;
    if(!model->HFET2vt1Given)
      HFET2_VT1 = VTO+CHARGE*NMAX*DI/EPSI;
    if(!model->HFET2vt2Given)
      VT2 = VTO;
    DELTA2 = DELTA*DELTA;
    for (here = model->HFET2instances; here != NULL; here=here->HFET2nextInstance) {
      double vt    = CONSTKoverQ*TEMP;
      double tdiff = TEMP - ckt->CKTnomTemp;
      TLAMBDA = LAMBDA + KLAMBDA*tdiff;
      TMU     = MU - KMU*tdiff;
      TNMAX   = NMAX - KNMAX*tdiff;
      TVTO    = TYPE*VTO - KVTO*tdiff;
      JSLW    = JS*L*W/2;
      GGRLW   = GGR*L*W/2;
      N0      = EPSI*ETA*vt/2/CHARGE/(DI+DELTAD);
      N01     = EPSI*ETA1*vt/2/CHARGE/D1;
      if(model->HFET2eta2Given)
        N02 = EPSI*ETA2*vt/2/CHARGE/D2;
      else
        N02 = 0.0;  
      GCHI0 = CHARGE*W*TMU/L;
      IMAX  = CHARGE*TNMAX*VS*W;
      VCRIT = vt*log(vt/(CONSTroot2 * 1e-11));
    }
  }
  return(OK);
  
}
