/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int HFET2temp(
GENmodel *inModel,
CKTcircuit *ckt)
{

  HFET2instance *here;
  HFET2model *model = (HFET2model*)inModel;
  double vt;
  double tdiff;

  for( ; model != NULL; model = HFET2nextModel(model)) {
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
    for (here = HFET2instances(model); here != NULL; 
         here=HFET2nextInstance(here)) {

    if(!here->HFET2dtempGiven)
       here->HFET2dtemp = 0.0;
    if(!here->HFET2tempGiven)
       TEMP = ckt->CKTtemp + here->HFET2dtemp;

      vt    = CONSTKoverQ*TEMP;
      tdiff = TEMP - ckt->CKTnomTemp;

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
