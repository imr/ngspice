/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


#define EPSILONGAAS (12.244*8.85418e-12)

int
MESAtemp(GENmodel *inModel, CKTcircuit *ckt)
{

  MESAmodel *model = (MESAmodel*)inModel;
  MESAinstance *here;
  double temp;
  double vt;
  double d;
  

  for( ; model != NULL; model = MESAnextModel(model)) {
    if(!model->MESAlambdahfGiven)
      model->MESAlambdahf = model->MESAlambda;
    if(model->MESAlevel == 2)
      model->MESAvpo = CHARGE*model->MESAnd*model->MESAd*model->MESAd/
                       2/EPSILONGAAS;
    else {
      model->MESAvpou  = CHARGE*model->MESAndu*model->MESAdu*model->MESAdu/
                         2/EPSILONGAAS;
      model->MESAvpod  = CHARGE*model->MESAndelta*model->MESAth*
                        (2*model->MESAdu + model->MESAth)/2/EPSILONGAAS;
      model->MESAvpo   = model->MESAvpou+model->MESAvpod;
    }
    model->MESAdeltaSqr = model->MESAdelta*model->MESAdelta;
    
    for (here = MESAinstances(model); here != NULL ;
              here=MESAnextInstance(here)) {

      vt                = CONSTKoverQ * here->MESAts;
      if(model->MESAmu1 == 0 && model->MESAmu2 == 0)
        here->MESAtMu = model->MESAmu*pow(here->MESAts/
                         model->MESAtmu,model->MESAxtm0);
      else {
        double muimp = model->MESAmu*pow(here->MESAts/
                       model->MESAtmu,model->MESAxtm0);
        double mupo  = model->MESAmu1*pow(model->MESAtmu/
                       here->MESAts,model->MESAxtm1) +
                       model->MESAmu2*pow(model->MESAtmu/
                       here->MESAts,model->MESAxtm2);
        here->MESAtMu = 1/(1/muimp+1/mupo);
      }
      here->MESAtTheta = model->MESAtheta;
      here->MESAtPhib  = model->MESAphib-model->MESAphib1*(here->MESAts-ckt->CKTnomTemp);
      here->MESAtVto   = model->MESAthreshold-model->MESAtvto*(here->MESAts-ckt->CKTnomTemp);
      here->MESAimax   = CHARGE*model->MESAnmax*model->MESAvs*here->MESAwidth;

      if(model->MESAlevel == 2)
        here->MESAgchi0  = CHARGE*here->MESAwidth/here->MESAlength;
      else
        here->MESAgchi0  = CHARGE*here->MESAwidth/here->MESAlength*here->MESAtMu;
      here->MESAbeta   = 2*EPSILONGAAS*model->MESAvs*model->MESAzeta*here->MESAwidth/
                          model->MESAd;
      here->MESAtEta   = model->MESAeta*(1+here->MESAts/model->MESAteta0)+
                          model->MESAteta1/here->MESAts;
      here->MESAtLambda= model->MESAlambda*(1-here->MESAts/model->MESAtlambda);
      here->MESAtLambdahf = model->MESAlambdahf*(1-here->MESAts/model->MESAtlambda);
      if(model->MESAlevel == 3)
        d = model->MESAdu;
      else
        d = model->MESAd;
      if(model->MESAlevel == 4)
        here->MESAn0   = model->MESAepsi*here->MESAtEta*vt/2/CHARGE/d;
      else
        here->MESAn0   = EPSILONGAAS*here->MESAtEta*vt/CHARGE/d;
      here->MESAnsb0   = EPSILONGAAS*here->MESAtEta*vt/CHARGE/
                                (model->MESAdu + model->MESAth);
      here->MESAisatb0 = CHARGE*here->MESAn0*vt*
                          here->MESAwidth/here->MESAlength;
      if(model->MESAlevel == 4)                    
        here->MESAcf     = 0.5*model->MESAepsi*here->MESAwidth;
      else  
        here->MESAcf     = 0.5*EPSILONGAAS*here->MESAwidth;
      here->MESAcsatfs = 0.5*model->MESAastar*here->MESAts*
        here->MESAts*exp(-here->MESAtPhib/(CONSTboltz*here->MESAts))*
        here->MESAlength*here->MESAwidth;
      here->MESAcsatfd = 0.5*model->MESAastar*here->MESAtd*
        here->MESAtd*exp(-here->MESAtPhib/(CONSTboltz*here->MESAtd))*
        here->MESAlength*here->MESAwidth;
      here->MESAggrwl  = model->MESAggr*here->MESAlength*here->MESAwidth*
        exp(model->MESAxchi*(here->MESAts-ckt->CKTnomTemp));
      if(here->MESAcsatfs != 0)
        here->MESAvcrits  = vt*log(vt/(CONSTroot2 * here->MESAcsatfs));
      else
        here->MESAvcrits = DBL_MAX;
      if(here->MESAcsatfd != 0) {
        double vtd = CONSTKoverQ * here->MESAtd;
        here->MESAvcritd  = vtd*log(vtd/(CONSTroot2 * here->MESAcsatfd));
      } else
        here->MESAvcritd = DBL_MAX;
      temp              = exp(here->MESAts/model->MESAtf);
      here->MESAfl     = model->MESAflo*temp;
      here->MESAdelf   = model->MESAdelfo*temp;
      if(model->MESArdi != 0.0)
        here->MESAtRdi   = model->MESArdi*(1+
          model->MESAtc1*(here->MESAtd-ckt->CKTnomTemp)+
          model->MESAtc2*(here->MESAtd-ckt->CKTnomTemp)*(here->MESAtd-ckt->CKTnomTemp));
      else
        here->MESAtRdi = 0;
      if(model->MESArsi != 0.0)    
        here->MESAtRsi   = model->MESArsi*(1+
          model->MESAtc1*(here->MESAts-ckt->CKTnomTemp)+
          model->MESAtc2*(here->MESAts-ckt->CKTnomTemp)*(here->MESAts-ckt->CKTnomTemp));
      else
        here->MESAtRsi = 0;
      if(model->MESAgateResist != 0.0)
        here->MESAtRg   = model->MESAgateResist*(1+
          model->MESAtc1*(here->MESAts-ckt->CKTnomTemp)+
          model->MESAtc2*(here->MESAts-ckt->CKTnomTemp)*(here->MESAts-ckt->CKTnomTemp));
      else
        here->MESAtRg = 0;
      if(model->MESAsourceResist != 0.0)
        here->MESAtRs   = model->MESAsourceResist*(1+
          model->MESAtc1*(here->MESAts-ckt->CKTnomTemp)+
          model->MESAtc2*(here->MESAts-ckt->CKTnomTemp)*(here->MESAts-ckt->CKTnomTemp));
      else
        here->MESAtRs = 0;
      if(model->MESAdrainResist != 0.0)
        here->MESAtRd   = model->MESAdrainResist*(1+
          model->MESAtc1*(here->MESAtd-ckt->CKTnomTemp)+
          model->MESAtc2*(here->MESAtd-ckt->CKTnomTemp)*(here->MESAtd-ckt->CKTnomTemp));
      else
        here->MESAtRd = 0;
      if(model->MESAri != 0.0)
        here->MESAtRi   = model->MESAri*(1+
          model->MESAtc1*(here->MESAts-ckt->CKTnomTemp)+
          model->MESAtc2*(here->MESAts-ckt->CKTnomTemp)*(here->MESAts-ckt->CKTnomTemp));
      else
        here->MESAtRi = 0;
      if(model->MESArf != 0.0)
        here->MESAtRf   = model->MESArf*(1+
          model->MESAtc1*(here->MESAtd-ckt->CKTnomTemp)+
          model->MESAtc2*(here->MESAtd-ckt->CKTnomTemp)*(here->MESAtd-ckt->CKTnomTemp));
      else
        here->MESAtRf = 0;
      if(here->MESAtRd != 0)
        here->MESAdrainConduct = 1/here->MESAtRd;
      else
        here->MESAdrainConduct = 0;
      if(here->MESAtRs != 0)
        here->MESAsourceConduct = 1/here->MESAtRs;
      else
        here->MESAsourceConduct = 0;
      if(here->MESAtRg != 0)
        here->MESAgateConduct = 1/here->MESAtRg;
      else
        here->MESAgateConduct = 0;
      if(here->MESAtRi != 0)
        here->MESAtGi = 1/here->MESAtRi;
      else
        here->MESAtGi = 0;
      if(here->MESAtRf != 0)
        here->MESAtGf = 1/here->MESAtRf;
      else
        here->MESAtGf = 0;
    }
  }
  return(OK);
}
