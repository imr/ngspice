/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hfetdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
HFETAtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;
    double vt;
    double temp;

    /*  loop through all the diode models */
    for( ; model != NULL; model = HFETAnextModel(model)) {
        if(model->HFETArd != 0) {
            model->HFETAdrainConduct = 1/model->HFETArd;
        } else {
            model->HFETAdrainConduct = 0;
        }
        if(model->HFETArs != 0) {
            model->HFETAsourceConduct = 1/model->HFETArs;
        } else {
            model->HFETAsourceConduct = 0;
        }
        if(model->HFETArg != 0) {
            model->HFETAgateConduct = 1/model->HFETArg;
        } else {
            model->HFETAgateConduct = 0;
        }
        if(model->HFETAri != 0) {
            model->HFETAgi = 1/model->HFETAri;
        } else {
            model->HFETAgi = 0;
        }
        if(model->HFETArf != 0) {
            model->HFETAgf = 1/model->HFETArf;
        } else {
            model->HFETAgf = 0;
        }
        model->HFETAdeltaSqr = model->HFETAdelta*model->HFETAdelta;
        model->HFETAthreshold *= model->HFETAtype;

        if(!model->HFETAvt2Given)
          VT2 = VTO;
        if(!model->HFETAvt1Given)
          IN_VT1 = VTO+CHARGE*NMAX*DI/EPSI;
          
        for (here = HFETAinstances(model); here != NULL ;
                here=HFETAnextInstance(here)) {

            if(!here->HFETAdtempGiven) {
                here->HFETAdtemp = 0.0;
            }

            if(!here->HFETAtempGiven) {
                here->HFETAtemp = ckt->CKTtemp + here->HFETAdtemp;
            }

            vt      = CONSTKoverQ*TEMP;
            TLAMBDA = LAMBDA + KLAMBDA*(TEMP-ckt->CKTnomTemp);            
            TMU     = MU - KMU*(TEMP-ckt->CKTnomTemp);
            TVTO    = VTO - KVTO*(TEMP-ckt->CKTnomTemp);
            N0      = EPSI*ETA*vt/2/CHARGE/(DI+DELTAD);
            N01     = EPSI*ETA1*vt/2/CHARGE/D1;
            if(model->HFETAeta2Given)
              N02   = EPSI*ETA2*vt/2/CHARGE/D2;
            else
              N02   = 0.0;  
            GCHI0   = CHARGE*W*TMU/L;
            CF      = 0.5*EPSI*W;
            IMAX    = CHARGE*NMAX*VS*W;
            IS1D    = JS1D*W*L/2;
            IS2D    = JS2D*W*L/2;
            IS1S    = JS1S*W*L/2;
            IS2S    = JS2S*W*L/2;
            ISO     = ASTAR*W*L/2;
            GGRWL   = GGR*L*W/2;
            temp    = exp(TEMP/model->HFETAtf);
            FGDS    = model->HFETAfgds*temp;
            DELF    = model->HFETAdelf*temp;
            if(model->HFETAgatemod == 0) {
              if(IS1S != 0)
                here->HFETAvcrit  = vt*log(vt/(CONSTroot2*IS1S));
              else
                here->HFETAvcrit = DBL_MAX;
            } else {
              if(ISO != 0.0)            
                here->HFETAvcrit = vt*log(vt/(CONSTroot2*ISO));
              else
                here->HFETAvcrit = DBL_MAX;
            }    
        }
    }
    return(OK);
}
