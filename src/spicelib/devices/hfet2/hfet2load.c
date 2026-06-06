/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
#include "ngspice/const.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

void Pause(void);

static void hfeta2(HFET2model *model, HFET2instance *here, CKTcircuit *ckt,
                  double vgs, double vds, double *cdrain, double *gm,
                  double *gds, double *capgs, double *capgd);


int HFET2load(GENmodel *inModel, CKTcircuit *ckt)
{

  HFET2model *model = (HFET2model*)inModel;
  HFET2instance *here;
  double capgd;
  double capgs;
  double cd;
  double cdhat = 0.0;
  double cdrain;
  double cdreq;
  double ceq;
  double ceqgd;
  double ceqgs;
  double cg;
  double cgd;
  double cghat = 0.0;
  double delvds;
  double delvgd;
  double delvgs;
  double gdpr;
  double gds;
  double geq;
  double ggd;
  double ggs;
  double gm;
  double gspr;
  double vcrit;
  double vds;
  double vgd;
  double vgd1;
  double vgs;
  double vgs1;
  double vt;
  double vto;
#ifndef PREDICTOR
  double xfact;
#endif
  int    icheck;
  int    ichk1;
  int    error;
  int    inverse;

  double m;

  for( ; model != NULL; model = HFET2nextModel(model)) {
    for(here = HFET2instances(model); here != NULL; 
        here=HFET2nextInstance(here)) {

      gdpr = model->HFET2drainConduct;
      gspr = model->HFET2sourceConduct;
      vcrit = VCRIT;
      vto = TVTO;
      vt  = CONSTKoverQ*TEMP;
      icheck = 1;
      if( ckt->CKTmode & MODEINITSMSIG) {
        vgs = *(ckt->CKTstate0 + here->HFET2vgs);
        vgd = *(ckt->CKTstate0 + here->HFET2vgd);
      } else if(ckt->CKTmode & MODEINITTRAN) {
        vgs = *(ckt->CKTstate1 + here->HFET2vgs);
        vgd = *(ckt->CKTstate1 + here->HFET2vgd);
      } else if((ckt->CKTmode & MODEINITJCT) && (ckt->CKTmode & MODETRANOP) &&
                    (ckt->CKTmode & MODEUIC) ) {
        vds = model->HFET2type*here->HFET2icVDS;
        vgs = model->HFET2type*here->HFET2icVGS;
        vgd = vgs-vds;
      } else if ( (ckt->CKTmode & MODEINITJCT) && (here->HFET2off == 0)  ) {
        vgs = -1;
        vgd = -1;
      } else if((ckt->CKTmode & MODEINITJCT) ||
                ((ckt->CKTmode & MODEINITFIX) && (here->HFET2off))) {
        vgs = 0;
        vgd = 0;
      } else {
#ifndef PREDICTOR
        if(ckt->CKTmode & MODEINITPRED) {
          xfact = ckt->CKTdelta/ckt->CKTdeltaOld[2];
          *(ckt->CKTstate0 + here->HFET2vgs) = 
               *(ckt->CKTstate1 + here->HFET2vgs);
          vgs = (1+xfact) * *(ckt->CKTstate1 + here->HFET2vgs) -
               xfact * *(ckt->CKTstate2 + here->HFET2vgs);
          *(ckt->CKTstate0 + here->HFET2vgd) = 
               *(ckt->CKTstate1 + here->HFET2vgd);
          vgd = (1+xfact)* *(ckt->CKTstate1 + here->HFET2vgd) -
               xfact * *(ckt->CKTstate2 + here->HFET2vgd);
          *(ckt->CKTstate0 + here->HFET2cg) = 
               *(ckt->CKTstate1 + here->HFET2cg);
          *(ckt->CKTstate0 + here->HFET2cd) = 
               *(ckt->CKTstate1 + here->HFET2cd);
          *(ckt->CKTstate0 + here->HFET2cgd) =
               *(ckt->CKTstate1 + here->HFET2cgd);
          *(ckt->CKTstate0 + here->HFET2gm) =
               *(ckt->CKTstate1 + here->HFET2gm);
          *(ckt->CKTstate0 + here->HFET2gds) =
               *(ckt->CKTstate1 + here->HFET2gds);
          *(ckt->CKTstate0 + here->HFET2ggs) =
               *(ckt->CKTstate1 + here->HFET2ggs);
          *(ckt->CKTstate0 + here->HFET2ggd) =
               *(ckt->CKTstate1 + here->HFET2ggd);
        } else {
#endif /* PREDICTOR */
          vgs = model->HFET2type*
                 (*(ckt->CKTrhsOld+ here->HFET2gateNode)- *(ckt->CKTrhsOld+ 
                 here->HFET2sourcePrimeNode));
          vgd = model->HFET2type*
                 (*(ckt->CKTrhsOld+here->HFET2gateNode)- *(ckt->CKTrhsOld+
                 here->HFET2drainPrimeNode));
#ifndef PREDICTOR
        }
#endif /* PREDICTOR */
        delvgs=vgs - *(ckt->CKTstate0 + here->HFET2vgs);
        delvgd=vgd - *(ckt->CKTstate0 + here->HFET2vgd);
        delvds=delvgs - delvgd;
        cghat= *(ckt->CKTstate0 + here->HFET2cg) + 
               *(ckt->CKTstate0 + here->HFET2ggd)*delvgd +
               *(ckt->CKTstate0 + here->HFET2ggs)*delvgs;
        cdhat= *(ckt->CKTstate0 + here->HFET2cd) +
               *(ckt->CKTstate0 + here->HFET2gm)*delvgs +
               *(ckt->CKTstate0 + here->HFET2gds)*delvds -
               *(ckt->CKTstate0 + here->HFET2ggd)*delvgd;

         /*   bypass if solution has not changed */

         if((ckt->CKTbypass) &&
           (!(ckt->CKTmode & MODEINITPRED)) &&
           (fabs(delvgs) < ckt->CKTreltol*MAX(fabs(vgs),
            fabs(*(ckt->CKTstate0 + here->HFET2vgs)))+
            ckt->CKTvoltTol) )
         if ( (fabs(delvgd) < ckt->CKTreltol*MAX(fabs(vgd),
            fabs(*(ckt->CKTstate0 + here->HFET2vgd)))+
            ckt->CKTvoltTol))
         if ( (fabs(cghat-*(ckt->CKTstate0 + here->HFET2cg)) 
            < ckt->CKTreltol*MAX(fabs(cghat),
            fabs(*(ckt->CKTstate0 + here->HFET2cg)))+
            ckt->CKTabstol) ) if ( /* hack - expression too big */
            (fabs(cdhat-*(ckt->CKTstate0 + here->HFET2cd))
            < ckt->CKTreltol*MAX(fabs(cdhat),
            fabs(*(ckt->CKTstate0 + here->HFET2cd)))+
            ckt->CKTabstol) ) {

            /* we can do a bypass */
            vgs= *(ckt->CKTstate0 + here->HFET2vgs);
            vgd= *(ckt->CKTstate0 + here->HFET2vgd);
            vds= vgs-vgd;
            cg= *(ckt->CKTstate0 + here->HFET2cg);
            cd= *(ckt->CKTstate0 + here->HFET2cd);
            cgd= *(ckt->CKTstate0 + here->HFET2cgd);
            gm= *(ckt->CKTstate0 + here->HFET2gm);
            gds= *(ckt->CKTstate0 + here->HFET2gds);
            ggs= *(ckt->CKTstate0 + here->HFET2ggs);
            ggd= *(ckt->CKTstate0 + here->HFET2ggd);
            goto load;
          }
  
          /*  limit nonlinear branch voltages */ 

          ichk1=1;
          vgs = DEVpnjlim(vgs,*(ckt->CKTstate0 + here->HFET2vgs),CONSTvt0,vcrit, &icheck);
          vgd = DEVpnjlim(vgd,*(ckt->CKTstate0 + here->HFET2vgd),CONSTvt0,vcrit,&ichk1);
          if(ichk1 == 1) {
            icheck=1;
          }
          vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->HFET2vgs),TVTO);
          vgd = DEVfetlim(vgd,*(ckt->CKTstate0 + here->HFET2vgd),TVTO);
        }
      cg = 0;
      cgd = 0;
      ggd = 0;
      ggs = 0;
      vds  = vgs-vgd;
      {
        double arg  = -vgs*DEL/vt;
        double earg = exp(arg);
        double vtn  = N*vt;
        double expe = exp(vgs/vtn);
        ggs  = JSLW*expe/vtn+GGRLW*earg*(1-arg);
        cg   = JSLW*(expe-1)+GGRLW*vgs*earg;
        arg  = -vgd*DEL/vt;
        earg = exp(arg);
        expe = exp(vgd/vtn);
        ggd  = JSLW*expe/vtn+GGRLW*earg*(1-arg);
        cgd  = JSLW*(expe-1)+GGRLW*vgd*earg;
        cg  += cgd;
      }      
      if(vds < 0) {
        vds = -vds;
        inverse = 1;
      } else
        inverse = 0;
      hfeta2(model,here,ckt,vds>0?vgs:vgd,vds,&cdrain,&gm,&gds,&capgs,&capgd);
      if(inverse) {
        double temp;
        cdrain = -cdrain;
        vds = -vds;
        temp = capgs;
        capgs = capgd;
        capgd = temp;
      }
      cd = cdrain - cgd;
      if((ckt->CKTmode & (MODETRAN|MODEINITSMSIG)) || ((ckt->CKTmode & MODETRANOP) &&
        (ckt->CKTmode & MODEUIC)) ){
        /*    charge storage elements */ 
        vgs1 = *(ckt->CKTstate1 + here->HFET2vgs);
        vgd1 = *(ckt->CKTstate1 + here->HFET2vgd);

        if(ckt->CKTmode & MODEINITTRAN) {
          *(ckt->CKTstate1 + here->HFET2qgs) = capgs*vgs;
          *(ckt->CKTstate1 + here->HFET2qgd) = capgd*vgd;
        }
        *(ckt->CKTstate0+here->HFET2qgs) = *(ckt->CKTstate1+here->HFET2qgs)
                        + capgs*(vgs-vgs1);
        *(ckt->CKTstate0+here->HFET2qgd) = *(ckt->CKTstate1+here->HFET2qgd)
                        + capgd*(vgd-vgd1);

        /*   store small-signal parameters */

        if( (!(ckt->CKTmode & MODETRANOP)) || (!(ckt->CKTmode & MODEUIC)) ) {
          if(ckt->CKTmode & MODEINITSMSIG) {
            *(ckt->CKTstate0 + here->HFET2qgs) = capgs;
            *(ckt->CKTstate0 + here->HFET2qgd) = capgd;
            continue; /*go to 1000*/
          }
          
          /*   transient analysis */ 

          if(ckt->CKTmode & MODEINITTRAN) {
            *(ckt->CKTstate1 + here->HFET2qgs) = *(ckt->CKTstate0 + here->HFET2qgs);
            *(ckt->CKTstate1 + here->HFET2qgd) = *(ckt->CKTstate0 + here->HFET2qgd);
          }
          error = NIintegrate(ckt,&geq,&ceq,capgs,here->HFET2qgs);
          if(error) return(error);
          ggs = ggs + geq;
          cg = cg + *(ckt->CKTstate0 + here->HFET2cqgs);
          error = NIintegrate(ckt,&geq,&ceq,capgd,here->HFET2qgd);
          if(error) return(error);
          ggd = ggd + geq;
          cg = cg + *(ckt->CKTstate0 + here->HFET2cqgd);
          cd = cd - *(ckt->CKTstate0 + here->HFET2cqgd);
          cgd = cgd + *(ckt->CKTstate0 + here->HFET2cqgd);
          if (ckt->CKTmode & MODEINITTRAN) {
            *(ckt->CKTstate1 + here->HFET2cqgs) = *(ckt->CKTstate0 + here->HFET2cqgs);
            *(ckt->CKTstate1 + here->HFET2cqgd) = *(ckt->CKTstate0 + here->HFET2cqgd);
          }
        }
      }

      /*  check convergence */ 

      if( (!(ckt->CKTmode & MODEINITFIX)) | (!(ckt->CKTmode & MODEUIC))) {
        if((icheck == 1) 
          || (fabs(cghat-cg) >= ckt->CKTreltol*
          MAX(fabs(cghat),fabs(cg))+ckt->CKTabstol) ||
          (fabs(cdhat-cd) > ckt->CKTreltol*
          MAX(fabs(cdhat),fabs(cd))+ckt->CKTabstol) 
           ) {
          ckt->CKTnoncon++;
        }
      }
      *(ckt->CKTstate0 + here->HFET2vgs) = vgs;
      *(ckt->CKTstate0 + here->HFET2vgd) = vgd;
      *(ckt->CKTstate0 + here->HFET2cg) = cg;
      *(ckt->CKTstate0 + here->HFET2cd) = cd;
      *(ckt->CKTstate0 + here->HFET2cgd) = cgd;
      *(ckt->CKTstate0 + here->HFET2gm) = gm;
      *(ckt->CKTstate0 + here->HFET2gds) = gds;
      *(ckt->CKTstate0 + here->HFET2ggs) = ggs;
      *(ckt->CKTstate0 + here->HFET2ggd) = ggd;
      
      /*    load current vector */

load:

      m = here->HFET2m;

      ceqgd=model->HFET2type*(cgd-ggd*vgd);
      ceqgs=model->HFET2type*((cg-cgd)-ggs*vgs);
      cdreq=model->HFET2type*((cd+cgd)-gds*vds-gm*vgs);
      *(ckt->CKTrhs + here->HFET2gateNode)        += m * (-ceqgs-ceqgd);
      *(ckt->CKTrhs + here->HFET2drainPrimeNode)  += m * (-cdreq+ceqgd);
      *(ckt->CKTrhs + here->HFET2sourcePrimeNode) += m * (cdreq+ceqgs);

      /*   load y matrix */ 

      *(here->HFET2drainDrainPrimePtr)          += m * (-gdpr);
      *(here->HFET2gateDrainPrimePtr)           += m * (-ggd);
      *(here->HFET2gateSourcePrimePtr)          += m * (-ggs);
      *(here->HFET2sourceSourcePrimePtr)        += m * (-gspr);
      *(here->HFET2drainPrimeDrainPtr)          += m * (-gdpr);
      *(here->HFET2drainPrimeGatePtr)           += m * (gm-ggd);
      *(here->HFET2drainPriHFET2ourcePrimePtr)  += m * (-gds-gm);
      *(here->HFET2sourcePrimeGatePtr)          += m * (-ggs-gm);
      *(here->HFET2sourcePriHFET2ourcePtr)      += m * (-gspr);
      *(here->HFET2sourcePrimeDrainPrimePtr)    += m * (-gds);
      *(here->HFET2drainDrainPtr)               += m * (gdpr);
      *(here->HFET2gateGatePtr)                 += m * (ggd+ggs);
      *(here->HFET2sourceSourcePtr)             += m * (gspr);
      *(here->HFET2drainPrimeDrainPrimePtr)     += m * (gdpr+gds+ggd);
      *(here->HFET2sourcePriHFET2ourcePrimePtr) += m * (gspr+gds+gm+ggs);
    }
  }
  return(OK);
  
}




static void hfeta2(HFET2model *model, HFET2instance *here, CKTcircuit *ckt,
                  double vgs, double vds, double *cdrain, double *gm,
                  double *gds, double *capgs, double *capgd)

{
           
  double vt;
  double vgt;
  double vgt0;
  double sigma;
  double vgte;
  double isat;
  double isatm;
  double ns;
  double nsm;
  double a;
  double b;
  double c;
  double d;
  double e;
  double g;
  double h;
  double p;
  double q;
  double s;
  double t;
  double u;
  double nsc = 0.0;
  double nsn = 0.0;
  double temp;
  double etavth;
  double gch;
  double gchi;
  double gchim;
  double vsate;
  double vdse;
  double cg1;
  double cgc;
  double rt;
  double vl;
  double delidgch;
  double delgchgchi;
  double delgchins;
  double delnsnsm;
  double delnsmvgt;
  double delvgtevgt;
  double delidvsate;
  double delvsateisat;
  double delisatisatm;
  double delisatmvgte;
  double delisatmgchim;
  double delvsategch;
  double delidvds;
  double delvgtvgs;
  double delvsatevgt;

    NG_IGNORE(ckt);

  vt     = CONSTKoverQ*TEMP;
  etavth = ETA*vt;
  vl     = VS/TMU*L;
  rt     = RSI+RDI;
  vgt0   = vgs - TVTO;
  s      = exp((vgt0-VSIGMAT)/VSIGMA);
  sigma  = SIGMA0/(1+s);
  vgt    = vgt0+sigma*vds;
  u      = 0.5*vgt/vt-1;
  t      = sqrt(DELTA2+u*u);
  vgte   = vt*(2+u+t);
  b      = exp(vgt/etavth);
  if(model->HFET2eta2Given && model->HFET2d2Given) {
    nsc    = N02*exp((vgt+TVTO-VT2)/(ETA2*vt));
    nsn    = 2*N0*log(1+0.5*b);
    nsm    = nsn*nsc/(nsn+nsc);
  } else {
    nsm = 2*N0*log(1+0.5*b);
  }
  if(nsm < 1.0e-38) {
    *cdrain = 0;
    *gm = 0.0;
    *gds = 0.0;
    *capgs = CF;
    *capgd = CF;
    return;
  }
  c      = pow(nsm/TNMAX,GAMMA);
  q      = pow(1+c,1.0/GAMMA);
  ns     = nsm/q;
  gchi   = GCHI0*ns;
  gch    = gchi/(1+gchi*rt);
  gchim  = GCHI0*nsm;
  h      = sqrt(1+2*gchim*RSI + vgte*vgte/(vl*vl));
  p      = 1+gchim*RSI+h;
  isatm  = gchim*vgte/p;
  g      = pow(isatm/IMAX,GAMMA);
  isat   = isatm/pow(1+g,1/GAMMA);
  vsate  = isat/gch;
  d      = pow(vds/vsate,M);
  e      = pow(1+d,1.0/M);
  delidgch      = vds*(1+TLAMBDA*vds)/e;
  *cdrain       = gch*delidgch;
  delidvsate    = (*cdrain)*d/vsate/(1+d);
  delidvds      = gch*(1+2*TLAMBDA*vds)/e-(*cdrain)*
                  pow(vds/vsate,M-1)/(vsate*(1+d));
  a             = 1+gchi*rt;
  delgchgchi    = 1.0/(a*a);
  delgchins     = GCHI0;
  delnsnsm      = ns/nsm*(1-c/(1+c));
  delvgtevgt    = 0.5*(1+u/t);
  delnsmvgt     = N0/etavth/(1.0/b + 0.5);
  if(model->HFET2eta2Given && model->HFET2d2Given)
    delnsmvgt     = nsc*(nsc*delnsmvgt+nsn*nsn/(ETA2*vt))/((nsc+nsn)*(nsc+nsn));
  delvsateisat  = 1.0/gch;
  delisatisatm  = isat/isatm*(1-g/(1+g));
  delisatmvgte  = gchim*(p - vgte*vgte/(vl*vl*h))/(p*p);
  delvsategch   = -vsate/gch;
  delisatmgchim = vgte*(p - gchim*RSI*(1+1.0/h))/(p*p);
  delvgtvgs     = 1-vds*SIGMA0/VSIGMA*s/((1+s)*(1+s));
  p             = delgchgchi*delgchins*delnsnsm*delnsmvgt;
  delvsatevgt   = (delvsateisat*delisatisatm*(delisatmvgte*delvgtevgt +
                  delisatmgchim*GCHI0*delnsmvgt)+delvsategch*p);
  g             = delidgch*p + delidvsate*delvsatevgt;
  *gm           = g*delvgtvgs;
  *gds          = delidvds + g*sigma;

  /* Capacitance calculations */
  temp          = ETA1*vt;
  cg1           = 1/(D1/EPSI+temp*exp(-(vgs-HFET2_VT1)/temp));
  cgc           = W*L*(CHARGE*delnsnsm*delnsmvgt*delvgtvgs+cg1);
  vdse          = vds*pow(1+pow(vds/vsate,MC),-1.0/MC);
  a             = (vsate-vdse)/(2*vsate-vdse);
  a             = a*a;
  temp          = 2.0/3.0;
  p             = PP + (1-PP)*exp(-vds/vsate);
  *capgs        = CF+2*temp*cgc*(1-a)/(1+p);
  a             = vsate/(2*vsate-vdse);
  a             = a*a;
  *capgd        = CF+2*p*temp*cgc*(1-a)/(1+p);

}
