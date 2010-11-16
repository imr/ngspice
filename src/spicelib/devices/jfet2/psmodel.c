/*
 	Parker-Skellern MESFET model
 	
	Copyright (C) 1994, 1995, 1996  Macquarie University                    
	All Rights Reserved 
	Author: Anthony Parker
	Date:	2  Feb 1994  created
	        9  Feb 1994  correct NaN problem in strong cut-off region
	        20 MAR 1994  corrected capacitance initialization  
	        24 MAR 1994  added parameter MVST  
	        28 MAR 1994  reorganized declaration scopes 
            19 APR 1994  added new parameters: PS_HFETA, PS_HFE1, PS_HFE2,
                             PS_HFG1, and PS_HFG2
            18 May 1994  corrected 1/0 error when PS_VSUB=0
            15 Jul 1994  corrected errors in acload routine
            10 Aug 1995  added PS_VSUB to gds += gm*PS_VSUB*mvst*(vgt-vgst*(a..
			12 Sep 1995  changed _XXX to PS_XXX to aid portability
            13 Sep 1995  change to give arg=1-1/subfac; 
			                           if(vst!=0) gds+=gm*PS_VSUB..;
                                        gm *= arg;
			10 Feb 1996  change to names to match MicroSim code.
			5  Jul 1996  corrected diode eq (change Gmin*vgs to Gmin*vgd).
 
*****************************************************************************/

/*-----------
| functions defined in this file are:
    PSids()          returns dc drain source current and assigns other
                     current and branch conductances
    qgg()            static function that returns gate charge
    PScharge()       returns gate-source and gate-drain charge and capacitance
    PSacload()       returns small-signal conductance elements
    PSinstanceinit() initializes model parameters
 */

#define PSMODEL_C      /* activate local definitions in psmesfet.h */
#include "psmodel.h"

/*-----------
| dc current and conductance calculation */
double
PSids(
cref *ckt,
modl *model,
inst *here,
double vgs,
double vgd,
double *igs,
double *igd,
double *ggs,
double *ggd,
double *Gm,
double *Gds)
{
#define FX  -10.0 /* not too small else fatal rounding error in (rpt-a_rpt) */
#define MX  40.0  /* maximum exponential argument */
#define EMX 2.353852668370199842e17  /* exp(MX) */

   double idrain, arg;
   double area = AREA;

   { /* gate junction diodes */
      double zz;
      { /* gate-junction forward conduction */
         double Gmin   = GMIN;
         double Vt     = NVT;
         double isat   = IS   * area;
         if ((arg=vgs/Vt) > FX) {
            if(arg < MX) {
               *ggs=(zz=isat*exp(arg))/Vt+Gmin; *igs= zz -isat +Gmin*vgs;
            } else {
               *ggs=(zz=isat*EMX)/Vt+Gmin;     *igs=zz*(arg-MX+1)-isat+Gmin*vgs;
            }
         } else {
            *ggs = Gmin;                       *igs = -isat + Gmin * vgs;
         }
         if ((arg=vgd/Vt) > FX) {
            if(arg < MX) {
               *ggd=(zz=isat*exp(arg))/Vt+Gmin; *igd= zz -isat +Gmin*vgd;
            } else {
               *ggd=(zz=isat*EMX)/Vt+Gmin;     *igd=zz*(arg-MX+1)-isat+Gmin*vgd;
            }
         } else {
            *ggd = Gmin;                       *igd = -isat + Gmin * vgd;
         }
      }
      { /* gate-junction reverse 'breakdown' conduction */
         double Vbd    = VBD;
         double ibd    = IBD  * area;
         if ((arg=-vgs/Vbd) > FX) {
            if(arg < MX) {
               *ggs += (zz=ibd*exp(arg))/Vbd;  *igs -= zz-ibd;
            } else {
               *ggs += (zz=ibd*EMX)/Vbd;       *igs -= zz*((arg-MX)+1) - ibd;
            }
         } else                                *igs += ibd;
         if ((arg=-vgd/Vbd) > FX) {
            if(arg < MX) {
               *ggd += (zz=ibd*exp(arg))/Vbd;  *igd -= zz-ibd;
            } else {
               *ggd += (zz=ibd*EMX)/Vbd;       *igd -= zz*((arg-MX)+1) - ibd;
            }
         } else                                *igd += ibd;
      }
   }
    
   { /*  compute drain current and derivitives */
      double gm, gds;
      double vdst = vgs - vgd;
      double stepofour = STEP * FOURTH;
      { /* Include rate dependent threshold modulation */
         double vgst, dvgd, dvgs, h, vgdtrap, vgstrap, eta, gam;
         double vto = VTO;
         double LFg = LFGAM,   LFg1 = LFG1,  LFg2 = LFG2;
         double HFg = HFGAM,   HFg1 = HFG1,  HFg2 = HFG2;
         double HFe = HFETA,   HFe1 = HFE1,  HFe2 = HFE2;
         if(TRAN_ANAL) {
            double taug = TAUG;
            h = taug/(taug + stepofour);  h*=h; h*=h; /*4th power*/
            VGDTRAP_NOW = vgdtrap = h*VGDTRAP_BEFORE + (1-h) * vgd;
            VGSTRAP_NOW = vgstrap = h*VGSTRAP_BEFORE + (1-h) * vgs;
         } else {
            h = 0;
            VGDTRAP_NOW = vgdtrap = vgd;
            VGSTRAP_NOW = vgstrap = vgs;
         }
         vgst = vgs - vto;
         vgst -= (      LFg - LFg1*vgstrap + LFg2*vgdtrap)*vgdtrap;
         vgst += (eta = HFe - HFe1*vgdtrap + HFe2*vgstrap)*(dvgs = vgstrap-vgs);
         vgst += (gam = HFg - HFg1*vgstrap + HFg2*vgdtrap)*(dvgd = vgdtrap-vgd);
         { /* Exponential Subthreshold effect ids(vgst,vdst) */
            double vgt, subfac;
            double mvst = MVST;
            double vst = VSUB * (1 + mvst*vdst);
            if (vgst > FX*vst) {
               if (vgst > (arg=MX*vst)) { /* numerically large */
                  vgt = (EMX/(subfac = EMX+1))*(vgst-arg) + arg;
               } else  /* limit gate bias exponentially */
                  vgt = vst * log( subfac=(1 + exp(vgst/vst)) );
               { /* Dual Power-law ids(vgt,vdst) */
                  double mQ  = Q;
                  double PmQ = P - mQ;
                  double dvpd_dvdst=(double)D3*pow(vgt,PmQ);
                  double vdp = vdst*dvpd_dvdst; /*D3=P/Q/((VBI-vto)^PmQ)*/
                  { /* Early saturation effect ids(vgt,vdp) */
                     double za  = (double)ZA; /* sqrt(1 + Z)/2 */
                     double mxi = MXI;
                     double vsatFac = vgt/(mxi*vgt  + (double)XI_WOO);
                     double vsat=vgt/(1 + vsatFac);
                     double  aa = za*vdp+vsat/2.0;
                     double a_aa = aa-vsat;
                     double  rpt = sqrt( aa * aa + (arg=vsat*vsat*Z/4.0));
                     double a_rpt = sqrt(a_aa * a_aa + arg);
                     double vdt = (rpt - a_rpt);
                     double dvdt_dvdp = za * (aa/rpt - a_aa/a_rpt);
                     double dvdt_dvgt = (vdt - vdp*dvdt_dvdp)
                           *(1 + mxi*vsatFac*vsatFac)/(1 + vsatFac)/vgt;
                     { /* Intrinsic Q-law FET equation ids(vgt,vdt) */
                        gds=pow(vgt-vdt,mQ-1);
                        idrain = vdt*gds + vgt*(gm=pow(vgt,mQ-1)-gds);
                        gds *= mQ;
                        gm *= mQ;
                     }
                     gm += gds*dvdt_dvgt;
                     gds *= dvdt_dvdp;
                  }
                  gm += gds*PmQ*vdp/vgt;
                  gds *= dvpd_dvdst;
               }
               arg = 1 - 1/subfac;
               if(vst != 0) gds += gm*VSUB*mvst*(vgt-vgst*arg)/vst;
               gm *= arg;
            } else { /* in extreme cut-off (numerically) */
               idrain = gm = gds = 0.0e0;
            }
         }
         gds += gm*(arg = h*gam +
                   (1-h)*(HFe1*dvgs-HFg2*dvgd+2*LFg2*vgdtrap-LFg1*vgstrap+LFg));
         gm *= 1 - h*eta + (1-h)*(HFe2*dvgs -HFg1*dvgd + LFg1*vgdtrap) - arg;
      }
      { /* apply channel length modulation and beta scaling */
         double lambda = LAM;
         double beta   = BETA  * area;
         gm *= (arg = beta*(1 + lambda*vdst));
         gds = beta*lambda*idrain + gds*arg;
         idrain *= arg;
      }
        
      { /* apply thermal reduction of drain current */
        double h, pfac, pAverage;
        double delta = DELT / area;
        if(TRAN_ANAL) {
           double taud = TAUD;
           h = taud/(taud + stepofour);    h*=h; h*=h;
           POWR_NOW = pAverage = h*POWR_BEFORE + (1-h)*vdst*idrain;
        } else {
           POWR_NOW = POWR_BEFORE = pAverage = vdst*idrain;  h = 0;
        }
        idrain /= (pfac = 1+pAverage*delta);
        *Gm  = gm * (arg = (h*delta*POWR_BEFORE + 1)/pfac/pfac);
        *Gds = gds * arg - (1-h) * delta*idrain*idrain;
      }
   }
   return(idrain);
}

/*-----------
| code based on Statz et. al. capacitance model,  IEEE Tran ED Feb 87 */
static double
qgg(
    double vgs, 
    double vgd, 
    double gamma, 
    double pb, 
    double alpha, 
    double vto, 
    double vmax, 
    double xc, 
    double cgso, 
    double cgdo, 
    double *cgs, 
    double *cgd
)
/*vgs, vgd, gamma, pb, alpha, vto, vmax, xc, cgso, cgdo, *cgs, *cgd;*/
{
   double qrt, ext, Cgso, cpm, cplus, cminus;
   double vds   = vgs-vgd;
   double d1_xc = 1-xc;
   double vert  = sqrt( vds * vds + alpha );
   double veff  = 0.5*(vgs + vgd + vert) + gamma*vds;
   double vnr   = d1_xc*(veff-vto);
   double vnrt  = sqrt( vnr*vnr + 0.04 );
   double vnew  = veff + 0.5*(vnrt - vnr);
   if ( vnew < vmax ) {
      ext  = 0;
      qrt  = sqrt(1 - vnew/pb);
      Cgso = 0.5*cgso/qrt*(1+xc + d1_xc*vnr/vnrt);
   } else {
      double vx  = 0.5*(vnew-vmax);
      double par = 1+vx/(pb-vmax);
      qrt  = sqrt(1 - vmax/pb);
      ext  = vx*(1 + par)/qrt;
      Cgso = 0.5*cgso/qrt*(1+xc + d1_xc*vnr/vnrt) * par;
   }
   cplus = 0.5*(1 + (cpm = vds/vert)); cminus = cplus - cpm;
   *cgs = Cgso*(cplus +gamma) + cgdo*(cminus+gamma);
   *cgd = Cgso*(cminus-gamma) + cgdo*(cplus -gamma);
   return(cgso*((pb+pb)*(1-qrt) + ext) + cgdo*(veff - vert));
}

/*-----------
| call during ac analysis initialisation or during transient analysis */
void
PScharge(
cref *ckt,
modl *model,
inst *here,
double vgs,
double vgd,
double *capgs,
double *capgd
)
{
#define QGG(a,b,c,d) qgg(a,b,gac,phib,alpha,vto,vmax,xc,czgs,czgd,c,d)
/*  double qgg(); */
      
   double czgs = CGS * AREA;
   double czgd = CGD * AREA;
   double vto   = VTO;
   double alpha = (double)ALPHA;  /* (XI*woo/(XI+1)/2)^2 */
   double xc    = XC;
   double vmax  = VMAX;
   double phib  = VBI;
   double gac   = ACGAM;
   
   if(/*TRAN_INIT ||*/ !TRAN_ANAL)
       QGS_NOW = QGD_NOW = QGS_BEFORE = QGD_BEFORE
          = QGG(vgs,vgd,capgs,capgd);
   else {
      double cgsna,cgsnc;
      double cgdna,cgdnb, a_cap;
      double vgs1  = VGS1;
      double vgd1  = VGD1;
      double qgga=QGG(vgs ,vgd ,&cgsna,&cgdna);
      double qggb=QGG(vgs1,vgd ,&a_cap,&cgdnb);
      double qggc=QGG(vgs ,vgd1,&cgsnc,&a_cap);
      double qggd=QGG(vgs1,vgd1,&a_cap,&a_cap);
      QGS_NOW = QGS_BEFORE + 0.5 * (qgga-qggb + qggc-qggd);
      QGD_NOW = QGD_BEFORE + 0.5 * (qgga-qggc + qggb-qggd);
      *capgs = 0.5 * (cgsna + cgsnc);
      *capgd = 0.5 * (cgdna + cgdnb);
   }
}


/*-----------
| call for each frequency in ac analysis */
void
PSacload(
cref *ckt,
modl *model,
inst *here,
double vgs,
double vgd,
double ids,
double omega,
double *Gm,
double *xGm,
double *Gds,
double *xGds
)
{
    double arg;
    double vds = vgs - vgd;
    double LFgam = LFGAM;
    double LFg1 = LFG1;
    double LFg2 = LFG2*vgd;
    double HFg1 = HFG1;
    double HFg2 = HFG2*vgd;
    double HFeta  = HFETA;
    double HFe1 = HFE1;
    double HFe2 = HFE2*vgs;
    double hfgam= HFGAM - HFg1*vgs + HFg2;
    double eta  = HFeta - HFe1*vgd + HFe2;
    double lfga = LFgam - LFg1*vgs + LFg2 + LFg2;
    double gmo  = *Gm/(1 - lfga + LFg1*vgd);
    
    double wtg = TAUG * omega;
    double wtgdet = 1 + wtg*wtg;
    double gwtgdet = gmo/wtgdet;

    double gdsi = (arg=hfgam - lfga)*gwtgdet;
    double gdsr = arg*gmo - gdsi;
    double gmi  = (eta + LFg1*vgd)*gwtgdet + gdsi;
 
    double xgds = wtg*gdsi;
    double  gds = *Gds + gdsr;
    double  xgm = -wtg*gmi;
    double   gm = gmi + gmo*(1 -eta - hfgam);
                
    double delta = DELT / AREA;
    double wtd = TAUD * omega ;
    double wtddet = 1 + wtd * wtd;
    double fac = delta * ids;
    double del = 1/(1 - fac * vds);
    double dd = (del-1) / wtddet;
    double dr = del - dd;
    double di = wtd * dd;
                
    double cdsqr = fac * ids * del * wtd/wtddet;

    NG_IGNORE(ckt);

    *Gm   = dr*gm  - di*xgm;
    *xGm  = di*gm  + dr*xgm;
 
    *Gds  = dr*gds - di*xgds + cdsqr*wtd;
    *xGds = di*gds + dr*xgds + cdsqr;
}


void    /* call when temperature changes */
PSinstanceinit(
   modl *model,
   inst *here 
)
{
#ifndef PARAM_CAST     /* allow casting to parameter type */
#define PARAM_CAST     /* if not specified then don't cast */
#endif
    
    double woo = (VBI - VTO);
    XI_WOO = PARAM_CAST (XI * woo);
    ZA     = PARAM_CAST (sqrt(1 + Z)/2);
    ALPHA  = PARAM_CAST (XI_WOO*XI_WOO/(XI+1)/(XI+1)/ 4);
    D3     = PARAM_CAST (P/Q/pow(woo,(P - Q)));
}
