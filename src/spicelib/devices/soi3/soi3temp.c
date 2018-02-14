/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SOI3temp(GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model *)inModel;
    SOI3instance *here;

/* All variables ending in 1 denote that they pertain to the model
   and so use model->SOI3tnom for temperature - the others use
   here->SOI3temp.  REFTEMP is temp at which hard-coded quantities
   are given. */

    double egfet,egfet1;    /* Band Gap */
    double fact1,fact2;     /* temperature/REFTEMP */
    double kt,kt1;          /* kT @ various temps */
    double arg1;            /* ??? */
    double ratio,ratio4;    /* (temp/tnom) and (temp/tnom)^(3/2) */
    double phio;            /* temp adjusted phi PHI0*/
    double pbo;
    double gmanew,gmaold;
    double capfact;
    double pbfact1,pbfact;  /* ??? */
    double vt,vtnom;
    double wkfngfs;         /* work function difference phi(gate,Si) */
    double wkfngf;          /* work fn of front gate */
    double wkfngbs;         /* work fn diff of back gate = 0 usu. */
    double fermig;          /* fermi level of gate */
    double fermis;          /* fermi level of Si */
    double xd_max;	       /* Minimum Si film thickness for this model to be valid */
    double eta_s;

    /* JimB - new variables for improved threshold voltage conversion model. */

    double Edelta0;
    double psi_delta0;

    /* loop through all the transistor models */
    for( ; model != NULL; model = SOI3nextModel(model))
    {

        /* perform model defaulting */
        if(!model->SOI3tnomGiven)
        {
            model->SOI3tnom = ckt->CKTnomTemp;
        }

        fact1 = model->SOI3tnom/REFTEMP;
        vtnom = model->SOI3tnom*CONSTKoverQ;
        kt1 = CONSTboltz * model->SOI3tnom;
        egfet1 = 1.16-(7.02e-4*model->SOI3tnom*model->SOI3tnom)/
                (model->SOI3tnom+1108);
        arg1 = -egfet1/(kt1+kt1)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
        /* this is -Egnom + Egref. - sign due to it being in bracket with log(fact1)
           'cos we wanted log(1/fact1). */
        pbfact1 = -2*vtnom *(1.5*log(fact1)+CHARGE*arg1);
        /* 2 comes from fact phi=2*phi_F
                                 ^       */

        /* now model parameter preprocessing */

        if(!model->SOI3frontOxideThicknessGiven ||
            model->SOI3frontOxideThickness == 0 ||
           !model->SOI3backOxideThicknessGiven  ||
            model->SOI3backOxideThickness == 0  ||
           !model->SOI3bodyThicknessGiven       ||
            model->SOI3bodyThickness == 0)
        {

        		SPfrontEnd->IFerrorf (ERR_FATAL,
            	"%s: SOI3 device film thickness must be supplied",
            	model->SOI3modName);
            return(E_BADPARM);              
        }
        else /* Oxide and film thicknesses are supplied. */
        {
            model->SOI3frontOxideCapFactor = 3.9 * 8.854214871e-12/
                    model->SOI3frontOxideThickness;
            model->SOI3backOxideCapFactor = 3.9 * 8.854214871e-12/
                    model->SOI3backOxideThickness;
            model->SOI3bodyCapFactor = 11.7 * 8.854214871e-12/
                    model->SOI3bodyThickness;
            model->SOI3C_ssf = CHARGE*model->SOI3frontSurfaceStateDensity*1e4;
            model->SOI3C_ssb = CHARGE*model->SOI3backSurfaceStateDensity*1e4;
            eta_s = 1 + model->SOI3C_ssf/model->SOI3frontOxideCapFactor;
			       
            if(!model->SOI3transconductanceGiven)
            {
               if(!model->SOI3surfaceMobilityGiven)
               {
                  model->SOI3surfaceMobility=600;
               }
               model->SOI3transconductance = model->SOI3surfaceMobility *
                        model->SOI3frontOxideCapFactor * 1e-4 /*(m**2/cm**2)
                                                            for mobility */;
            }

            if(model->SOI3substrateDopingGiven)
            { /* work everything out */
                if(model->SOI3substrateDoping*1e6 /*(cm**3/m**3)*/ >1.45e16)
                {
                    if(!model->SOI3phiGiven)
                    {
                        model->SOI3phi = 2*vtnom*
                                log(model->SOI3substrateDoping*
                                1e6/*(cm**3/m**3)*//1.45e16);
                        model->SOI3phi = MAX(0.1,model->SOI3phi);
                    }
/* Now that we have ascertained both the doping        *
 * and the body film thickness, check to see           *
 * if we have a thick film device.  If not, complain ! */
                    xd_max=2*sqrt((2* 11.7 * 8.854214871e-12 * model->SOI3phi)/
                        (CHARGE*1e6 /*(cm**3/m**3)*/ * model->SOI3substrateDoping));

                    if(model->SOI3bodyThickness < xd_max)
                    {
						  		SPfrontEnd->IFerrorf (ERR_WARNING,
                    			"%s: Body Film thickness may be too small \nfor this model to be valid",
               				model->SOI3modName);
                    		/* return(E_PAUSE); don't want to stop,
                    		just issue a warning */
                    }
/* End of thick film check - msll 21/2/94
   Changed to only give warning  - msll 31/10/95  */
                    if(!model->SOI3vfbFGiven)
                    {
                       if(!model->SOI3frontFixedChargeDensityGiven)
                          model->SOI3frontFixedChargeDensity = 0;
                       fermis = model->SOI3type * 0.5 * model->SOI3phi;
                       wkfngf = 3.2;
                       if(!model->SOI3gateTypeGiven) model->SOI3gateType=1;
                       if(model->SOI3gateType != 0)
                       {
                          fermig = model->SOI3type *model->SOI3gateType*0.5*egfet1;
                          wkfngf = 3.25 + 0.5 * egfet1 - fermig;
                       }
                       wkfngfs = wkfngf - (3.25 + 0.5 * egfet1 +fermis);

		      /* Fixed oxide charge is normally +ve for both
		         n and p-channel, so need -ve voltage to neutralise it */
                       model->SOI3vfbF = wkfngfs -
                            model->SOI3frontFixedChargeDensity *
                            1e4 /*(cm**2/m**2)*/ *
                            CHARGE/model->SOI3frontOxideCapFactor;
                    }
                    if(!model->SOI3vfbBGiven)
                    {
                       wkfngbs = (1-model->SOI3type)*0.5*model->SOI3phi;
                       /* assume p-sub */
                       model->SOI3vfbB = wkfngbs -
                            model->SOI3backFixedChargeDensity *
                            1e4 /*(cm**2/m**2)*/ *
                            CHARGE/model->SOI3backOxideCapFactor;
                    }

                    if(!model->SOI3gammaGiven)
                    {
                        model->SOI3gamma = sqrt(2 * 11.70 * 8.854214871e-12 *
                        CHARGE * model->SOI3substrateDoping *
                        1e6 /*(cm**3/m**3)*/)/model->SOI3frontOxideCapFactor;
                    }

                    if(!model->SOI3gammaBGiven)
                    {
                        model->SOI3gammaB = sqrt(2 * 11.70 * 8.854214871e-12 *
                        CHARGE * model->SOI3substrateDoping *
                        1e6 /*(cm**3/m**3)*/)/model->SOI3backOxideCapFactor;
                    }
                    if(model->SOI3vt0Given)
                    { /* NSUB given AND VT0 given - change vfbF */
                       model->SOI3vfbF = model->SOI3vt0 - model->SOI3type *
                                         (eta_s*model->SOI3phi +
                                          model->SOI3gamma*sqrt(model->SOI3phi));
                    }
                    else
                    {
                       if(model->SOI3vtexGiven)
                       { /* JimB - Improved threshold voltage conversion model. */
                          if (model->SOI3delta0 < 0)
                          {
                             model->SOI3delta0 = 0;
                          }
                          if (model->SOI3vdex < 0)
                          {
                             /* Use convention that Vd at which Vtex was extracted */
                             /* is always +ve. */
                             model->SOI3vdex = -model->SOI3vdex;
                          }
                          /* Exponential term delta*phiF/phit (SOI3phi = 2phiF) */
                          Edelta0 = exp(MIN(MAX_EXP_ARG,
                          				(model->SOI3delta0*model->SOI3phi)/(2*vtnom))
                           		   );
                          /* Modified surface potential term (2+delta)*phiF + vdex/2 */
                          /* (SOI3phi = 2phiF) */
                          psi_delta0 = ((2+model->SOI3delta0)*model->SOI3phi/2) +
                                       model->SOI3vdex/2;
                          model->SOI3vfbF = model->SOI3vtex - model->SOI3type *
                          						  (eta_s*psi_delta0 + model->SOI3gamma*
                                            sqrt(psi_delta0 + vtnom*Edelta0)
                                            );
                       }
                    }
                }
                else /* Substrate doping less than intrinsic silicon, so set to zero. */
                {
                    model->SOI3substrateDoping = 0;
                    SPfrontEnd->IFerrorf (ERR_FATAL,
                            "%s: Nsub < Ni", model->SOI3modName);
                    return(E_BADPARM);
                }
            }
            else /* NSUB not given, have to assume that VT0, PHI and GAMMA are given */
            {
               xd_max=(2* 11.7 * 8.854214871e-12*sqrt(model->SOI3phi))/
               		(model->SOI3gamma*model->SOI3frontOxideCapFactor);
               if(model->SOI3bodyThickness < xd_max)
               {
					   SPfrontEnd->IFerrorf (ERR_WARNING,
                  "%s :Body Film thickness may be too small \nfor this model to be valid",
                  model->SOI3modName);
                  /* return(E_PAUSE); */
					}
/* End of thick film check - msll 21/2/94
   Changed to only give warning  - msll 31/10/95  */

   				/* If vtext given in netlist, but no vt0. */
   				if( (model->SOI3vtexGiven) && (!model->SOI3vt0Given) )
               { /* JimB - Improved threshold voltage conversion model. */
                  if (model->SOI3delta0 < 0)
                  {
                     model->SOI3delta0 = 0;
                  }
                  if (model->SOI3vdex < 0)
                  {
                     /* Use convention that Vd at which Vtex was extracted */
                     /* is always +ve. */
                     model->SOI3vdex = -model->SOI3vdex;
                  }
                  /* Exponential term delta*phiF/phit (SOI3phi = 2phiF) */
                  Edelta0 = exp(MIN(MAX_EXP_ARG,
                  			 (model->SOI3delta0*model->SOI3phi)/(2*vtnom))
                            );
                  /* Modified surface potential term (2+delta)*phiF + vdex/2 */
                  /* (SOI3phi = 2phiF) */
                  psi_delta0 = ((2+model->SOI3delta0)*model->SOI3phi/2) +
                  					model->SOI3vdex/2;
                  model->SOI3vfbF = model->SOI3vtex - model->SOI3type *
                  						(eta_s*psi_delta0 + model->SOI3gamma*
                                    sqrt(psi_delta0 + vtnom*Edelta0)
                                    );
               }
               else /* If no vtex, then use vt0, either netlist or default value. */
               { /* Use standard threshold voltage model. */
   				   model->SOI3vfbF = model->SOI3vt0 - model->SOI3type *
               						   (eta_s*model->SOI3phi + model->SOI3gamma*sqrt(model->SOI3phi));
               }
               if (!model->SOI3vfbBGiven)
               {
   				   model->SOI3vfbB = 0; /* NSUB not given, vfbB not given */
               }
            }
        }
        if((model->SOI3vsatGiven)&&(model->SOI3vsat != 0))
        {
          model->SOI3TVF0 = 0.8*exp(model->SOI3tnom/600);
        }
        else
        {
          model->SOI3TVF0 = 0;
        }


        /* loop through all instances of the model */
        for(here = SOI3instances(model); here!= NULL;
                here = SOI3nextInstance(here))
        {

            double czbd;    /* zero voltage bulk-drain capacitance */
            double czbs;    /* zero voltage bulk-source capacitance */
            double cj0;     /* default value of zero voltage bulk-source/drain capacitance*/
            double Nratio;  /* ratio of Nsub*Nplus/Nsub+Nplus */
            double arg;     /* 1 - fc */
            double sarg;    /* (1-fc) ^^ (-mj) */

            /* perform the parameter defaulting */
            
	    /* JimB - if device temperature not given, OR, if self-heating switched */
            /* on, then set device temperature equal to circuit temperature.  Can't */
            /* set device temp with self-heating on, otherwise get multiple thermal */
            /* ground nodes, but doesn't matter, since any sizeable thermal gradient*/
            /* across an IC circuit is probably due to self-heating anyway.         */
            if ( (!here->SOI3tempGiven) || (here->SOI3rt != 0) )
            {
               here->SOI3temp = ckt->CKTtemp;
            }
            vt = here->SOI3temp * CONSTKoverQ;
            ratio = here->SOI3temp/model->SOI3tnom;
            fact2 = here->SOI3temp/REFTEMP;
            kt = here->SOI3temp * CONSTboltz;
            egfet = 1.16-(7.02e-4*here->SOI3temp*here->SOI3temp)/
                    (here->SOI3temp+1108);
            if (!model->SOI3chidGiven)
            {
               model->SOI3chid = CHARGE*egfet/CONSTboltz;
            }
            if (!model->SOI3chid1Given)
            {
               model->SOI3chid1 = CHARGE*egfet/CONSTboltz;
            }
            arg = -egfet/(kt+kt)+1.1150877/(CONSTboltz*(REFTEMP+REFTEMP));
            pbfact = -2*vt *(1.5*log(fact2)+CHARGE*arg);

            if(!here->SOI3lGiven)
            {
               here->SOI3l = ckt->CKTdefaultMosL;
            }
            if(!here->SOI3wGiven)
            {
               here->SOI3w = ckt->CKTdefaultMosW;
            }

            if(here->SOI3l - 2 * model->SOI3latDiff <=0)
            {
					SPfrontEnd->IFerrorf (ERR_WARNING,
               	"%s: Effective channel length less than zero \nIncreasing \
                  this instance length by 2*LD to remove effect of LD",
                  here->SOI3name);
					here->SOI3l += 2*model->SOI3latDiff;
            }
            ratio4 =  exp(model->SOI3k*log(ratio)); /* ratio4 = (temp/tnom)^k */
                                                    /* i.e. mobilitites prop to */
                                                    /* T^-(k) where k=1.5 for old SPICE */
            here->SOI3tTransconductance = model->SOI3transconductance / ratio4;
            here->SOI3tSurfMob = model->SOI3surfaceMobility/ratio4;
            phio= (model->SOI3phi-pbfact1)/fact1; /* this is PHI @ REFTEMP */
            here->SOI3tPhi = fact2 * phio + pbfact;
            here->SOI3tVfbF = model->SOI3vfbF +
                  (model->SOI3type * model->SOI3gateType * 0.5*(egfet1-egfet)) +
                 (model->SOI3type * 0.5 * (model->SOI3phi - here->SOI3tPhi));
            here->SOI3tVfbB =  model->SOI3vfbB +
                                (1-model->SOI3type)*0.5*(here->SOI3tPhi-model->SOI3phi);

            here->SOI3tVto = here->SOI3tVfbF + model->SOI3type *
                    (model->SOI3gamma * sqrt(here->SOI3tPhi) + eta_s*here->SOI3tPhi);

            here->SOI3tSatCur = model->SOI3jctSatCur*
                    exp(-egfet/vt+egfet1/vtnom);
            here->SOI3tSatCur1 = model->SOI3jctSatCur1*
                    exp(-egfet/vt+egfet1/vtnom);
            here->SOI3tSatCurDens = model->SOI3jctSatCurDensity *
                    exp(-egfet/vt+egfet1/vtnom);
            here->SOI3tSatCurDens1 = model->SOI3jctSatCurDensity1 *
                    exp(-egfet/vt+egfet1/vtnom);

            pbo = (model->SOI3bulkJctPotential - pbfact1)/fact1;
            gmaold = (model->SOI3bulkJctPotential-pbo)/pbo;
            capfact = 1/(1+model->SOI3bulkJctSideGradingCoeff*
                    (4e-4*(model->SOI3tnom-REFTEMP)-gmaold));
            here->SOI3tCbd = model->SOI3capBD * capfact;
            here->SOI3tCbs = model->SOI3capBS * capfact;
            here->SOI3tCjsw = model->SOI3sideWallCapFactor * capfact;
            here->SOI3tBulkPot = fact2 * pbo+pbfact;
            gmanew = (here->SOI3tBulkPot-pbo)/pbo;
            capfact = (1+model->SOI3bulkJctSideGradingCoeff*
                    (4e-4*(here->SOI3temp-REFTEMP)-gmanew));
            here->SOI3tCbd *= capfact;
            here->SOI3tCbs *= capfact;
            here->SOI3tCjsw *= capfact;
            here->SOI3tDepCap = model->SOI3fwdCapDepCoeff * here->SOI3tBulkPot;

            if (here->SOI3tSatCurDens == 0)
            {
               if (here->SOI3tSatCur == 0)
               {
                  here->SOI3sourceVcrit = here->SOI3drainVcrit =
                          vt*log(vt/(CONSTroot2*1.0e-15));
               }
               else
               {
                  here->SOI3sourceVcrit = here->SOI3drainVcrit =
                          vt*log(vt/(CONSTroot2*here->SOI3tSatCur));
               }
            }
            else
            {
               here->SOI3drainVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->SOI3tSatCurDens * (here->SOI3w)));
               here->SOI3sourceVcrit =
                        vt * log( vt / (CONSTroot2 *
                        here->SOI3tSatCurDens * (here->SOI3w)));
            }

            if(model->SOI3capBDGiven)
            {
               czbd = here->SOI3tCbd;
            }
            else
            {
               if(model->SOI3sideWallCapFactorGiven)
               {
                  czbd = here->SOI3tCjsw * (here->SOI3w*model->SOI3bodyThickness);
               }
               /* JimB - 2/1/99.  Calculate default value for Cj0 */
               /* using PN junction theory. */
               else
               {
                  Nratio = (1e6*model->SOI3nplusDoping * model->SOI3substrateDoping)/
                           (model->SOI3nplusDoping + model->SOI3substrateDoping);
                  cj0 = sqrt((Nratio * 11.7 * 8.854214871e-12 * CHARGE)/
                             (2 * here->SOI3tBulkPot));

                  /* JimB - temperature dependence code */
            		gmaold = (model->SOI3bulkJctPotential-pbo)/pbo;
            		capfact = 1/(1+model->SOI3bulkJctSideGradingCoeff*
                    		(4e-4*(model->SOI3tnom-REFTEMP)-gmaold));
                  cj0 *= capfact;
            		gmanew = (here->SOI3tBulkPot-pbo)/pbo;
            		capfact = (1+model->SOI3bulkJctSideGradingCoeff*
                    		(4e-4*(here->SOI3temp-REFTEMP)-gmanew));
            		cj0 *= capfact;

                  czbd = cj0 * (here->SOI3w*model->SOI3bodyThickness);
               }
            }

            arg = 1-model->SOI3fwdCapDepCoeff;
            sarg = exp( (-model->SOI3bulkJctSideGradingCoeff) * log(arg) );
            here->SOI3Cbd = czbd;
            here->SOI3f2d = czbd*(1-model->SOI3fwdCapDepCoeff*
                        (1+model->SOI3bulkJctSideGradingCoeff))*
                        sarg/arg;
            here->SOI3f3d = czbd * model->SOI3bulkJctSideGradingCoeff * sarg/arg /
                        here->SOI3tBulkPot;
            here->SOI3f4d = czbd*here->SOI3tBulkPot*(1-arg*sarg)/
                        (1-model->SOI3bulkJctSideGradingCoeff)
                    -here->SOI3f3d/2*
                        (here->SOI3tDepCap*here->SOI3tDepCap)
                    -here->SOI3tDepCap * here->SOI3f2d;

            if(model->SOI3capBSGiven)
            {
               czbs=here->SOI3tCbs;
            }
            else
            {
               if(model->SOI3sideWallCapFactorGiven)
               {
                  czbs=here->SOI3tCjsw * (here->SOI3w*model->SOI3bodyThickness);
               }
               /* JimB - 2/1/99.  Calculate default value for Cj0 */
               /* using PN junction theory. */
               else
               {
                  Nratio = (1e6*model->SOI3nplusDoping * model->SOI3substrateDoping)/
                           (model->SOI3nplusDoping + model->SOI3substrateDoping);
                  cj0 = sqrt((Nratio * 11.7 * 8.854214871e-12 * CHARGE)/
                             (2 * here->SOI3tBulkPot));

                  /* JimB - temperature dependence code */
            		gmaold = (model->SOI3bulkJctPotential-pbo)/pbo;
            		capfact = 1/(1+model->SOI3bulkJctSideGradingCoeff*
                    		(4e-4*(model->SOI3tnom-REFTEMP)-gmaold));
                  cj0 *= capfact;
            		gmanew = (here->SOI3tBulkPot-pbo)/pbo;
            		capfact = (1+model->SOI3bulkJctSideGradingCoeff*
                    		(4e-4*(here->SOI3temp-REFTEMP)-gmanew));
            		cj0 *= capfact;

                  czbs = cj0 * (here->SOI3w*model->SOI3bodyThickness);
               }
            }
            arg = 1-model->SOI3fwdCapDepCoeff;
            sarg = exp( (-model->SOI3bulkJctSideGradingCoeff) * log(arg) );
            here->SOI3Cbs = czbs;
            here->SOI3f2s = czbs*(1-model->SOI3fwdCapDepCoeff*
                        (1+model->SOI3bulkJctSideGradingCoeff))*
                        sarg/arg;
            here->SOI3f3s = czbs * model->SOI3bulkJctSideGradingCoeff * sarg/arg /
                        here->SOI3tBulkPot;
            here->SOI3f4s = czbs*here->SOI3tBulkPot*(1-arg*sarg)/
                        (1-model->SOI3bulkJctSideGradingCoeff)
                    -here->SOI3f3s/2*
                        (here->SOI3tDepCap*here->SOI3tDepCap)
                    -here->SOI3tDepCap * here->SOI3f2s;

            if(model->SOI3drainResistanceGiven)
            {
               if(model->SOI3drainResistance != 0)
               {
                  here->SOI3drainConductance = 1/model->SOI3drainResistance;
               }
               else
               {
                  here->SOI3drainConductance = 0;
               }
            }
            else if (model->SOI3sheetResistanceGiven)
            {
               if(model->SOI3sheetResistance != 0)
               {
                  here->SOI3drainConductance =
                        1/(model->SOI3sheetResistance*here->SOI3drainSquares);
               }
               else
               {
                  here->SOI3drainConductance = 0;
               }
            }
            else if(model->SOI3rdwGiven)
            {
               if (model->SOI3rdw != 0)
               {
                	/* JimB - 1e6 multiplying factor converts W from m to microns */
                  here->SOI3drainConductance =
                    (here->SOI3w/model->SOI3rdw)*1e6;
               }
               else
               {
                  here->SOI3drainConductance = 0;
               }
            }
            else
            {
                here->SOI3drainConductance = 0;
            }
            
            if(model->SOI3sourceResistanceGiven)
            {
               if(model->SOI3sourceResistance != 0)
               {
                  here->SOI3sourceConductance = 1/model->SOI3sourceResistance;
               }
               else
               {
                  here->SOI3sourceConductance = 0;
               }
            }
            else if (model->SOI3sheetResistanceGiven)
            {
               if(model->SOI3sheetResistance != 0)
               {
                  here->SOI3sourceConductance =
                        1/(model->SOI3sheetResistance*here->SOI3sourceSquares);
               }
               else
               {
                  here->SOI3sourceConductance = 0;
               }
            }
            else if(model->SOI3rswGiven)
            {
               if (model->SOI3rsw != 0)
               {
                	/* JimB - 1e6 multiplying factor converts W from m to microns */
                  here->SOI3sourceConductance =
                    (here->SOI3w/model->SOI3rsw)*1e6;
               }
               else
               {
                  here->SOI3sourceConductance = 0;
               }
            }
            else
            {
               here->SOI3sourceConductance = 0;
            }
/* extra stuff for newer model - msll Jan96 */
        } /* finish looping through all instances of the model*/
    } /* finish looping through all the transistor models */
    return(OK);
}




