/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White,
						 and Craig Easson.

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
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SOI3setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    SOI3model *model = (SOI3model *)inModel;
    SOI3instance *here;
    int error;
    CKTnode *tmp;

    /* JimB - new variable for RT and CT scaling */
    double thermal_area;

    double rtargs[5];
    double * rtptr;
    int node_count;
  
     CKTnode *tmpNode;
     IFuid tmpName;
	 
	 

    /****** Part 1 - set any model parameters that are not present in ******/
    /****** the netlist to default values.                            ******/

    /*  loop through all the SOI3 device models */
    for( ; model != NULL; model = SOI3nextModel(model)) {

        if(!model->SOI3typeGiven) {
            model->SOI3type = NSOI3;
        }
        if(!model->SOI3latDiffGiven) {
            model->SOI3latDiff = 0;
        }
        if(!model->SOI3jctSatCurDensityGiven) {
            model->SOI3jctSatCurDensity = 1.0e-10;
        }
        if(!model->SOI3jctSatCurDensity1Given) {
            model->SOI3jctSatCurDensity1 = 0.0;
        }
        if(!model->SOI3jctSatCurGiven) {
            model->SOI3jctSatCur = 0.0;
        }
        if(!model->SOI3jctSatCur1Given) {
            model->SOI3jctSatCur1 = 0.0;
        }
        if(!model->SOI3transconductanceGiven) {
            model->SOI3transconductance = 2e-5;
        }
        if(!model->SOI3frontGateSourceOverlapCapFactorGiven) {
            model->SOI3frontGateSourceOverlapCapFactor = 0;
        }
        if(!model->SOI3frontGateDrainOverlapCapFactorGiven) {
            model->SOI3frontGateDrainOverlapCapFactor = 0;
        }
        if(!model->SOI3frontGateBulkOverlapCapFactorGiven) {
            model->SOI3frontGateBulkOverlapCapFactor = 0;
        }
        if(!model->SOI3backGateSourceOverlapCapAreaFactorGiven) {
            model->SOI3backGateSourceOverlapCapAreaFactor = 0;
        }
        if(!model->SOI3backGateDrainOverlapCapAreaFactorGiven) {
            model->SOI3backGateDrainOverlapCapAreaFactor = 0;
        }
        if(!model->SOI3backGateBulkOverlapCapAreaFactorGiven) {
            model->SOI3backGateBulkOverlapCapAreaFactor = 0;
        }
        if(!model->SOI3sideWallCapFactorGiven) {
            model->SOI3sideWallCapFactor = 0;
        }
        if(!model->SOI3bulkJctPotentialGiven) {
            model->SOI3bulkJctPotential = 0.8;
        }
        if(!model->SOI3bulkJctSideGradingCoeffGiven) {
            model->SOI3bulkJctSideGradingCoeff = 0.5;
        }
        if(!model->SOI3fwdCapDepCoeffGiven) {
            model->SOI3fwdCapDepCoeff = 0.5;
        }                                                    
        if(!model->SOI3lambdaGiven) {
            model->SOI3lambda = 0;
        }
        if(!model->SOI3thetaGiven) {
            model->SOI3theta = 0;
        }
        /* JimB - If SiO2 thermal conductivity given in netlist then use */
        /* that value, otherwise use literature value. (Units W/K*m). */
        if(!model->SOI3oxideThermalConductivityGiven) {
            model->SOI3oxideThermalConductivity = 1.4;
        }
        /* JimB - If Si specific heat given in netlist then use that value, */
        /* otherwise use literature value. (Units J/kg*K). */
        if(!model->SOI3siliconSpecificHeatGiven) {
            model->SOI3siliconSpecificHeat = 700;
        }
        /* JimB - If density of Si given in netlist then use that value, */
        /* otherwise use literature value. (kg/m^3). */
        if(!model->SOI3siliconDensityGiven) {
            model->SOI3siliconDensity = 2330;
        }
        if(!model->SOI3frontFixedChargeDensityGiven) {
            model->SOI3frontFixedChargeDensity = 0;
        }
        if(!model->SOI3backFixedChargeDensityGiven) {
            model->SOI3backFixedChargeDensity = 0;
        }
        if(!model->SOI3frontSurfaceStateDensityGiven) {
            model->SOI3frontSurfaceStateDensity = 0;
        }
        if(!model->SOI3backSurfaceStateDensityGiven) {
            model->SOI3backSurfaceStateDensity = 0;
        }
        if(!model->SOI3gammaGiven) {
            model->SOI3gamma = 0;
        }
        if(!model->SOI3fNcoefGiven) {
            model->SOI3fNcoef = 0;
        }
        if(!model->SOI3fNexpGiven) {
            model->SOI3fNexp = 1;
        }
/* extra stuff for newer model - msll Jan96 */
        if(!model->SOI3sigmaGiven) {
            model->SOI3sigma = 0;
        }
        if(!model->SOI3chiFBGiven) {
            model->SOI3chiFB = 0;
        }
        if(!model->SOI3chiPHIGiven) {
            model->SOI3chiPHI = 0;
        }
        if(!model->SOI3deltaWGiven) {
            model->SOI3deltaW = 0;
        }
        if(!model->SOI3deltaLGiven) {
            model->SOI3deltaL = 0;
        }
        if(!model->SOI3vsatGiven) {
            model->SOI3vsat = 0; /* special case - must check for it */
        }
        if(!model->SOI3kGiven) {
            model->SOI3k = 1.5; /* defaults to old SPICE value */
        }
        if(!model->SOI3lxGiven) {
            model->SOI3lx = 0;
        }
        if(!model->SOI3vpGiven) {
            model->SOI3vp = 0;
        }
        if(!model->SOI3gammaBGiven) {
            model->SOI3gammaB = 0;
        }
        if(!model->SOI3etaGiven) {
            model->SOI3eta = 1.0; /* normal field for imp. ion. */
        }
        if(!model->SOI3alpha0Given) {
            model->SOI3alpha0=0;
        }
        if(!model->SOI3beta0Given) {
            model->SOI3beta0=1.92e6;
        }
        if(!model->SOI3lmGiven) {
            model->SOI3lm = 0;
        }
        if(!model->SOI3lm1Given) {
            model->SOI3lm1 = 0;
        }
        if(!model->SOI3lm2Given) {
            model->SOI3lm2 = 0;
        }
        if((!model->SOI3etadGiven) || (model->SOI3etad == 0 )) {
            model->SOI3etad = 1.0;
        }
        if((!model->SOI3etad1Given) || (model->SOI3etad1 == 0 )) {
            model->SOI3etad1 = 1.0;
        }
        if(!model->SOI3chibetaGiven) {
            model->SOI3chibeta = 0.0;
        }
        if(!model->SOI3dvtGiven) {
            model->SOI3dvt = 1;
        }
        if(!model->SOI3nLevGiven) {
            model->SOI3nLev = 0;
        }
        if(!model->SOI3betaBJTGiven) {
            model->SOI3betaBJT = 0.0;
        }
        if(!model->SOI3tauFBJTGiven) {
            model->SOI3tauFBJT = 0.0;
        }
        if(!model->SOI3tauRBJTGiven) {
            model->SOI3tauRBJT = 0.0;
        }
        if(!model->SOI3betaEXPGiven) {
            model->SOI3betaEXP = 2.0;
        }
        if(!model->SOI3tauEXPGiven) {
            model->SOI3tauEXP = 0.0;
        }
        if(!model->SOI3rswGiven) {
            model->SOI3rsw = 0.0;
        }
        if(!model->SOI3rdwGiven) {
            model->SOI3rdw = 0.0;
        }
        if(!model->SOI3minimumFeatureSizeGiven) {
            model->SOI3minimumFeatureSize = 0.0;
        }
        if(!model->SOI3vtexGiven) {
            model->SOI3vtex = 0.0;
        }
        if(!model->SOI3vdexGiven) {
            model->SOI3vdex = 0.0;
        }
        if(!model->SOI3delta0Given) {
            model->SOI3delta0 = 0.0;
        }
        if(!model->SOI3satChargeShareFactorGiven) {
            model->SOI3satChargeShareFactor = 0.5;
        }
        if(!model->SOI3nplusDopingGiven) {
            model->SOI3nplusDoping = 1e20;
        }
        if(!model->SOI3rtaGiven) {
            model->SOI3rta = 0;
        }
        if(!model->SOI3ctaGiven) {
            model->SOI3cta = 0;
        }
        if(!model->SOI3mexpGiven) {
            model->SOI3mexp = 0;
        }

        /* now check to determine which CLM model to use */
        if((model->SOI3lx != 0) && (model->SOI3lambda != 0))
        {

            SPfrontEnd->IFerrorf (ERR_WARNING,
             "%s: Non-zero values for BOTH LAMBDA and LX. \nDefaulting to simple LAMBDA model",
                        model->SOI3modName);

				model->SOI3useLAMBDA = TRUE;
        }

        /* if only lx given, AND vp!=0, AND mexp (integer) is at least 1, use lx/vp, else basic lambda model*/
        if ((model->SOI3lxGiven) && (model->SOI3lx != 0) &&
            (!model->SOI3lambdaGiven) && (model->SOI3vp != 0) && (model->SOI3mexp > 0))
        {
           model->SOI3useLAMBDA = FALSE;
        }
        else
        {
           model->SOI3useLAMBDA = TRUE;
        }


        /****** Part 2 - set any instance parameters that are not present ******/
        /****** in the netlist to default values.                         ******/

        /* loop through all the instances of the model */
        for (here = SOI3instances(model); here != NULL ;
                here=SOI3nextInstance(here)) {


            if(!here->SOI3icVBSGiven) {
                here->SOI3icVBS = 0;
            }
            if(!here->SOI3icVDSGiven) {
                here->SOI3icVDS = 0;
            }
            if(!here->SOI3icVGFSGiven) {
                here->SOI3icVGFS = 0;
            }
            if(!here->SOI3icVGBSGiven) {
                here->SOI3icVGBS = 0;
            }
            if(!here->SOI3drainSquaresGiven || here->SOI3drainSquares==0) {
                here->SOI3drainSquares=1;
            }
            if(!here->SOI3sourceSquaresGiven || here->SOI3sourceSquares==0) {
                here->SOI3sourceSquares=1;
            }

	     if (!here->SOI3mGiven)
                here->SOI3m = 1;

            /****** Part 3 - Initialise transconductances. ******/

            /* initialise gM's */
            here->SOI3iMdb= 0.0;
            here->SOI3iMsb= 0.0;
            here->SOI3gMmbs = 0.0;
            here->SOI3gMmf = 0.0;
            here->SOI3gMmb = 0.0;
            here->SOI3gMd = 0.0;
            here->SOI3gMdeltaT = 0.0;
            
            /* allocate a chunk of the state vector */
            here->SOI3states = *states;
            *states += SOI3numStates;
/*               if(ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & TRANSEN) ){
                   *states += SOI3numSenStates * (ckt->CKTsenInfo->SENparms);
               }
*/
				/****** Part 4 - check resistance values for internal nodes, ******/
            /****** to see which internal nodes need to be created.      ******/

            /* Start with internal source and drain nodes */

            if((model->SOI3drainResistance != 0
                    || (model->SOI3sheetResistance != 0 &&
                        here->SOI3drainSquares != 0)
                    || model->SOI3rdw != 0)
                    && here->SOI3dNodePrime == 0)
            {
                error = CKTmkVolt(ckt,&tmp,here->SOI3name,"drain");

                if(error)
					 {
                    return(error);
                }
                here->SOI3dNodePrime = tmp->number;
		
		 if (ckt->CKTcopyNodesets) {
		   if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }

            }
            else
            {
                here->SOI3dNodePrime = here->SOI3dNode;
            }

            if((model->SOI3sourceResistance != 0 ||
                    (model->SOI3sheetResistance != 0 &&
                     here->SOI3sourceSquares != 0) ||
                    model->SOI3rsw != 0) &&
                    here->SOI3sNodePrime==0)
            {
                error = CKTmkVolt(ckt,&tmp,here->SOI3name,"source");

                if(error)
					 {
                    return(error);
                }
                here->SOI3sNodePrime = tmp->number;
		 
		 if (ckt->CKTcopyNodesets) {
		   if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }

		
            }
            else
            {
                here->SOI3sNodePrime = here->SOI3sNode;
            }

            /* Now for thermal node */

            /* JimB - If minimum feature size has non-zero value, then this */
            /* will be used to calculate thermal area of SiO2 - this gives  */
            /* more accurate values of RT for short-channel devices.  Assume*/
            /* 4*fmin added to L, and 2*fmin added to W (fmin in microns).  */
				thermal_area = (here->SOI3w + 2*1e-6*model->SOI3minimumFeatureSize)
                  					* (here->SOI3l + 4*1e-6*model->SOI3minimumFeatureSize);

            /* Now calculate RT and CT. */

            /* If RT is given on instance line, use it, otherwise calculate from */
            /* above variables. */
            if (!here->SOI3rtGiven)
            {
               if (model->SOI3rtaGiven)
               {
            	   here->SOI3rt = model->SOI3rta/thermal_area;
               }
               else
               {
                  if (model->SOI3oxideThermalConductivity != 0)
					   {
						   here->SOI3rt = model->SOI3backOxideThickness /
                  	   			   (model->SOI3oxideThermalConductivity * thermal_area);
                  }
                  else
                  /* If conductivity set to zero in netlist, switch off self-heating. */
                  {
                     here->SOI3rt = 0;
                  }
               }
            }
            if (!here->SOI3rt1Given)
            {
              here->SOI3rt1=0;
            }
            if (!here->SOI3rt2Given)
            {
              here->SOI3rt2=0;
            }
            if (!here->SOI3rt3Given)
            {
              here->SOI3rt3=0;
            }
            if (!here->SOI3rt4Given)
            {
              here->SOI3rt4=0;
            }

				/* Now same thing for CT, but less complex as CT=0 does not cause problems */
            if (!here->SOI3ctGiven)
            {
               if (model->SOI3ctaGiven)
               {
            	   here->SOI3ct = model->SOI3cta*thermal_area;
               }
               else
               {
                  here->SOI3ct = model->SOI3siliconDensity * model->SOI3siliconSpecificHeat *
               					   thermal_area * model->SOI3bodyThickness;
               }
            }
            if (!here->SOI3ct1Given)
            {
              here->SOI3ct1=0;
            }
            if (!here->SOI3ct2Given)
            {
              here->SOI3ct2=0;
            }
            if (!here->SOI3ct3Given)
            {
              here->SOI3ct3=0;
            }
            if (!here->SOI3ct4Given)
            {
              here->SOI3ct4=0;
            }
            
            /* JimB - 15/9/99 */
            rtargs[0]=here->SOI3rt;
            rtargs[1]=here->SOI3rt1;
            rtargs[2]=here->SOI3rt2;
            rtargs[3]=here->SOI3rt3;
            rtargs[4]=here->SOI3rt4;
	 			rtptr = rtargs; /* Set pointer to start address of rtargs array. */
            node_count=0;

            while ( (*rtptr) && (node_count<5) )
				{
               node_count++;
               if (node_count<5)
               {
               	rtptr++; /* Increment pointer to next array element. */
               }
            }
            here->SOI3numThermalNodes=node_count;

				/* Thermal node is now external and so is automatically created by CKTcreate in INP2A.
   			It is also bound to the created node's number.  However, if rt=0 then no thermal so
   			make tout be the thermal ground.  Can't simply use CKTbindNode 'cos the row and column
   			associated with the original node has already been created.  Thus problems will occur
   			during pivoting.  Instead put zero voltage source here.  First create branch for it.*/

            if ((here->SOI3rt == 0) && (here->SOI3branch == 0))
            {
               error = CKTmkCur(ckt,&tmp,here->SOI3name,"branch");

					if(error)
					{
               	return(error);
					}
               here->SOI3branch = tmp->number;
            }
            else
            { /* have thermal - now how many time constants ? */
              if ((here->SOI3numThermalNodes > 1) &&
                  (here->SOI3tout1Node == 0))
              {
                error = CKTmkVolt(ckt,&tmp,here->SOI3name,"tout1");
                if (error) return (error);
                here->SOI3tout1Node = tmp->number;
              }
              else
              {
                here->SOI3tout1Node = 0;
              }
              if ((here->SOI3numThermalNodes > 2) &&
                  (here->SOI3tout2Node == 0))
              {
                error = CKTmkVolt(ckt,&tmp,here->SOI3name,"tout2");
                if (error) return (error);
                here->SOI3tout2Node = tmp->number;
              }
              else
              {
                here->SOI3tout2Node = 0;
              }
              if ((here->SOI3numThermalNodes > 3) &&
                  (here->SOI3tout3Node == 0))
              {
                error = CKTmkVolt(ckt,&tmp,here->SOI3name,"tout3");
                if (error) return (error);
                here->SOI3tout3Node = tmp->number;
              }
              else
              {
                here->SOI3tout3Node = 0;
              }
              if ((here->SOI3numThermalNodes > 4) &&
                  (here->SOI3tout4Node == 0))
              {
                error = CKTmkVolt(ckt,&tmp,here->SOI3name,"tout4");
                if (error) return (error);
                here->SOI3tout4Node = tmp->number;
              }
              else
              {
                here->SOI3tout4Node = 0;
              }
            }


/****** Part 5 - allocate memory to matrix elements corresponding to ******/
/****** pairs of nodes.                                              ******/

/* macro to make elements with built in test for out of memory */

#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)


            TSTALLOC(SOI3D_dPtr,SOI3dNode,SOI3dNode);
            TSTALLOC(SOI3D_dpPtr,SOI3dNode,SOI3dNodePrime);
            TSTALLOC(SOI3DP_dPtr,SOI3dNodePrime,SOI3dNode);

            TSTALLOC(SOI3S_sPtr,SOI3sNode,SOI3sNode);
            TSTALLOC(SOI3S_spPtr,SOI3sNode,SOI3sNodePrime);
            TSTALLOC(SOI3SP_sPtr,SOI3sNodePrime,SOI3sNode);

            TSTALLOC(SOI3GF_gfPtr,SOI3gfNode,SOI3gfNode);
            TSTALLOC(SOI3GF_gbPtr,SOI3gfNode,SOI3gbNode);
            TSTALLOC(SOI3GF_dpPtr,SOI3gfNode,SOI3dNodePrime);
            TSTALLOC(SOI3GF_spPtr,SOI3gfNode,SOI3sNodePrime);
            TSTALLOC(SOI3GF_bPtr,SOI3gfNode,SOI3bNode);

            TSTALLOC(SOI3GB_gfPtr,SOI3gbNode,SOI3gfNode);
            TSTALLOC(SOI3GB_gbPtr,SOI3gbNode,SOI3gbNode);
            TSTALLOC(SOI3GB_dpPtr,SOI3gbNode,SOI3dNodePrime);
            TSTALLOC(SOI3GB_spPtr,SOI3gbNode,SOI3sNodePrime);
            TSTALLOC(SOI3GB_bPtr,SOI3gbNode,SOI3bNode);
            
            TSTALLOC(SOI3B_gfPtr,SOI3bNode,SOI3gfNode);
            TSTALLOC(SOI3B_gbPtr,SOI3bNode,SOI3gbNode);
            TSTALLOC(SOI3B_dpPtr,SOI3bNode,SOI3dNodePrime);
            TSTALLOC(SOI3B_spPtr,SOI3bNode,SOI3sNodePrime);
            TSTALLOC(SOI3B_bPtr,SOI3bNode,SOI3bNode);

            TSTALLOC(SOI3DP_gfPtr,SOI3dNodePrime,SOI3gfNode);
            TSTALLOC(SOI3DP_gbPtr,SOI3dNodePrime,SOI3gbNode);
            TSTALLOC(SOI3DP_dpPtr,SOI3dNodePrime,SOI3dNodePrime);
            TSTALLOC(SOI3DP_spPtr,SOI3dNodePrime,SOI3sNodePrime);
            TSTALLOC(SOI3DP_bPtr,SOI3dNodePrime,SOI3bNode);

            TSTALLOC(SOI3SP_gfPtr,SOI3sNodePrime,SOI3gfNode);
            TSTALLOC(SOI3SP_gbPtr,SOI3sNodePrime,SOI3gbNode);
            TSTALLOC(SOI3SP_dpPtr,SOI3sNodePrime,SOI3dNodePrime);
            TSTALLOC(SOI3SP_spPtr,SOI3sNodePrime,SOI3sNodePrime);
            TSTALLOC(SOI3SP_bPtr,SOI3sNodePrime,SOI3bNode);

            if (here->SOI3rt == 0)
            {
              TSTALLOC(SOI3TOUT_ibrPtr,SOI3toutNode,SOI3branch);
              TSTALLOC(SOI3IBR_toutPtr,SOI3branch,SOI3toutNode);
            }
            else
            {
              TSTALLOC(SOI3TOUT_toutPtr,SOI3toutNode,SOI3toutNode);
              if (here->SOI3numThermalNodes > 1)
              {
                TSTALLOC(SOI3TOUT_tout1Ptr,SOI3toutNode,SOI3tout1Node);
                TSTALLOC(SOI3TOUT1_toutPtr,SOI3tout1Node,SOI3toutNode);
                TSTALLOC(SOI3TOUT1_tout1Ptr,SOI3tout1Node,SOI3tout1Node);
              }
              if (here->SOI3numThermalNodes > 2)
              {
                TSTALLOC(SOI3TOUT1_tout2Ptr,SOI3tout1Node,SOI3tout2Node);
                TSTALLOC(SOI3TOUT2_tout1Ptr,SOI3tout2Node,SOI3tout1Node);
                TSTALLOC(SOI3TOUT2_tout2Ptr,SOI3tout2Node,SOI3tout2Node);
              }
              if (here->SOI3numThermalNodes > 3)
              {
                TSTALLOC(SOI3TOUT2_tout3Ptr,SOI3tout2Node,SOI3tout3Node);
                TSTALLOC(SOI3TOUT3_tout2Ptr,SOI3tout3Node,SOI3tout2Node);
                TSTALLOC(SOI3TOUT3_tout3Ptr,SOI3tout3Node,SOI3tout3Node);
              }
              if (here->SOI3numThermalNodes > 4)
              {
                TSTALLOC(SOI3TOUT3_tout4Ptr,SOI3tout3Node,SOI3tout4Node);
                TSTALLOC(SOI3TOUT4_tout3Ptr,SOI3tout4Node,SOI3tout3Node);
                TSTALLOC(SOI3TOUT4_tout4Ptr,SOI3tout4Node,SOI3tout4Node);
              }

              TSTALLOC(SOI3TOUT_toutPtr,SOI3toutNode,SOI3toutNode);
              TSTALLOC(SOI3TOUT_gfPtr,SOI3toutNode,SOI3gfNode);
              TSTALLOC(SOI3TOUT_gbPtr,SOI3toutNode,SOI3gbNode);
              TSTALLOC(SOI3TOUT_dpPtr,SOI3toutNode,SOI3dNodePrime);
              TSTALLOC(SOI3TOUT_spPtr,SOI3toutNode,SOI3sNodePrime);
              TSTALLOC(SOI3TOUT_bPtr,SOI3toutNode,SOI3bNode);

              TSTALLOC(SOI3GF_toutPtr,SOI3gfNode,SOI3toutNode);
              TSTALLOC(SOI3GB_toutPtr,SOI3gbNode,SOI3toutNode);
              TSTALLOC(SOI3DP_toutPtr,SOI3dNodePrime,SOI3toutNode);
              TSTALLOC(SOI3SP_toutPtr,SOI3sNodePrime,SOI3toutNode);
              TSTALLOC(SOI3B_toutPtr,SOI3bNode,SOI3toutNode);
            }
        }
    }
    return(OK);
}



int
SOI3unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model;
    SOI3instance *here;

    for (model = (SOI3model *)inModel; model != NULL;
	    model = SOI3nextModel(model))
    {
       for (here = SOI3instances(model); here != NULL;
                here=SOI3nextInstance(here))
		 {
	       if (here->SOI3tout4Node > 0)
		       CKTdltNNum(ckt, here->SOI3tout4Node);
               here->SOI3tout4Node = 0;

	       if (here->SOI3tout3Node > 0)
             CKTdltNNum(ckt, here->SOI3tout3Node);
               here->SOI3tout3Node = 0;

	       if (here->SOI3tout2Node > 0)
		       CKTdltNNum(ckt, here->SOI3tout2Node);
               here->SOI3tout2Node = 0;

	       if (here->SOI3tout1Node > 0)
             CKTdltNNum(ckt, here->SOI3tout1Node);
               here->SOI3tout1Node = 0;

	       if (here->SOI3branch > 0)
		       CKTdltNNum(ckt, here->SOI3branch);
               here->SOI3branch=0;

	       if (here->SOI3sNodePrime > 0
		       && here->SOI3sNodePrime != here->SOI3sNode)
		       CKTdltNNum(ckt, here->SOI3sNodePrime);
               here->SOI3sNodePrime= 0;

	       if (here->SOI3dNodePrime > 0
             && here->SOI3dNodePrime != here->SOI3dNode)
             CKTdltNNum(ckt, here->SOI3dNodePrime);
               here->SOI3dNodePrime= 0;

	    }
    }
    return OK;
}
