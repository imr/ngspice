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
#include "ngspice/cktdefs.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SOI3acLoad(GENmodel *inModel, CKTcircuit *ckt)
{
    SOI3model *model = (SOI3model*)inModel;
    SOI3instance *here;
    int xnrm;
    int xrev;

    double cgfgf,cgfd,cgfs,cgfdeltaT,cgfgb;
    double cdgf,cdd,cds,cddeltaT,cdgb;
    double csgf,csd,css,csdeltaT,csgb;
    double cbgf,cbd,cbs,cbdeltaT,cbgb;
    double cgbgf,cgbd,cgbs,cgbdeltaT,cgbgb;

    double xcgfgf,xcgfd,xcgfs,xcgfdeltaT,xcgfgb;
    double xcdgf,xcdd,xcds,xcddeltaT,xcdgb;
    double xcsgf,xcsd,xcss,xcsdeltaT,xcsgb;
    double xcbgf,xcbd,xcbs,xcbdeltaT,xcbgb;
    double xcgbgf,xcgbd,xcgbs,xcgbdeltaT,xcgbgb;

    double capbd,capbs; /* diode capacitances */

    double xcBJTbsbs,xcBJTbsdeltaT;
    double xcBJTbdbd,xcBJTbddeltaT;

    double rtargs[5];
    double grt[5];
    double ctargs[5];
    double xct[5]; /* 1/reactance of thermal cap */
    int tnodeindex;

    double cgfb0;
    double cgfd0;
    double cgfs0;

    double cgbb0;
    double cgbd0;
    double cgbs0;

    double EffectiveLength;

    double omega;

    double m;

    omega = ckt->CKTomega;

    for( ; model != NULL; model = SOI3nextModel(model))
    {
        for(here = SOI3instances(model); here!= NULL;
                here = SOI3nextInstance(here))
        {
	
            if (here->SOI3mode < 0)
            {
                xnrm=0;
                xrev=1;
            }
            else
            {
                xnrm=1;
                xrev=0;
            }

            EffectiveLength=here->SOI3l - 2*model->SOI3latDiff;

            /* JimB - can use basic device geometry to estimate front and back */
            /* overlap capacitances to a first approximation. Use this default */
            /* model if capacitance factors aren't given in model netlist. */

            /* Calculate front gate overlap capacitances. */

        		if(model->SOI3frontGateSourceOverlapCapFactorGiven)
            {
               cgfs0 = model->SOI3frontGateSourceOverlapCapFactor * here->SOI3w;
        		}
            else
            {
               cgfs0 = model->SOI3latDiff * here->SOI3w * model->SOI3frontOxideCapFactor;
            }
        		if(model->SOI3frontGateDrainOverlapCapFactorGiven)
            {
				   cgfd0 = model->SOI3frontGateDrainOverlapCapFactor * here->SOI3w;
        		}
            else
            {
               cgfd0 = model->SOI3latDiff * here->SOI3w * model->SOI3frontOxideCapFactor;
            }
        		if(model->SOI3frontGateBulkOverlapCapFactorGiven)
            {
               cgfb0 = model->SOI3frontGateBulkOverlapCapFactor * EffectiveLength;
            }
            else
            {
               cgfb0 = EffectiveLength * (0.1*1e-6*model->SOI3minimumFeatureSize)
               			* model->SOI3frontOxideCapFactor;
            }

            /* Calculate back gate overlap capacitances. */

        		if( (model->SOI3backGateSourceOverlapCapAreaFactorGiven) &&
            (!model->SOI3backGateSourceOverlapCapAreaFactor || here->SOI3asGiven) )
            {
               cgbs0 = model->SOI3backGateSourceOverlapCapAreaFactor * here->SOI3as;
        		}
            else
            {
               cgbs0 = (2*1e-6*model->SOI3minimumFeatureSize + model->SOI3latDiff) * here->SOI3w
               			* model->SOI3backOxideCapFactor;
            }
        		if( (model->SOI3backGateDrainOverlapCapAreaFactorGiven) &&
            (!model->SOI3backGateDrainOverlapCapAreaFactor || here->SOI3adGiven) )
            {
               cgbd0 = model->SOI3backGateDrainOverlapCapAreaFactor * here->SOI3ad;
        		}
            else
            {
               cgbd0 = (2*1e-6*model->SOI3minimumFeatureSize + model->SOI3latDiff) * here->SOI3w
               			* model->SOI3backOxideCapFactor;
            }
        		if( (model->SOI3backGateBulkOverlapCapAreaFactorGiven) &&
            (!model->SOI3backGateBulkOverlapCapAreaFactor || here->SOI3abGiven) )
            {
               cgbb0 = model->SOI3backGateBulkOverlapCapAreaFactor * here->SOI3ab;
            }
            else
            {
               cgbb0 = EffectiveLength * (0.1*1e-6*model->SOI3minimumFeatureSize + here->SOI3w)
               			* model->SOI3backOxideCapFactor;
            }

            capbd = here->SOI3capbd;
            capbs = here->SOI3capbs;

            cgfgf = *(ckt->CKTstate0 + here->SOI3cgfgf);
            cgfd  = *(ckt->CKTstate0 + here->SOI3cgfd);
            cgfs  = *(ckt->CKTstate0 + here->SOI3cgfs);
            cgfdeltaT  = *(ckt->CKTstate0 + here->SOI3cgfdeltaT);
            cgfgb = *(ckt->CKTstate0 + here->SOI3cgfgb);
            csgf = *(ckt->CKTstate0 + here->SOI3csgf);
            csd  = *(ckt->CKTstate0 + here->SOI3csd);
            css  = *(ckt->CKTstate0 + here->SOI3css);
            csdeltaT  = *(ckt->CKTstate0 + here->SOI3csdeltaT);
            csgb = *(ckt->CKTstate0 + here->SOI3csgb);
            cdgf = *(ckt->CKTstate0 + here->SOI3cdgf);
            cdd  = *(ckt->CKTstate0 + here->SOI3cdd);
            cds  = *(ckt->CKTstate0 + here->SOI3cds);
            cddeltaT  = *(ckt->CKTstate0 + here->SOI3cddeltaT);
            cdgb  = *(ckt->CKTstate0 + here->SOI3cdgb);
            cgbgf = *(ckt->CKTstate0 + here->SOI3cgbgf);
            cgbd  = *(ckt->CKTstate0 + here->SOI3cgbd);
            cgbs  = *(ckt->CKTstate0 + here->SOI3cgbs);
            cgbdeltaT  = *(ckt->CKTstate0 + here->SOI3cgbdeltaT);
            cgbgb = *(ckt->CKTstate0 + here->SOI3cgbgb);
            cbgf = -(cgfgf + cdgf + csgf + cgbgf);
            cbd = -(cgfd + cdd + csd + cgbd);
            cbs = -(cgfs + cds + css + cgbs);
            cbdeltaT = -(cgfdeltaT + cddeltaT + csdeltaT + cgbdeltaT);
            cbgb = -(cgfgb + cdgb + csgb + cgbgb);

				xcgfgf = (cgfgf + cgfd0 + cgfs0 + cgfb0) * omega;
				xcgfd  = (cgfd - cgfd0) * omega;
				xcgfs  = (cgfs - cgfs0) * omega;
				xcgfdeltaT = cgfdeltaT * omega;
				xcgfgb = cgfgb * omega;
				xcdgf = (cdgf - cgfd0) * omega;
				xcdd  = (cdd + capbd + cgfd0 + cgbd0) * omega;
				xcds  = cds * omega;
				xcddeltaT = cddeltaT * omega;
				xcdgb = (cdgb - cgbd0) * omega;
				xcsgf = (csgf - cgfs0) * omega;
				xcsd  = csd * omega;
				xcss  = (css + capbs + cgfs0 + cgbs0) * omega;
				xcsdeltaT = csdeltaT * omega;
				xcsgb = (csgb - cgbs0) * omega;
				xcbgf = (cbgf - cgfb0) * omega;
            xcbd = (cbd - capbd) * omega;
            xcbs = (cbs - capbs) * omega;
				xcbdeltaT = cbdeltaT * omega;
				xcbgb = (cbgb - cgbb0) * omega;
				xcgbgf = cgbgf * omega;
				xcgbd  = (cgbd - cgbd0) * omega;
				xcgbs  = (cgbs - cgbs0) * omega;
				xcgbdeltaT = cgbdeltaT * omega;
				xcgbgb = (cgbgb + cgbd0 + cgbs0 + cgbb0) * omega;

            xcBJTbsbs = *(ckt->CKTstate0 + here->SOI3cBJTbsbs) * omega;
            xcBJTbsdeltaT = *(ckt->CKTstate0 + here->SOI3cBJTbsdeltaT) * omega;
            xcBJTbdbd = *(ckt->CKTstate0 + here->SOI3cBJTbdbd) * omega;
            xcBJTbddeltaT = *(ckt->CKTstate0 + here->SOI3cBJTbddeltaT) * omega;

            /* JimB - 15/9/99 */
            /* Code for multiple thermal time constants.  Start by moving all */
            /* rt and ct constants into arrays. */
            rtargs[0]=here->SOI3rt;
            rtargs[1]=here->SOI3rt1;
            rtargs[2]=here->SOI3rt2;
            rtargs[3]=here->SOI3rt3;
            rtargs[4]=here->SOI3rt4;

            ctargs[0]=here->SOI3ct;
            ctargs[1]=here->SOI3ct1;
            ctargs[2]=here->SOI3ct2;
            ctargs[3]=here->SOI3ct3;
            ctargs[4]=here->SOI3ct4;

            /* Set all admittance components to zero. */
            grt[0]=grt[1]=grt[2]=grt[3]=grt[4]=0.0;
            xct[0]=xct[1]=xct[2]=xct[3]=xct[4]=0.0;
            /* Now calculate conductances and susceptances from rt and ct. */
            /* Don't need to worry about divide by zero when calculating */
            /* grt components, as soi3setup() only creates a thermal node */
            /* if corresponding rt is greater than zero. */
            for(tnodeindex=0;tnodeindex<here->SOI3numThermalNodes;tnodeindex++)
            {
				   xct[tnodeindex] = ctargs[tnodeindex] * ckt->CKTomega;
               grt[tnodeindex] = 1/rtargs[tnodeindex];
            }
            /* End JimB */

            /*
             *    load matrix
             */

            m = here->SOI3m;

            *(here->SOI3GF_gfPtr + 1) += m * xcgfgf;
            *(here->SOI3GF_gbPtr + 1) += m * xcgfgb;
            *(here->SOI3GF_dpPtr + 1) += m * xcgfd;
            *(here->SOI3GF_spPtr + 1) += m * xcgfs;
            *(here->SOI3GF_bPtr + 1)  -= m * (xcgfgf + xcgfd + xcgfs + xcgfgb);

            *(here->SOI3GB_gfPtr + 1) += m * xcgbgf;
            *(here->SOI3GB_gbPtr + 1) += m * xcgbgb;
            *(here->SOI3GB_dpPtr + 1) += m * xcgbd;
            *(here->SOI3GB_spPtr + 1) += m * xcgbs;
            *(here->SOI3GB_bPtr + 1)  -= m * (xcgbgf + xcgbd + xcgbs + xcgbgb);

            *(here->SOI3B_gfPtr + 1) += m * xcbgf;
            *(here->SOI3B_gbPtr + 1) += m * xcbgb;
            *(here->SOI3B_dpPtr + 1) += m * (xcbd-xcBJTbdbd);
            *(here->SOI3B_spPtr + 1) += m * (xcbs-xcBJTbsbs);
            *(here->SOI3B_bPtr + 1)  += m * (-(xcbgf+xcbd+xcbs+xcbgb)
                                        +xcBJTbsbs+xcBJTbdbd);

            *(here->SOI3DP_gfPtr + 1) += m * xcdgf;
            *(here->SOI3DP_gbPtr + 1) += m * xcdgb;
            *(here->SOI3DP_dpPtr + 1) += m * (xcdd+xcBJTbdbd);
            *(here->SOI3DP_spPtr + 1) += m * xcds;
            *(here->SOI3DP_bPtr + 1)  -= m * (xcdgf + xcdd + xcds + xcdgb + xcBJTbdbd);

            *(here->SOI3SP_gfPtr + 1) += m * xcsgf;
            *(here->SOI3SP_gbPtr + 1) += m * xcsgb;
            *(here->SOI3SP_dpPtr + 1) += m * xcsd;
            *(here->SOI3SP_spPtr + 1) += m * (xcss+xcBJTbsbs);
            *(here->SOI3SP_bPtr + 1)  -= m * (xcsgf + xcsd + xcss + xcsgb + xcBJTbsbs);
        
/* if no thermal behaviour specified, then put in zero valued indpt. voltage source
   between TOUT and ground */
            if (here->SOI3rt==0)
            {
            	*(here->SOI3TOUT_ibrPtr + 1) += m * 1.0;
   		*(here->SOI3IBR_toutPtr + 1) += m * 1.0;
            	*(ckt->CKTirhs + (here->SOI3branch)) = 0;
            }
            else
            {
            	*(here->SOI3TOUT_toutPtr + 1) += m * xct[0];
            	if (here->SOI3numThermalNodes > 1)
              	{
               	*(here->SOI3TOUT_tout1Ptr + 1) += m * (-xct[0]);
               	*(here->SOI3TOUT1_toutPtr + 1) += m * (-xct[0]);
               	*(here->SOI3TOUT1_tout1Ptr + 1) += m * (xct[0]+xct[1]);
              	}
              	if (here->SOI3numThermalNodes > 2)
              	{
               	*(here->SOI3TOUT1_tout2Ptr + 1) += m * (-xct[1]);
               	*(here->SOI3TOUT2_tout1Ptr + 1) += m * (-xct[1]);
               	*(here->SOI3TOUT2_tout2Ptr + 1) += m * (xct[1]+xct[2]);
              	}
            	if (here->SOI3numThermalNodes > 3)
              	{
               	*(here->SOI3TOUT2_tout3Ptr + 1) += m * (-xct[2]);
               	*(here->SOI3TOUT3_tout2Ptr + 1) += m * (-xct[2]);
               	*(here->SOI3TOUT3_tout3Ptr + 1) += m * (xct[2]+xct[3]);
              	}
              	if (here->SOI3numThermalNodes > 4)
              	{
               	*(here->SOI3TOUT3_tout4Ptr + 1) += m * (-xct[3]);
               	*(here->SOI3TOUT4_tout3Ptr + 1) += m * (-xct[3]);
               	*(here->SOI3TOUT4_tout4Ptr + 1) += m * (xct[3]+xct[4]);
              	}
            	*(here->SOI3GF_toutPtr + 1) += m * xcgfdeltaT*model->SOI3type;
            	*(here->SOI3DP_toutPtr + 1) += m * (xcddeltaT - xcBJTbddeltaT)*model->SOI3type;
            	*(here->SOI3SP_toutPtr + 1) += m * (xcsdeltaT - xcBJTbsdeltaT)*model->SOI3type;
            	*(here->SOI3B_toutPtr + 1) += m * model->SOI3type *
                                            (xcbdeltaT + xcBJTbsdeltaT + xcBJTbddeltaT);
            	*(here->SOI3GB_toutPtr + 1) += m * xcgbdeltaT*model->SOI3type;
            }


            /* and now real part */
            *(here->SOI3D_dPtr)  += (m * here->SOI3drainConductance);
            *(here->SOI3D_dpPtr) += (m * (-here->SOI3drainConductance));
            *(here->SOI3DP_dPtr) += (m * (-here->SOI3drainConductance));
            
            *(here->SOI3S_sPtr)  += (m * here->SOI3sourceConductance);
            *(here->SOI3S_spPtr) += (m * (-here->SOI3sourceConductance));
            *(here->SOI3SP_sPtr) += (m * (-here->SOI3sourceConductance));

            *(here->SOI3DP_gfPtr) += ((m * (xnrm-xrev)*here->SOI3gmf +
                                      xnrm*here->SOI3gMmf));
            *(here->SOI3DP_gbPtr) += (m * ((xnrm-xrev)*here->SOI3gmb +
                                      xnrm*here->SOI3gMmb));
            *(here->SOI3DP_dpPtr) += m * (here->SOI3drainConductance+here->SOI3gds+
            			      here->SOI3gbd+xrev*(here->SOI3gmf+here->SOI3gmbs+
                                      here->SOI3gmb)+xnrm*here->SOI3gMd);
            *(here->SOI3DP_spPtr) += m * (-here->SOI3gds - here->SOI3gBJTdb_bs
                                      -xnrm*(here->SOI3gmf+here->SOI3gmb+here->SOI3gmbs +
                                      here->SOI3gMmf+here->SOI3gMmb+here->SOI3gMmbs+here->SOI3gMd));
            *(here->SOI3DP_bPtr) += m * (-here->SOI3gbd + here->SOI3gBJTdb_bs
                                     +(xnrm-xrev)*here->SOI3gmbs+
                                      xnrm*here->SOI3gMmbs);

            *(here->SOI3SP_gfPtr) += m * (-(xnrm-xrev)*here->SOI3gmf+
                                      xrev*here->SOI3gMmf);
            *(here->SOI3SP_gbPtr) += m * (-(xnrm-xrev)*here->SOI3gmb+
                                      xrev*here->SOI3gMmb);
            *(here->SOI3SP_dpPtr) += m * (-here->SOI3gds - here->SOI3gBJTsb_bd
                                      -xrev*(here->SOI3gmf+here->SOI3gmb+here->SOI3gmbs+
                                      here->SOI3gMmf+here->SOI3gMmb+here->SOI3gMmbs+here->SOI3gMd));
            *(here->SOI3SP_spPtr) += m * (here->SOI3sourceConductance+here->SOI3gds+
                    						  here->SOI3gbs+xnrm*(here->SOI3gmf+here->SOI3gmbs+
                    						  here->SOI3gmb)+xrev*here->SOI3gMd);
            *(here->SOI3SP_bPtr) += m * (-here->SOI3gbs + here->SOI3gBJTsb_bd
                                     -(xnrm-xrev)*here->SOI3gmbs+
                                      xrev*here->SOI3gMmbs);

            *(here->SOI3B_gfPtr) += m * (-here->SOI3gMmf);
            *(here->SOI3B_gbPtr) += m * (-here->SOI3gMmb);
            *(here->SOI3B_dpPtr) += m * (-(here->SOI3gbd) + here->SOI3gBJTsb_bd +
                                   xrev*(here->SOI3gMmf+here->SOI3gMmb+
                                         here->SOI3gMmbs+here->SOI3gMd) -
                                   xnrm*here->SOI3gMd);
            *(here->SOI3B_spPtr) += m * (-(here->SOI3gbs) + here->SOI3gBJTdb_bs +
                                   xnrm*(here->SOI3gMmf+here->SOI3gMmb+
                                         here->SOI3gMmbs+here->SOI3gMd) -
                                   xrev*here->SOI3gMd);
            *(here->SOI3B_bPtr) += m * (here->SOI3gbd+here->SOI3gbs -
                                   here->SOI3gMmbs
                                   - here->SOI3gBJTdb_bs - here->SOI3gBJTsb_bd);

/* if no thermal behaviour specified, then put in zero valued indpt. voltage source
   between TOUT and ground */
            if (here->SOI3rt==0)
            {
              *(here->SOI3TOUT_ibrPtr) += m * 1.0;
              *(here->SOI3IBR_toutPtr) += m * 1.0;
              *(ckt->CKTrhs + (here->SOI3branch)) = 0;
            }
            else
            {
            	*(here->SOI3TOUT_toutPtr) += m * (-(here->SOI3gPdT)+grt[0]);
            	if (here->SOI3numThermalNodes > 1)
              	{
                	*(here->SOI3TOUT_tout1Ptr) += m * (-grt[0]);
                	*(here->SOI3TOUT1_toutPtr) += m * (-grt[0]);
                	*(here->SOI3TOUT1_tout1Ptr) += m * (grt[0]+grt[1]);
              	}
              	if (here->SOI3numThermalNodes > 2)
              	{
                 	*(here->SOI3TOUT1_tout2Ptr) += m * (-grt[1]);
                	*(here->SOI3TOUT2_tout1Ptr) += m * (-grt[1]);
                	*(here->SOI3TOUT2_tout2Ptr) += m * (grt[1]+grt[2]);
              	}
      			if (here->SOI3numThermalNodes > 3)
              	{
                	*(here->SOI3TOUT2_tout3Ptr) += m * (-grt[2]);
                	*(here->SOI3TOUT3_tout2Ptr) += m * (-grt[2]);
                	*(here->SOI3TOUT3_tout3Ptr) += m * (grt[2]+grt[3]);
              	}
              	if (here->SOI3numThermalNodes > 4)
              	{
                 	*(here->SOI3TOUT3_tout4Ptr) += m * (-grt[3]);
                	*(here->SOI3TOUT4_tout3Ptr) += m * (-grt[3]);
                	*(here->SOI3TOUT4_tout4Ptr) += m * (grt[3]+grt[4]);
              	}

              *(here->SOI3TOUT_dpPtr) += m * (xnrm*(-(here->SOI3gPds*model->SOI3type))
                                        +xrev*(here->SOI3gPds+here->SOI3gPmf+
                                               here->SOI3gPmb+here->SOI3gPmbs)*
                                               model->SOI3type);
              *(here->SOI3TOUT_gfPtr) += m * (-(here->SOI3gPmf*model->SOI3type));
              *(here->SOI3TOUT_gbPtr) += m * (-(here->SOI3gPmb*model->SOI3type));
              *(here->SOI3TOUT_bPtr) += m * (-(here->SOI3gPmbs*model->SOI3type));
              *(here->SOI3TOUT_spPtr) += m * (xnrm*(here->SOI3gPds+here->SOI3gPmf+
                                          here->SOI3gPmb+here->SOI3gPmbs)*model->SOI3type
                                        +xrev*(-(here->SOI3gPds*model->SOI3type)));

              *(here->SOI3DP_toutPtr) += m * (xnrm-xrev)*here->SOI3gt*model->SOI3type;
              *(here->SOI3SP_toutPtr) += m * (xrev-xnrm)*here->SOI3gt*model->SOI3type;
/* need to mult by type in above as conductances will be used with exterior voltages
  which will be -ve for PMOS except for gPdT */
/* now for thermal influence on impact ionisation current and transient stuff */
              *(here->SOI3DP_toutPtr) += m * (xnrm*here->SOI3gMdeltaT -
                                          here->SOI3gbdT + here->SOI3gBJTdb_deltaT)*model->SOI3type;
              *(here->SOI3SP_toutPtr) += m * (xrev*here->SOI3gMdeltaT -
                                          here->SOI3gbsT + here->SOI3gBJTsb_deltaT)*model->SOI3type;
              *(here->SOI3B_toutPtr) -= m * (here->SOI3gMdeltaT - here->SOI3gbsT -
                                         here->SOI3gbdT + here->SOI3gBJTdb_deltaT +
                                         here->SOI3gBJTsb_deltaT)*model->SOI3type;
            }
        }
    }
    return(OK);
}
