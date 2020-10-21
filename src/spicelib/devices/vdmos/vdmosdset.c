/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "vdmosdefs.h"
#include "ngspice/distodef.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
VDMOSdSetup(GENmodel *inModel, CKTcircuit *ckt)
        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    VDMOSmodel *model = (VDMOSmodel *) inModel;
    VDMOSinstance *here;
    double Beta;
    double OxideCap;
    double gm;
    double gds;
    double vgst;
    double vds;
    double vdsat;
    double vgd;
    double vgs;
    double gm2;
    double gds2;
    double gmds;
    double gm3;
    double gds3;
    double gm2ds;
    double gmds2;
    double lcapgs2;
    double lcapgs3;
    double lcapgd2;
    double lcapgd3;


    /*  loop through all the VDMOS device models */
    for( ; model != NULL; model = VDMOSnextModel(model)) {
        /* loop through all the instances of the model */
        for (here = VDMOSinstances(model); here != NULL ;
                here=VDMOSnextInstance(here)) {

            Beta = here->VDMOStTransconductance;

            OxideCap = model->VDMOSoxideCapFactor * here->VDMOSm;

            vgs = model->VDMOStype * ( 
                *(ckt->CKTrhsOld+here->VDMOSgNode) -
                *(ckt->CKTrhsOld+here->VDMOSsNodePrime));
            vds = model->VDMOStype * ( 
                *(ckt->CKTrhsOld+here->VDMOSdNodePrime) -
                *(ckt->CKTrhsOld+here->VDMOSsNodePrime));

            /* now some common crunching for some more useful quantities */

            vgd=vgs-vds;


            /* now to determine whether the user was able to correctly
             * identify the source and drain of his device
             */
            if(vds >= 0) {
                /* normal mode */
                here->VDMOSmode = 1;
            } else {
                /* inverse mode */
                here->VDMOSmode = -1;
            }

            /*
             *     this block of code evaluates the drain current and its 
             *     derivatives using the shichman-hodges model and the 
             *     charges associated with the gate and channel for 
             *     mosfets
             *
             */

            /* the following variables are local to this code block until 
             * it is obvious that they can be made global 
             */
            {
                double von = here->VDMOStVth * model->VDMOStype;
                vgst = (here->VDMOSmode == 1 ? vgs : vgd) - von;
                vdsat = MAX(vgst, 0);
                double slope = model->VDMOSksubthres;
                double lambda = model->VDMOSlambda;
                double theta = model->VDMOStheta;
                double shift = model->VDMOSsubshift;
                double mtr = model->VDMOSmtr;

                /* scale vds with mtr (except with lambda) */
                double vdss = vds*mtr*here->VDMOSmode;
                double t0 = 1 + lambda*vds;
                double t1 = 1 + theta*vgs;
                double betap = Beta*t0/t1;
                double dbetapdvgs = -Beta*theta*t0/(t1*t1);
                double dbetapdvds = Beta*lambda/t1;

                double t2 = exp((vgst-shift)/slope);
                vgst = slope * log(1 + t2);
                double dvgstdvgs = t2/(t2+1);

                if (vgst <= vdss) {
                    /* saturation region */
                    gm = betap*vgst*dvgstdvgs + 0.5*dbetapdvgs*vgst*vgst;
                    gds = .5*dbetapdvds*vgst*vgst;
                        gm2 = betap;
                        gds2 = 0;
                        gmds = Beta*lambda*vgst;
                        gm3 = 0;
                        gds3 = 0;
                        gm2ds = Beta*lambda;
                        gmds2 = 0;
                }
                else {
                    /* linear region */
                    gm = betap*vdss*dvgstdvgs + vdss*dbetapdvgs*(vgst-.5*vdss);
                    gds = vdss*dbetapdvds*(vgst-.5*vdss) + betap*mtr*(vgst-.5*vdss) - .5*vdss*betap*mtr;
                        gm2 = 0;
                        gds2 = 2*Beta * lambda*(vgst - vds*here->VDMOSmode) - betap;
                        gmds = Beta * lambda * vds * here->VDMOSmode + betap;
                        gm3=0;
                        gds3 = -Beta*lambda*3.;
                        gm2ds=0;
                        gmds2 = 2*lambda*Beta;
                }
                /*
                 *     finished
                 */
            } /* code block */


            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            /*
             *     meyer's capacitor model
             */
            /*
             * the meyer capacitance equations are in DEVqmeyer
             * these expressions are derived from those equations.
             * these expressions are incorrect; they assume just one
             * controlling variable for each charge storage element
             * while actually there are several;  the VDMOS small
             * signal ac linear model is also wrong because it 
             * ignores controlled capacitive elements. these can be 
             * corrected (as can the linear ss ac model) if the 
             * expressions for the charge are available
             */

            {

                double phi;
                double cox;
                double vddif;
                double vddif1;
                double vddif2;
                /* von, vgst and vdsat have already been adjusted for 
                    possible source-drain interchange */
                phi = here->VDMOStPhi;
                cox = OxideCap; /*FIXME: using a guess for the oxide thickness of 1e-07 in vdmostemp.c */
                lcapgs2=lcapgs3=lcapgd2=lcapgd3=0;
                if (vgst <= 0) {
                    lcapgs2 = cox/(3*phi);
                } else  {    /* the VDMOSmodes are around because 
                                vds has not been adjusted */
                    if (vdsat <= here->VDMOSmode*vds) {
                        lcapgs2=lcapgs3=lcapgd2=lcapgd3=0;
                    } else {
                        vddif = 2.0*vdsat-here->VDMOSmode*vds;
                        vddif1 = vdsat-here->VDMOSmode*vds/*-1.0e-12*/;
                        vddif2 = vddif*vddif;
                        lcapgd2 = -vdsat*here->VDMOSmode*vds*cox/(3*vddif*vddif2);
                        lcapgd3 = - here->VDMOSmode*vds*cox*(vddif - 6*vdsat)/(9*vddif2*vddif2);
                        lcapgs2 = -vddif1*here->VDMOSmode*vds*cox/(3*vddif*vddif2);
                        lcapgs3 = - here->VDMOSmode*vds*cox*(vddif - 6*vddif1)/(9*vddif2*vddif2);
                    }
                }
            }

            /*
             *   process to get Taylor coefficients, taking into
             * account type and mode.
             */

            if (here->VDMOSmode == 1)
                {
                /* normal mode - no source-drain interchange */

                here->cdr_x2 = gm2;
                here->cdr_y2 = gds2;;
                here->cdr_xy = gmds;
                here->cdr_x3 = gm3;
                here->cdr_y3 = gds3;
                here->cdr_x2y = gm2ds;
                here->cdr_xy2 = gmds2;

                /* the gate caps have been divided and made into
                    Taylor coeffs., but not adjusted for type */

                here->capgs2 = model->VDMOStype*lcapgs2;
                here->capgs3 = lcapgs3;
                here->capgd2 = model->VDMOStype*lcapgd2;
                here->capgd3 = lcapgd3;
            } else {
                /*
                 * inverse mode - source and drain interchanged
                 */

                here->cdr_x2 = -gm2;
                here->cdr_y2 = -(gm2 + gds2 + 2*(gmds));
                here->cdr_xy = gm2 + gmds;
                here->cdr_x3 = -gm3;
                here->cdr_y3 = gm3 + gds3 + 
                    3*(gm2ds + gmds2) ;
                here->cdr_x2y = gm3 + gm2ds;
                here->cdr_xy2 = -(gm3 + 2*(gm2ds) +
                                    gmds2);

                here->capgs2 = model->VDMOStype*lcapgd2;
                here->capgs3 = lcapgd3;

                here->capgd2 = model->VDMOStype*lcapgs2;
                here->capgd3 = lcapgs3;

            }

            /* now to adjust for type and multiply by factors to convert to Taylor coeffs. */

            here->cdr_x2 = 0.5*model->VDMOStype*here->cdr_x2;
            here->cdr_y2 = 0.5*model->VDMOStype*here->cdr_y2;
            here->cdr_xy = model->VDMOStype*here->cdr_xy;
            here->cdr_x3 = here->cdr_x3/6.;
            here->cdr_y3 = here->cdr_y3/6.;
            here->cdr_x2y = 0.5*here->cdr_x2y;
            here->cdr_xy2 = 0.5*here->cdr_xy2;


        }
    }
    return(OK);
}
