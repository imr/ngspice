/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Hong J. Park, Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim1def.h"
#include "ngspice/trandefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"

int
B1load(GENmodel *inModel, CKTcircuit *ckt)

        /* actually load the current value into the 
         * sparse matrix previously provided 
         */
{
    B1model *model = (B1model*)inModel;
    B1instance *here;
    double DrainSatCurrent = 0.0;
    double EffectiveLength = 0.0;
    double GateBulkOverlapCap = 0.0;
    double GateDrainOverlapCap = 0.0;
    double GateSourceOverlapCap = 0.0;
    double SourceSatCurrent = 0.0;
    double DrainArea = 0.0;
    double SourceArea = 0.0;
    double DrainPerimeter = 0.0;
    double SourcePerimeter = 0.0;
    double arg = 0.0;
    double capbd = 0.0;
    double capbs = 0.0;
    double cbd = 0.0;
    double cbhat = 0.0;
    double cbs = 0.0;
    double cd = 0.0;
    double cdrain = 0.0;
    double cdhat = 0.0;
    double cdreq = 0.0;
    double ceq = 0.0;
    double ceqbd = 0.0;
    double ceqbs = 0.0;
    double ceqqb = 0.0;
    double ceqqd = 0.0;
    double ceqqg = 0.0;
    double czbd = 0.0;
    double czbdsw = 0.0;
    double czbs = 0.0;
    double czbssw = 0.0;
    double delvbd = 0.0;
    double delvbs = 0.0;
    double delvds = 0.0;
    double delvgd = 0.0;
    double delvgs = 0.0;
    double evbd = 0.0;
    double evbs = 0.0;
    double gbd = 0.0;
    double gbs = 0.0;
    double gcbdb = 0.0;
    double gcbgb = 0.0;
    double gcbsb = 0.0;
    double gcddb = 0.0;
    double gcdgb = 0.0;
    double gcdsb = 0.0;
    double gcgdb = 0.0;
    double gcggb = 0.0;
    double gcgsb = 0.0;
    double gcsdb = 0.0;
    double gcsgb = 0.0;
    double gcssb = 0.0;
    double gds = 0.0;
    double geq = 0.0;
    double gm = 0.0;
    double gmbs = 0.0;
    double sarg = 0.0;
    double sargsw = 0.0;
    double vbd = 0.0;
    double vbs = 0.0;
    double vcrit = 0.0;
    double vds = 0.0;
    double vdsat = 0.0;
    double vgb = 0.0;
    double vgd = 0.0;
    double vgdo = 0.0;
    double vgs = 0.0;
    double von = 0.0;
#ifndef PREDICTOR
    double xfact = 0.0;
#endif
    double xnrm = 0.0;
    double xrev = 0.0;
    int Check = 0;
    double cgdb = 0.0;
    double cgsb = 0.0;
    double cbdb = 0.0;
    double cdgb = 0.0;
    double cddb = 0.0;
    double cdsb = 0.0;
    double cggb = 0.0;
    double cbgb = 0.0;
    double cbsb = 0.0;
    double csgb = 0.0;
    double cssb = 0.0;
    double csdb = 0.0;
    double PhiB = 0.0;
    double PhiBSW = 0.0;
    double MJ = 0.0;
    double MJSW = 0.0;
    double argsw = 0.0;
    double qgate = 0.0;
    double qbulk = 0.0;
    double qdrn = 0.0;
    double qsrc = 0.0;
    double cqgate = 0.0;
    double cqbulk = 0.0;
    double cqdrn = 0.0;
    double vt0 = 0.0;
    double args[8];
    int    ByPass = 0;
#ifndef NOBYPASS
    double tempv = 0.0;
#endif /*NOBYPASS*/
    int error = 0;

    double m; /* parallel multiplier */

    /*  loop through all the B1 device models */
    for( ; model != NULL; model = B1nextModel(model)) {

        /* loop through all the instances of the model */
        for (here = B1instances(model); here != NULL ;
                here=B1nextInstance(here)) {
        
            EffectiveLength=here->B1l - model->B1deltaL * 1.e-6;/* m */
            DrainArea = here->B1drainArea;
            SourceArea = here->B1sourceArea;
            DrainPerimeter = here->B1drainPerimeter;
            SourcePerimeter = here->B1sourcePerimeter;
            if( (DrainSatCurrent=DrainArea*model->B1jctSatCurDensity) 
                    < 1e-15){
                DrainSatCurrent = 1.0e-15;
            }
            if( (SourceSatCurrent=SourceArea*model->B1jctSatCurDensity)
                    <1.0e-15){
                SourceSatCurrent = 1.0e-15;
            }
            GateSourceOverlapCap = model->B1gateSourceOverlapCap *here->B1w;
            GateDrainOverlapCap = model->B1gateDrainOverlapCap * here->B1w;
            GateBulkOverlapCap = model->B1gateBulkOverlapCap *EffectiveLength;
            von = model->B1type * here->B1von;
            vdsat = model->B1type * here->B1vdsat;
            vt0 = model->B1type * here->B1vt0;

            Check=1;
            ByPass = 0;
            if((ckt->CKTmode & MODEINITSMSIG)) {
                vbs= *(ckt->CKTstate0 + here->B1vbs);
                vgs= *(ckt->CKTstate0 + here->B1vgs);
                vds= *(ckt->CKTstate0 + here->B1vds);
            } else if ((ckt->CKTmode & MODEINITTRAN)) {
                vbs= *(ckt->CKTstate1 + here->B1vbs);
                vgs= *(ckt->CKTstate1 + here->B1vgs);
                vds= *(ckt->CKTstate1 + here->B1vds);
            } else if((ckt->CKTmode & MODEINITJCT) && !here->B1off) {
                vds= model->B1type * here->B1icVDS;
                vgs= model->B1type * here->B1icVGS;
                vbs= model->B1type * here->B1icVBS;
                if((vds==0) && (vgs==0) && (vbs==0) && 
                        ((ckt->CKTmode & 
                        (MODETRAN|MODEAC|MODEDCOP|MODEDCTRANCURVE)) ||
                        (!(ckt->CKTmode & MODEUIC)))) {
                    vbs = -1;
                    vgs = vt0;
                    vds = 0;
                }
            } else if((ckt->CKTmode & (MODEINITJCT | MODEINITFIX) ) && 
                    (here->B1off)) {
                vbs=vgs=vds=0;
            } else {
#ifndef PREDICTOR
                if((ckt->CKTmode & MODEINITPRED)) {
                    xfact=ckt->CKTdelta/ckt->CKTdeltaOld[1];
                    *(ckt->CKTstate0 + here->B1vbs) = 
                            *(ckt->CKTstate1 + here->B1vbs);
                    vbs = (1+xfact)* (*(ckt->CKTstate1 + here->B1vbs))
                            -(xfact * (*(ckt->CKTstate2 + here->B1vbs)));
                    *(ckt->CKTstate0 + here->B1vgs) = 
                            *(ckt->CKTstate1 + here->B1vgs);
                    vgs = (1+xfact)* (*(ckt->CKTstate1 + here->B1vgs))
                            -(xfact * (*(ckt->CKTstate2 + here->B1vgs)));
                    *(ckt->CKTstate0 + here->B1vds) = 
                            *(ckt->CKTstate1 + here->B1vds);
                    vds = (1+xfact)* (*(ckt->CKTstate1 + here->B1vds))
                            -(xfact * (*(ckt->CKTstate2 + here->B1vds)));
                    *(ckt->CKTstate0 + here->B1vbd) = 
                            *(ckt->CKTstate0 + here->B1vbs)-
                            *(ckt->CKTstate0 + here->B1vds);
                    *(ckt->CKTstate0 + here->B1cd) = 
                            *(ckt->CKTstate1 + here->B1cd);
                    *(ckt->CKTstate0 + here->B1cbs) = 
                            *(ckt->CKTstate1 + here->B1cbs);
                    *(ckt->CKTstate0 + here->B1cbd) = 
                            *(ckt->CKTstate1 + here->B1cbd);
                    *(ckt->CKTstate0 + here->B1gm) = 
                            *(ckt->CKTstate1 + here->B1gm);
                    *(ckt->CKTstate0 + here->B1gds) = 
                            *(ckt->CKTstate1 + here->B1gds);
                    *(ckt->CKTstate0 + here->B1gmbs) = 
                            *(ckt->CKTstate1 + here->B1gmbs);
                    *(ckt->CKTstate0 + here->B1gbd) = 
                            *(ckt->CKTstate1 + here->B1gbd);
                    *(ckt->CKTstate0 + here->B1gbs) = 
                            *(ckt->CKTstate1 + here->B1gbs);
                    *(ckt->CKTstate0 + here->B1cggb) = 
                            *(ckt->CKTstate1 + here->B1cggb);
                    *(ckt->CKTstate0 + here->B1cbgb) = 
                            *(ckt->CKTstate1 + here->B1cbgb);
                    *(ckt->CKTstate0 + here->B1cbsb) = 
                            *(ckt->CKTstate1 + here->B1cbsb);
                    *(ckt->CKTstate0 + here->B1cgdb) = 
                            *(ckt->CKTstate1 + here->B1cgdb);
                    *(ckt->CKTstate0 + here->B1cgsb) = 
                            *(ckt->CKTstate1 + here->B1cgsb);
                    *(ckt->CKTstate0 + here->B1cbdb) = 
                            *(ckt->CKTstate1 + here->B1cbdb);
                    *(ckt->CKTstate0 + here->B1cdgb) = 
                            *(ckt->CKTstate1 + here->B1cdgb);
                    *(ckt->CKTstate0 + here->B1cddb) = 
                            *(ckt->CKTstate1 + here->B1cddb);
                    *(ckt->CKTstate0 + here->B1cdsb) = 
                            *(ckt->CKTstate1 + here->B1cdsb);
                } else {
#endif /* PREDICTOR */
                    vbs = model->B1type * ( 
                        *(ckt->CKTrhsOld+here->B1bNode) -
                        *(ckt->CKTrhsOld+here->B1sNodePrime));
                    vgs = model->B1type * ( 
                        *(ckt->CKTrhsOld+here->B1gNode) -
                        *(ckt->CKTrhsOld+here->B1sNodePrime));
                    vds = model->B1type * ( 
                        *(ckt->CKTrhsOld+here->B1dNodePrime) -
                        *(ckt->CKTrhsOld+here->B1sNodePrime));
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                vbd=vbs-vds;
                vgd=vgs-vds;
                vgdo = *(ckt->CKTstate0 + here->B1vgs) - 
                    *(ckt->CKTstate0 + here->B1vds);
                delvbs = vbs - *(ckt->CKTstate0 + here->B1vbs);
                delvbd = vbd - *(ckt->CKTstate0 + here->B1vbd);
                delvgs = vgs - *(ckt->CKTstate0 + here->B1vgs);
                delvds = vds - *(ckt->CKTstate0 + here->B1vds);
                delvgd = vgd-vgdo;

                if (here->B1mode >= 0) {
                    cdhat=
                        *(ckt->CKTstate0 + here->B1cd) -
                        *(ckt->CKTstate0 + here->B1gbd) * delvbd +
                        *(ckt->CKTstate0 + here->B1gmbs) * delvbs +
                        *(ckt->CKTstate0 + here->B1gm) * delvgs + 
                        *(ckt->CKTstate0 + here->B1gds) * delvds ;
                } else {
                    cdhat=
                        *(ckt->CKTstate0 + here->B1cd) -
                        ( *(ckt->CKTstate0 + here->B1gbd) -
                          *(ckt->CKTstate0 + here->B1gmbs)) * delvbd -
                        *(ckt->CKTstate0 + here->B1gm) * delvgd +
                        *(ckt->CKTstate0 + here->B1gds) * delvds;
                }
                cbhat=
                    *(ckt->CKTstate0 + here->B1cbs) +
                    *(ckt->CKTstate0 + here->B1cbd) +
                    *(ckt->CKTstate0 + here->B1gbd) * delvbd +
                    *(ckt->CKTstate0 + here->B1gbs) * delvbs ;

#ifndef NOBYPASS
                    /* now lets see if we can bypass (ugh) */

                /* following should be one big if connected by && all over
                 * the place, but some C compilers can't handle that, so
                 * we split it up here to let them digest it in stages
                 */
                tempv = MAX(fabs(cbhat),fabs(*(ckt->CKTstate0 + here->B1cbs)
                        + *(ckt->CKTstate0 + here->B1cbd)))+ckt->CKTabstol;
                if((!(ckt->CKTmode & MODEINITPRED)) && (ckt->CKTbypass) )
                if( (fabs(delvbs) < (ckt->CKTreltol * MAX(fabs(vbs),
                        fabs(*(ckt->CKTstate0+here->B1vbs)))+
                        ckt->CKTvoltTol)) )
                if ( (fabs(delvbd) < (ckt->CKTreltol * MAX(fabs(vbd),
                        fabs(*(ckt->CKTstate0+here->B1vbd)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(delvgs) < (ckt->CKTreltol * MAX(fabs(vgs),
                        fabs(*(ckt->CKTstate0+here->B1vgs)))+
                        ckt->CKTvoltTol)))
                if ( (fabs(delvds) < (ckt->CKTreltol * MAX(fabs(vds),
                        fabs(*(ckt->CKTstate0+here->B1vds)))+
                        ckt->CKTvoltTol)) )
                if( (fabs(cdhat- *(ckt->CKTstate0 + here->B1cd)) <
                        ckt->CKTreltol * MAX(fabs(cdhat),fabs(*(ckt->CKTstate0 +
                        here->B1cd))) + ckt->CKTabstol) )
                if ( (fabs(cbhat-(*(ckt->CKTstate0 + here->B1cbs) +
                        *(ckt->CKTstate0 + here->B1cbd))) < ckt->CKTreltol *
                        tempv)) {
                    /* bypass code */
                    vbs = *(ckt->CKTstate0 + here->B1vbs);
                    vbd = *(ckt->CKTstate0 + here->B1vbd);
                    vgs = *(ckt->CKTstate0 + here->B1vgs);
                    vds = *(ckt->CKTstate0 + here->B1vds);
                    vgd = vgs - vds;
                    vgb = vgs - vbs;
                    cd = *(ckt->CKTstate0 + here->B1cd);
                    cbs = *(ckt->CKTstate0 + here->B1cbs);
                    cbd = *(ckt->CKTstate0 + here->B1cbd);
                    cdrain = here->B1mode * (cd + cbd);
                    gm = *(ckt->CKTstate0 + here->B1gm);
                    gds = *(ckt->CKTstate0 + here->B1gds);
                    gmbs = *(ckt->CKTstate0 + here->B1gmbs);
                    gbd = *(ckt->CKTstate0 + here->B1gbd);
                    gbs = *(ckt->CKTstate0 + here->B1gbs);
                    if((ckt->CKTmode & (MODETRAN | MODEAC)) || 
                            ((ckt->CKTmode & MODETRANOP) && 
                            (ckt->CKTmode & MODEUIC))) {
                        cggb = *(ckt->CKTstate0 + here->B1cggb);
                        cgdb = *(ckt->CKTstate0 + here->B1cgdb);
                        cgsb = *(ckt->CKTstate0 + here->B1cgsb);
                        cbgb = *(ckt->CKTstate0 + here->B1cbgb);
                        cbdb = *(ckt->CKTstate0 + here->B1cbdb);
                        cbsb = *(ckt->CKTstate0 + here->B1cbsb);
                        cdgb = *(ckt->CKTstate0 + here->B1cdgb);
                        cddb = *(ckt->CKTstate0 + here->B1cddb);
                        cdsb = *(ckt->CKTstate0 + here->B1cdsb);
                        capbs = *(ckt->CKTstate0 + here->B1capbs);
                        capbd = *(ckt->CKTstate0 + here->B1capbd);
                        ByPass = 1;
                        goto line755;
                    } else {
                        goto line850;
                    }
                }
#endif /*NOBYPASS*/

                von = model->B1type * here->B1von;
                if(*(ckt->CKTstate0 + here->B1vds) >=0) {
                    vgs = DEVfetlim(vgs,*(ckt->CKTstate0 + here->B1vgs)
                            ,von);
                    vds = vgs - vgd;
                    vds = DEVlimvds(vds,*(ckt->CKTstate0 + here->B1vds));
                    vgd = vgs - vds;
                } else {
                    vgd = DEVfetlim(vgd,vgdo,von);
                    vds = vgs - vgd;
                    vds = -DEVlimvds(-vds,-(*(ckt->CKTstate0 + here->B1vds)));
                    vgs = vgd + vds;
                }
                if(vds >= 0) {
                    vcrit =CONSTvt0*log(CONSTvt0/(CONSTroot2*SourceSatCurrent));
                    vbs = DEVpnjlim(vbs,*(ckt->CKTstate0 + here->B1vbs),
                            CONSTvt0,vcrit,&Check); /* B1 test */
                    vbd = vbs-vds;
                } else {
                    vcrit = CONSTvt0*log(CONSTvt0/(CONSTroot2*DrainSatCurrent));
                    vbd = DEVpnjlim(vbd,*(ckt->CKTstate0 + here->B1vbd),
                            CONSTvt0,vcrit,&Check); /* B1 test*/
                    vbs = vbd + vds;
                }
            } 

             /* determine DC current and derivatives */
            vbd = vbs - vds;
            vgd = vgs - vds;
            vgb = vgs - vbs;


            if(vbs <= 0.0 ) {
                gbs = SourceSatCurrent / CONSTvt0 + ckt->CKTgmin;
                cbs = gbs * vbs ;
            } else {
                evbs = exp(vbs/CONSTvt0);
                gbs = SourceSatCurrent*evbs/CONSTvt0 + ckt->CKTgmin;
                cbs = SourceSatCurrent * (evbs-1) + ckt->CKTgmin * vbs ;
            }
            if(vbd <= 0.0) {
                gbd = DrainSatCurrent / CONSTvt0 + ckt->CKTgmin;
                cbd = gbd * vbd ;
            } else {
                evbd = exp(vbd/CONSTvt0);
                gbd = DrainSatCurrent*evbd/CONSTvt0 +ckt->CKTgmin;
                cbd = DrainSatCurrent *(evbd-1)+ckt->CKTgmin*vbd;
            }
            /* line 400 */
            if(vds >= 0) {
                /* normal mode */
                here->B1mode = 1;
            } else {
                /* inverse mode */
                here->B1mode = -1;
            }
            /* call B1evaluate to calculate drain current and its 
             * derivatives and charge and capacitances related to gate
             * drain, and bulk
             */
           if( vds >= 0 )  {
                B1evaluate(vds,vbs,vgs,here,model,&gm,&gds,&gmbs,&qgate,
                    &qbulk,&qdrn,&cggb,&cgdb,&cgsb,&cbgb,&cbdb,&cbsb,&cdgb,
                    &cddb,&cdsb,&cdrain,&von,&vdsat,ckt);
            } else {
                B1evaluate(-vds,vbd,vgd,here,model,&gm,&gds,&gmbs,&qgate,
                    &qbulk,&qsrc,&cggb,&cgsb,&cgdb,&cbgb,&cbsb,&cbdb,&csgb,
                    &cssb,&csdb,&cdrain,&von,&vdsat,ckt);
            }
          
            here->B1von = model->B1type * von;
            here->B1vdsat = model->B1type * vdsat;  

        

            /*
             *  COMPUTE EQUIVALENT DRAIN CURRENT SOURCE
             */
            cd=here->B1mode * cdrain - cbd;
            if ((ckt->CKTmode & (MODETRAN | MODEAC | MODEINITSMSIG)) ||
                    ((ckt->CKTmode & MODETRANOP ) && 
                    (ckt->CKTmode & MODEUIC))) {
                /*
                 *  charge storage elements
                 *
                 *   bulk-drain and bulk-source depletion capacitances
                 *  czbd : zero bias drain junction capacitance
                 *  czbs : zero bias source junction capacitance
                 * czbdsw:zero bias drain junction sidewall capacitance
                 * czbssw:zero bias source junction sidewall capacitance
                 */

                czbd  = model->B1unitAreaJctCap * DrainArea;
                czbs  = model->B1unitAreaJctCap * SourceArea;
                czbdsw= model->B1unitLengthSidewallJctCap * DrainPerimeter;
                czbssw= model->B1unitLengthSidewallJctCap * SourcePerimeter;
                PhiB = model->B1bulkJctPotential;
                PhiBSW = model->B1sidewallJctPotential;
                MJ = model->B1bulkJctBotGradingCoeff;
                MJSW = model->B1bulkJctSideGradingCoeff;

                /* Source Bulk Junction */
                if( vbs < 0 ) {  
                    arg = 1 - vbs / PhiB;
                    argsw = 1 - vbs / PhiBSW;
                    sarg = exp(-MJ*log(arg));
                    sargsw = exp(-MJSW*log(argsw));
                    *(ckt->CKTstate0 + here->B1qbs) =
                        PhiB * czbs * (1-arg*sarg)/(1-MJ) + PhiBSW * 
                    czbssw * (1-argsw*sargsw)/(1-MJSW);
                    capbs = czbs * sarg + czbssw * sargsw ;
                } else {  
                    *(ckt->CKTstate0+here->B1qbs) =
                        vbs*(czbs+czbssw)+ vbs*vbs*(czbs*MJ*0.5/PhiB 
                        + czbssw * MJSW * 0.5/PhiBSW);
                    capbs = czbs + czbssw + vbs *(czbs*MJ/PhiB+
                        czbssw * MJSW / PhiBSW );
                }

                /* Drain Bulk Junction */
                if( vbd < 0 ) {  
                    arg = 1 - vbd / PhiB;
                    argsw = 1 - vbd / PhiBSW;
                    sarg = exp(-MJ*log(arg));
                    sargsw = exp(-MJSW*log(argsw));
                    *(ckt->CKTstate0 + here->B1qbd) =
                        PhiB * czbd * (1-arg*sarg)/(1-MJ) + PhiBSW * 
                    czbdsw * (1-argsw*sargsw)/(1-MJSW);
                    capbd = czbd * sarg + czbdsw * sargsw ;
                } else {  
                    *(ckt->CKTstate0+here->B1qbd) =
                        vbd*(czbd+czbdsw)+ vbd*vbd*(czbd*MJ*0.5/PhiB 
                        + czbdsw * MJSW * 0.5/PhiBSW);
                    capbd = czbd + czbdsw + vbd *(czbd*MJ/PhiB+
                        czbdsw * MJSW / PhiBSW );
                }

            }




            /*
             *  check convergence
             */
            if ( (here->B1off == 0)  || (!(ckt->CKTmode & MODEINITFIX)) ){
                if (Check == 1) {
                    ckt->CKTnoncon++;
		    ckt->CKTtroubleElt = (GENinstance *) here;
                }
            }
            *(ckt->CKTstate0 + here->B1vbs) = vbs;
            *(ckt->CKTstate0 + here->B1vbd) = vbd;
            *(ckt->CKTstate0 + here->B1vgs) = vgs;
            *(ckt->CKTstate0 + here->B1vds) = vds;
            *(ckt->CKTstate0 + here->B1cd) = cd;
            *(ckt->CKTstate0 + here->B1cbs) = cbs;
            *(ckt->CKTstate0 + here->B1cbd) = cbd;
            *(ckt->CKTstate0 + here->B1gm) = gm;
            *(ckt->CKTstate0 + here->B1gds) = gds;
            *(ckt->CKTstate0 + here->B1gmbs) = gmbs;
            *(ckt->CKTstate0 + here->B1gbd) = gbd;
            *(ckt->CKTstate0 + here->B1gbs) = gbs;

            *(ckt->CKTstate0 + here->B1cggb) = cggb;
            *(ckt->CKTstate0 + here->B1cgdb) = cgdb;
            *(ckt->CKTstate0 + here->B1cgsb) = cgsb;

            *(ckt->CKTstate0 + here->B1cbgb) = cbgb;
            *(ckt->CKTstate0 + here->B1cbdb) = cbdb;
            *(ckt->CKTstate0 + here->B1cbsb) = cbsb;

            *(ckt->CKTstate0 + here->B1cdgb) = cdgb;
            *(ckt->CKTstate0 + here->B1cddb) = cddb;
            *(ckt->CKTstate0 + here->B1cdsb) = cdsb;

            *(ckt->CKTstate0 + here->B1capbs) = capbs;
            *(ckt->CKTstate0 + here->B1capbd) = capbd;

           /* bulk and channel charge plus overlaps */

            if((!(ckt->CKTmode & (MODETRAN | MODEAC))) &&
                    ((!(ckt->CKTmode & MODETRANOP)) ||
                    (!(ckt->CKTmode & MODEUIC)))  && (!(ckt->CKTmode 
                    &  MODEINITSMSIG))) goto line850; 
#ifndef NOBYPASS         
line755:
#endif
            if( here->B1mode > 0 ) {

		args[0] = GateDrainOverlapCap;
		args[1] = GateSourceOverlapCap;
		args[2] = GateBulkOverlapCap;
		args[3] = capbd;
		args[4] = capbs;
		args[5] = cggb;
		args[6] = cgdb;
		args[7] = cgsb;

                B1mosCap(ckt,vgd,vgs,vgb,
			args,
			/*
			GateDrainOverlapCap,
			GateSourceOverlapCap,GateBulkOverlapCap,
			capbd,capbs,
                        cggb,cgdb,cgsb,
			*/
			cbgb,cbdb,cbsb,
			cdgb,cddb,cdsb,
                        &gcggb,&gcgdb,&gcgsb,
			&gcbgb,&gcbdb,&gcbsb,
			&gcdgb,&gcddb,&gcdsb,&gcsgb,&gcsdb,&gcssb,
			&qgate,&qbulk,
                        &qdrn,&qsrc);
            } else {

		args[0] = GateSourceOverlapCap;
		args[1] = GateDrainOverlapCap;
		args[2] = GateBulkOverlapCap;
		args[3] = capbs;
		args[4] = capbd;
		args[5] = cggb;
		args[6] = cgsb;
		args[7] = cgdb;

                B1mosCap(ckt,vgs,vgd,vgb,
			args,
			/*
			GateSourceOverlapCap,
			GateDrainOverlapCap,GateBulkOverlapCap,
			capbs,capbd,
			cggb,cgsb,cgdb,
			*/
			cbgb,cbsb,cbdb,
			csgb,cssb,csdb,
			&gcggb,&gcgsb,&gcgdb,
			&gcbgb,&gcbsb,&gcbdb,
			&gcsgb,&gcssb,&gcsdb,&gcdgb,&gcdsb,&gcddb,
			&qgate,&qbulk,
			&qsrc,&qdrn);
            }
             
            if(ByPass) goto line860;
            *(ckt->CKTstate0 + here->B1qg) = qgate;
            *(ckt->CKTstate0 + here->B1qd) = qdrn -  
                    *(ckt->CKTstate0 + here->B1qbd);
            *(ckt->CKTstate0 + here->B1qb) = qbulk +  
                    *(ckt->CKTstate0 + here->B1qbd) +  
                    *(ckt->CKTstate0 + here->B1qbs); 

            /* store small signal parameters */
            if((!(ckt->CKTmode & (MODEAC | MODETRAN))) &&
                    (ckt->CKTmode & MODETRANOP ) && (ckt->CKTmode &
                    MODEUIC ))   goto line850;
            if(ckt->CKTmode & MODEINITSMSIG ) {  
                *(ckt->CKTstate0+here->B1cggb) = cggb;
                *(ckt->CKTstate0+here->B1cgdb) = cgdb;
                *(ckt->CKTstate0+here->B1cgsb) = cgsb;
                *(ckt->CKTstate0+here->B1cbgb) = cbgb;
                *(ckt->CKTstate0+here->B1cbdb) = cbdb;
                *(ckt->CKTstate0+here->B1cbsb) = cbsb;
                *(ckt->CKTstate0+here->B1cdgb) = cdgb;
                *(ckt->CKTstate0+here->B1cddb) = cddb;
                *(ckt->CKTstate0+here->B1cdsb) = cdsb;     
                *(ckt->CKTstate0+here->B1capbd) = capbd;
                *(ckt->CKTstate0+here->B1capbs) = capbs;

                goto line1000;
            }
       
            if(ckt->CKTmode & MODEINITTRAN ) { 
                *(ckt->CKTstate1+here->B1qb) =
                    *(ckt->CKTstate0+here->B1qb) ;
                *(ckt->CKTstate1+here->B1qg) =
                    *(ckt->CKTstate0+here->B1qg) ;
                *(ckt->CKTstate1+here->B1qd) =
                    *(ckt->CKTstate0+here->B1qd) ;
            }
       
       
            error = NIintegrate(ckt,&geq,&ceq,0.0,here->B1qb);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ceq,0.0,here->B1qg);
            if(error) return(error);
            error = NIintegrate(ckt,&geq,&ceq,0.0,here->B1qd);
            if(error) return(error);
      
            goto line860;

line850:
            /* initialize to zero charge conductance and current */
            ceqqg = ceqqb = ceqqd = 0.0;
            gcdgb = gcddb = gcdsb = 0.0;
            gcsgb = gcsdb = gcssb = 0.0;
            gcggb = gcgdb = gcgsb = 0.0;
            gcbgb = gcbdb = gcbsb = 0.0;
            goto line900;
            
line860:
            /* evaluate equivalent charge current */
            cqgate = *(ckt->CKTstate0 + here->B1iqg);
            cqbulk = *(ckt->CKTstate0 + here->B1iqb);
            cqdrn = *(ckt->CKTstate0 + here->B1iqd);
            ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs;
            ceqqb = cqbulk - gcbgb * vgb + gcbdb * vbd + gcbsb * vbs;
            ceqqd = cqdrn - gcdgb * vgb + gcddb * vbd + gcdsb * vbs;

            if(ckt->CKTmode & MODEINITTRAN ) {  
                *(ckt->CKTstate1 + here->B1iqb) =  
                    *(ckt->CKTstate0 + here->B1iqb);
                *(ckt->CKTstate1 + here->B1iqg) =  
                    *(ckt->CKTstate0 + here->B1iqg);
                *(ckt->CKTstate1 + here->B1iqd) =  
                    *(ckt->CKTstate0 + here->B1iqd);
            }

            /*
             *  load current vector
             */
line900:
            m = here->B1m;

            ceqbs = model->B1type * (cbs-(gbs-ckt->CKTgmin)*vbs);
            ceqbd = model->B1type * (cbd-(gbd-ckt->CKTgmin)*vbd);
     
            ceqqg = model->B1type * ceqqg;
            ceqqb = model->B1type * ceqqb;
            ceqqd =  model->B1type * ceqqd;
            if (here->B1mode >= 0) {
                xnrm=1;
                xrev=0;
                cdreq=model->B1type*(cdrain-gds*vds-gm*vgs-gmbs*vbs);
            } else {
                xnrm=0;
                xrev=1;
                cdreq = -(model->B1type)*(cdrain+gds*vds-gm*vgd-gmbs*vbd);
            }

            *(ckt->CKTrhs + here->B1gNode) -= m * ceqqg;
            *(ckt->CKTrhs + here->B1bNode) -= m * (ceqbs+ceqbd+ceqqb);
            *(ckt->CKTrhs + here->B1dNodePrime) +=
                    m * (ceqbd-cdreq-ceqqd);
            *(ckt->CKTrhs + here->B1sNodePrime) += 
                   m * (cdreq+ceqbs+ceqqg+ceqqb+ceqqd);

            /*
             *  load y matrix
             */

            *(here->B1DdPtr) += m * (here->B1drainConductance);
            *(here->B1GgPtr) += m * (gcggb);
            *(here->B1SsPtr) += m * (here->B1sourceConductance);
            *(here->B1BbPtr) += m * (gbd+gbs-gcbgb-gcbdb-gcbsb);
            *(here->B1DPdpPtr) += 
                 m * (here->B1drainConductance+gds+gbd+xrev*(gm+gmbs)+gcddb);
            *(here->B1SPspPtr) += 
                m * (here->B1sourceConductance+gds+gbs+xnrm*(gm+gmbs)+gcssb);
            *(here->B1DdpPtr) += m * (-here->B1drainConductance);
            *(here->B1GbPtr) += m * (-gcggb-gcgdb-gcgsb);
            *(here->B1GdpPtr) += m * (gcgdb);
            *(here->B1GspPtr) += m * (gcgsb);
            *(here->B1SspPtr) += m * (-here->B1sourceConductance);
            *(here->B1BgPtr) += m * (gcbgb);
            *(here->B1BdpPtr) += m * (-gbd+gcbdb);
            *(here->B1BspPtr) += m * (-gbs+gcbsb);
            *(here->B1DPdPtr) += m * (-here->B1drainConductance);
            *(here->B1DPgPtr) += m * ((xnrm-xrev)*gm+gcdgb);
            *(here->B1DPbPtr) += m * (-gbd+(xnrm-xrev)*gmbs-gcdgb-gcddb-gcdsb);
            *(here->B1DPspPtr) += m * (-gds-xnrm*(gm+gmbs)+gcdsb);
            *(here->B1SPgPtr) += m * (-(xnrm-xrev)*gm+gcsgb);
            *(here->B1SPsPtr) += m * (-here->B1sourceConductance);
            *(here->B1SPbPtr) += m * (-gbs-(xnrm-xrev)*gmbs-gcsgb-gcsdb-gcssb);
            *(here->B1SPdpPtr) += m * (-gds-xrev*(gm+gmbs)+gcsdb);


line1000:  ;

        }   /* End of Mosfet Instance */

    }       /* End of Model Instance */
    return(OK);
}

