/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Gary W. Ng
**********/

#include "ngspice/ngspice.h"
#include "jfetdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/noisedef.h"
#include "ngspice/suffix.h"

/*
*JFETnoise (mode, operation, firstModel, ckt, data, OnDens)
*   This routine names and evaluates all of the noise sources
*   associated with JFET's.  It starts with the model *firstModel and
*   traverses all of its insts.  It then proceeds to any other models
*   on the linked list.  The total output noise density generated by
*   all of the JFET's is summed with the variable "OnDens".
 */

int
JFETnoise(int mode, int operation, GENmodel *genmodel, CKTcircuit *ckt, Ndata *data,
          double *OnDens)
{
    NOISEAN *job = (NOISEAN*) ckt->CKTcurJob;

    JFETmodel *firstModel = (JFETmodel*) genmodel;
    JFETmodel *model;
    JFETinstance *inst;
    double tempOnoise;
    double tempInoise;
    double noizDens[JFETNSRCS];
    double lnNdens[JFETNSRCS];
    int i;
    double vgs, vds, vgst, alpha, beta;
    double dtemp;

    /* define the names of the noise sources */

    static char *JFETnNames[JFETNSRCS] = {
        /* Note that we have to keep the order
           consistent with the strchr definitions in JFETdefs.h */
        "_rd",        /* noise due to rd */
        "_rs",        /* noise due to rs */
        "_id",        /* noise due to id */
        "_1overf",    /* flicker (1/f) noise */
        ""            /* total transistor noise */
    };

    for (model = firstModel; model != NULL; model = JFETnextModel(model)) {
        for (inst = JFETinstances(model); inst != NULL; inst = JFETnextInstance(inst)) {

            switch (operation) {

            case N_OPEN:

                /* see if we have to to produce a summary report */
                /* if so, name all the noise generators */

                if (job->NStpsSm != 0) {
                    switch (mode) {

                    case N_DENS:
                        for (i = 0; i < JFETNSRCS; i++) {
                            NOISE_ADD_OUTVAR(ckt, data, "onoise_%s%s", inst->JFETname, JFETnNames[i]);
                        }
                        break;

                    case INT_NOIZ:
                        for (i = 0; i < JFETNSRCS; i++) {
                            NOISE_ADD_OUTVAR(ckt, data, "onoise_total_%s%s", inst->JFETname, JFETnNames[i]);
                            NOISE_ADD_OUTVAR(ckt, data, "inoise_total_%s%s", inst->JFETname, JFETnNames[i]);
                        }
                        break;
                    }
                }
                break;

            case N_CALC:
                switch (mode) {

                case N_DENS:

                    if (inst->JFETtempGiven)
                        dtemp = inst->JFETtemp - ckt->CKTtemp + (model->JFETtnom-CONSTCtoK);
                    else
                        dtemp = inst->JFETdtemp;

                    NevalSrcInstanceTemp(&noizDens[JFETRDNOIZ],&lnNdens[JFETRDNOIZ],
                        ckt, THERMNOISE, inst->JFETdrainPrimeNode, inst->JFETdrainNode,
                        model->JFETdrainConduct * inst->JFETarea * inst->JFETm, dtemp);

                    NevalSrcInstanceTemp(&noizDens[JFETRSNOIZ],&lnNdens[JFETRSNOIZ],
                        ckt, THERMNOISE, inst->JFETsourcePrimeNode,
                        inst->JFETsourceNode, model->JFETsourceConduct *
                        inst->JFETarea * inst->JFETm, dtemp);

                    if (model->JFETnlev < 3) {
                        NevalSrcInstanceTemp(&noizDens[JFETIDNOIZ],&lnNdens[JFETIDNOIZ],
                            ckt, THERMNOISE, inst->JFETdrainPrimeNode,
                            inst->JFETsourcePrimeNode,
                            (2.0 / 3.0 * inst->JFETm * fabs(*(ckt->CKTstate0 + inst->JFETgm))), dtemp);
                    } else {
                        vgs = *(ckt->CKTstate0 + inst->JFETvgs);
                        vds = vgs - *(ckt->CKTstate0 + inst->JFETvgd);
                        vgst = vgs - inst->JFETtThreshold;
                        if (vgst >= vds)
                            alpha = 1 - vds / vgst; /* linear region */
                        else
                            alpha = 0; /* saturation region */
                        beta = inst->JFETtBeta * inst->JFETarea * inst->JFETm;

                        NevalSrcInstanceTemp(&noizDens[JFETIDNOIZ],&lnNdens[JFETIDNOIZ],
                            ckt, THERMNOISE, inst->JFETdrainPrimeNode,
                            inst->JFETsourcePrimeNode,
                            (2.0 / 3.0*beta*vgst*(1 + alpha + alpha*alpha) / (1 + alpha) * model->JFETgdsnoi), dtemp);
                    }

                    NevalSrc(&noizDens[JFETFLNOIZ], NULL, ckt,
                        N_GAIN, inst->JFETdrainPrimeNode,
                        inst->JFETsourcePrimeNode, (double) 0.0);
                    noizDens[JFETFLNOIZ] *= inst->JFETm * model->JFETfNcoef *
                        exp(model->JFETfNexp *
                            log(MAX(fabs(*(ckt->CKTstate0 + inst->JFETcd)), N_MINLOG))) /
                        data->freq;
                    lnNdens[JFETFLNOIZ] =
                        log(MAX(noizDens[JFETFLNOIZ], N_MINLOG));

                    noizDens[JFETTOTNOIZ] = noizDens[JFETRDNOIZ] +
                        noizDens[JFETRSNOIZ] +
                        noizDens[JFETIDNOIZ] +
                        noizDens[JFETFLNOIZ];
                    lnNdens[JFETTOTNOIZ] =
                        log(MAX(noizDens[JFETTOTNOIZ], N_MINLOG));

                   *OnDens += noizDens[JFETTOTNOIZ];

                    if (data->delFreq == 0.0) {

                        /* if we haven't done any previous integration, we need to */
                        /* initialize our "history" variables                      */

                        for (i = 0; i < JFETNSRCS; i++) {
                            inst->JFETnVar[LNLSTDENS][i] = lnNdens[i];
                        }

                        /* clear out our integration variables if it's the first pass */

                        if (data->freq == job->NstartFreq) {
                            for (i = 0; i < JFETNSRCS; i++) {
                                inst->JFETnVar[OUTNOIZ][i] = 0.0;
                                inst->JFETnVar[INNOIZ][i] = 0.0;
                            }
                        }
                    } else {
                        /* data->delFreq != 0.0 (we have to integrate) */
                        for (i = 0; i < JFETNSRCS; i++) {
                            if (i != JFETTOTNOIZ) {
                                tempOnoise = Nintegrate(noizDens[i], lnNdens[i],
                                    inst->JFETnVar[LNLSTDENS][i], data);
                                tempInoise = Nintegrate(noizDens[i] * data->GainSqInv,
                                    lnNdens[i] + data->lnGainInv,
                                    inst->JFETnVar[LNLSTDENS][i] + data->lnGainInv,
                                    data);
                                inst->JFETnVar[LNLSTDENS][i] = lnNdens[i];
                                data->outNoiz += tempOnoise;
                                data->inNoise += tempInoise;
                                if (job->NStpsSm != 0) {
                                    inst->JFETnVar[OUTNOIZ][i] += tempOnoise;
                                    inst->JFETnVar[OUTNOIZ][JFETTOTNOIZ] += tempOnoise;
                                    inst->JFETnVar[INNOIZ][i] += tempInoise;
                                    inst->JFETnVar[INNOIZ][JFETTOTNOIZ] += tempInoise;
                                }
                            }
                        }
                    }
                    if (data->prtSummary) {
                        for (i = 0; i < JFETNSRCS; i++) {
                            /* print a summary report */
                            data->outpVector[data->outNumber++] = noizDens[i];
                        }
                    }
                    break;

                case INT_NOIZ:
                    /* already calculated, just output */
                    if (job->NStpsSm != 0) {
                        for (i = 0; i < JFETNSRCS; i++) {
                            data->outpVector[data->outNumber++] = inst->JFETnVar[OUTNOIZ][i];
                            data->outpVector[data->outNumber++] = inst->JFETnVar[INNOIZ][i];
                        }
                    } /* if */
                    break;
                } /* switch (mode) */
                break;

            case N_CLOSE:
                return (OK); /* do nothing, the main calling routine will close */
                break; /* the plots */
            } /* switch (operation) */
        } /* for inst */
    } /* for model */

    return (OK);
}