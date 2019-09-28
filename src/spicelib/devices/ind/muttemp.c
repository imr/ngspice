/**********
Copyright 2003 Paolo Nenzi
Author: 2003 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


static int
cholesky(double *a, int n)
{
#define A(r,c) a[n*r + c]
    int i, j, k;
    for (i = 0; i < n; i++)
        for (j = 0; j <= i; j++) {
            double Summe = A(i, j);
            for (k = 0; k < j; k++)
                Summe -= A(i, k) * A(j, k);
            if (i > j)
                A(i, j) = Summe / A(j, j);
            else if (Summe > 0)
                A(i, i) = sqrt(Summe);
            else
                return 0;
        }
    return 1;
#undef A
}


int
MUTtemp(GENmodel *inModel, CKTcircuit *ckt)
{
    MUTmodel *model = (MUTmodel*) inModel;
    MUTinstance *here;

    struct INDsystem *first_system = NULL;

    NG_IGNORE(ckt);

    for (; model; model = MUTnextModel(model))
        for (here = MUTinstances(model); here; here = MUTnextInstance(here)) {

            /* Value Processing for mutual inductors */

            double ind1 = here->MUTind1->INDinduct;
            double ind2 = here->MUTind2->INDinduct;

            /*           _______
             * M = k * \/L1 * L2
             */
            here->MUTfactor = here->MUTcoupling * sqrt(fabs(ind1 * ind2));

            if (ckt->CKTindverbosity > 0) {

                struct INDsystem *system;

                if (!here->MUTind1->system && !here->MUTind2->system) {
                    system = TMALLOC (struct INDsystem, 1);
                    system->size = 2;
                    system->next_system = first_system;
                    first_system = system;
                    system->first_ind = here->MUTind1;
                    here->MUTind1->system_next_ind = here->MUTind2;
                    here->MUTind2->system_next_ind = NULL;
                    here->MUTind1->system = system;
                    here->MUTind2->system = system;
                    system->first_mut = here;
                    here->system_next_mut = NULL;
                } else if (here->MUTind1->system && !here->MUTind2->system) {
                    system = here->MUTind1->system;
                    system->size++;
                    here->MUTind2->system_next_ind = system->first_ind;
                    system->first_ind = here->MUTind2;
                    here->system_next_mut = system->first_mut;
                    system->first_mut = here;
                    here->MUTind2->system = system;
                } else if (!here->MUTind1->system && here->MUTind2->system) {
                    system = here->MUTind2->system;
                    system->size++;
                    here->MUTind1->system_next_ind = system->first_ind;
                    system->first_ind = here->MUTind1;
                    here->system_next_mut = system->first_mut;
                    system->first_mut = here;
                    here->MUTind1->system = system;
                } else if (here->MUTind1->system == here->MUTind2->system) {
                    system = here->MUTind2->system;
                    here->system_next_mut = system->first_mut;
                    system->first_mut = here;
                } else {
                    struct INDsystem *s1 = here->MUTind1->system;
                    struct INDsystem *s2 = here->MUTind2->system;
                    MUTinstance *mut;
                    INDinstance *ind;
                    /* append s2 to s1, leave a consumed s2 behind */
                    s1->size += s2->size;
                    s2->size = 0;
                    for (ind = s2->first_ind; ind; ind = ind->system_next_ind) {
                        ind->system = s1;
                        if (!ind->system_next_ind)
                            break;
                    }
                    ind->system_next_ind = s1->first_ind;
                    s1->first_ind = s2->first_ind;
                    s2->first_ind = NULL;
                    for (mut = s2->first_mut; mut; mut = mut->system_next_mut)
                        if (!mut->system_next_mut)
                            break;
                    mut->system_next_mut = s1->first_mut;
                    here->system_next_mut = s2->first_mut;
                    s1->first_mut = here;
                    s2->first_mut = NULL;
                }
            }
        }

    if (first_system) {
        struct INDsystem *system;
        int sz = 0;

        for (system = first_system; system; system = system->next_system)
            if (sz < system->size)
                sz = system->size;

        char *pop = TMALLOC(char, sz * sz);
        double *INDmatrix = TMALLOC(double, sz * sz);

        for (system = first_system; system; system = system->next_system) {
            if (!system->size)
                continue;

            int positive, i;

            sz = system->size;

            memset(pop, 0, (size_t)(sz*sz));
            memset(INDmatrix, 0, (size_t)(sz*sz) * sizeof(double));

            INDinstance *ind = system->first_ind;
            for (i = 0; ind; ind = ind->system_next_ind) {
                INDmatrix [i * sz + i] = ind->INDinduct;
                ind->system_idx = i++;
            }

            MUTinstance *mut = system->first_mut;
            int expect = (sz*sz - sz) / 2;
            int repetitions = 0;
            for (; mut; mut = mut->system_next_mut) {
                int j = mut->MUTind1->system_idx;
                int k = mut->MUTind2->system_idx;
                if (j < k)
                    SWAP(int, j, k);
                if (pop[j*sz + k]) {
                    repetitions ++;
                } else {
                    pop[j*sz + k] = 1;
                    expect --;
                }
                INDmatrix [j * sz + k] = INDmatrix [k * sz + j] = mut->MUTfactor;
            }

            positive = cholesky(INDmatrix, sz);

            if (!positive) {
                positive = 1;
                /* ignore check if all |K| == 1 and all L >= 0 */
                for (mut = system->first_mut; mut; mut = mut->system_next_mut)
                    if (fabs(mut->MUTcoupling) != 1.0) {
                        positive = 0;
                        break;
                    }
                for (ind = system->first_ind; ind; ind = ind->system_next_ind)
                    if (ind->INDinduct < 0) {
                        positive = 0;
                        break;
                    }
            }

            if (!positive || repetitions || (expect && ckt->CKTindverbosity > 1)) {
                fprintf(stderr, "The Inductive System consisting of\n");
                for (ind = system->first_ind; ind; ind = ind->system_next_ind)
                    fprintf(stderr, " %s", ind->INDname);
                fprintf(stderr, "\n");
                for (mut = system->first_mut; mut; mut = mut->system_next_mut)
                    fprintf(stderr, " %s", mut->MUTname);
                fprintf(stderr, "\n");
                if (!positive)
                    fprintf(stderr, "is not positive definite\n");
                for (mut = system->first_mut; mut; mut = mut->system_next_mut)
                    if (fabs(mut->MUTcoupling) > 1.0)
                        fprintf(stderr, " |%s| > 1\n", mut->MUTname);
                for (ind = system->first_ind; ind; ind = ind->system_next_ind)
                    if (ind->INDinduct < 0)
                        fprintf(stderr, " %s < 0\n", ind->INDname);
                if (repetitions)
                    fprintf(stderr, "has duplicate K instances\n");
                if (expect && ckt->CKTindverbosity > 1)
                    fprintf(stderr, "has an incomplete set of K couplings, (missing ones are implicitly 0)\n");
                fprintf(stderr, "\n");
            }
        }

        tfree(pop);
        tfree(INDmatrix);

        for (system = first_system; system;) {
            struct INDsystem *next_system = system->next_system;
            tfree(system);
            system = next_system;
        }
    }

    return(OK);
}
