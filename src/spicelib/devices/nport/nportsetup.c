/**********
Enhancement-242: native n-port device -- setup.

Loads the model's `.nport` fit file (once), then for each instance allocates the
direct admittance-stamp matrix elements (a multi-terminal conductance; no branch
currents / extra unknowns -- so it scales cleanly to many ports).
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "nportdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* pure-C reader in nportread.c */
extern int snp_nport_read(const char *path, int *pN, int *pNp,
                          double **ppoleRe, double **ppoleIm,
                          double **pd, double **pe,
                          double **presRe, double **presIm,
                          char *err, int errlen);

/* fill a model from its .nport file */
int
NPORTreadFile(NPORTmodel *model)
{
    char err[256];
    if (model->NPORTloaded)
        return OK;
    if (!model->NPORTfileGiven || !model->NPORTfile) {
        fprintf(stderr, "nport: model '%s' has no file= parameter\n",
                model->NPORTmodName);
        return E_BADPARM;
    }
    if (snp_nport_read(model->NPORTfile,
                       &model->NPORTnPorts, &model->NPORTnPoles,
                       &model->NPORTpoleRe, &model->NPORTpoleIm,
                       &model->NPORTd, &model->NPORTe,
                       &model->NPORTresRe, &model->NPORTresIm,
                       err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return E_BADPARM;
    }
    /* The instance wires up N ports + 1 reference, so the model's port count must
     * leave room for the reference in the fixed-size GENnode array; reject a
     * .nport file that claims more (otherwise setup would index past node[]). */
    if (model->NPORTnPorts + 1 > NPORT_MAXTERMS) {
        fprintf(stderr, "nport: model '%s' declares %d ports; the maximum is %d\n",
                model->NPORTmodName, model->NPORTnPorts, NPORT_MAXTERMS - 1);
        return E_BADPARM;
    }
    model->NPORTloaded = 1;
    return OK;
}

int
NPORTsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NPORTinstance *here;
    int error, i, j, N, Np, ref;
    int *node;

    NG_IGNORE(ckt);

    for (; model; model = NPORTnextModel(model)) {

        if ((error = NPORTreadFile(model)) != OK)
            return error;
        N = model->NPORTnPorts;
        Np = model->NPORTnPoles;

        for (here = NPORTinstances(model); here; here = NPORTnextInstance(here)) {

            here->NPORTn = N;
            node = GENnode(&here->gen);      /* ports 0..N-1, ref at [N] */

            /* The instance line must connect all N ports plus the reference; an
             * unbound (-1) terminal means it wired up fewer nodes than the model's
             * port count.  Stamping such a node would pass a negative row/col to the
             * sparse builder (spGetElement assert / out-of-bounds) -- reject it. */
            for (i = 0; i <= N; i++) {
                if (node[i] < 0) {
                    fprintf(stderr, "nport: instance '%s' connects fewer nodes than the "
                            "%d-port model '%s' needs (%d ports + reference)\n",
                            here->NPORTname, N, model->NPORTmodName, N);
                    return E_BADPARM;
                }
            }

            ref = node[N];
            here->NPORTrefNode = ref;

            /* transient companion state:
             *   poles:  4 per (input j, pole k)  [x_re, x_im, dx_re, dx_im]
             *   e-term: 2 per (i,j)              [charge, current] for NIintegrate */
            here->NPORTstateBase = *states;
            *states += 4 * N * Np + 2 * N * N;

            here->NPORTyPtr    = TMALLOC(double *, N * N);
            here->NPORTyColPtr = TMALLOC(double *, N);
            here->NPORTyRowPtr = TMALLOC(double *, N);
            here->NPORTyBind    = TMALLOC(BindElement *, N * N);  /* KLU (NULL until bound) */
            here->NPORTyColBind = TMALLOC(BindElement *, N);
            here->NPORTyRowBind = TMALLOC(BindElement *, N);
            here->NPORTyRefBind = NULL;
            here->NPORTallocated = 1;

#define TST(dst, r, c) do { \
    if ((dst = SMPmakeElt(matrix, (r), (c))) == NULL) return E_NOMEM; } while (0)

            for (i = 0; i < N; i++) {
                TST(here->NPORTyColPtr[i], node[i], ref);   /* (node_i, ref) */
                TST(here->NPORTyRowPtr[i], ref, node[i]);   /* (ref, node_i) */
                for (j = 0; j < N; j++)
                    TST(here->NPORTyPtr[i * N + j], node[i], node[j]);
            }
            TST(here->NPORTyRefPtr, ref, ref);              /* (ref, ref) */
#undef TST
        }
    }
    return OK;
}

int
NPORTunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    NPORTmodel *model = (NPORTmodel *)inModel;
    NPORTinstance *here;

    NG_IGNORE(ckt);

    for (; model; model = NPORTnextModel(model)) {
        for (here = NPORTinstances(model); here; here = NPORTnextInstance(here)) {
            if (here->NPORTallocated) {
                tfree(here->NPORTyPtr);
                tfree(here->NPORTyColPtr);
                tfree(here->NPORTyRowPtr);
                tfree(here->NPORTyBind);
                tfree(here->NPORTyColBind);
                tfree(here->NPORTyRowBind);
                here->NPORTallocated = 0;
            }
        }
    }
    return OK;
}
