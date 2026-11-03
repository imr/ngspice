/**********
Enhancement-131: transient checkpoint / restart.

Two interactive commands let a (long) transient run be saved to disk and later
resumed -- possibly in a fresh ngspice process -- so a multi-hour analysis can
survive a crash, be split across sessions, or be moved to another machine:

    savestate <file>    dump the current transient state of the active circuit
    loadstate <file>    restore it into the (identically built) active circuit
                        and continue the .tran defined in the deck

The saved state is exactly the state the in-memory `resume` path relies on: the
solution vector (CKTrhsOld), the integration history (CKTstates[]), the current
time / step / order and the pending breakpoints.  On reload we rebuild the
circuit (CKTsetup/CKTtemp), pour the saved state back in, and drive DCtran()
through its Enhancement-131 "checkpoint" branch, which keeps that state and
opens a fresh output plot for the continued run.

Checkpoint files are binary and specific to the machine/build that wrote them
(native type sizes); a mismatched circuit is rejected via a stored signature.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/wordlist.h"
#include "ngspice/cpextern.h"
#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/smpdefs.h"

#include "circuits.h"
#include "spiceif.h"
#include "com_checkpoint.h"

/* 8-byte magic + version so a stale/foreign file is caught early. */
static const char CKPT_MAGIC[8] = { 'N', 'G', 'C', 'K', 'P', 'T', '0', '1' };

#define WR(ptr, n)                                                      \
    do {                                                                \
        if (fwrite((ptr), sizeof(*(ptr)), (size_t)(n), fp) != (size_t)(n)) \
            goto io_err;                                                \
    } while (0)

#define RD(ptr, n)                                                      \
    do {                                                                \
        if (fread((ptr), sizeof(*(ptr)), (size_t)(n), fp) != (size_t)(n)) \
            goto io_err;                                                 \
    } while (0)


/* Find the .tran job attached to the circuit's default task, or NULL. */
static JOB *
find_tran_job(void)
{
    int tran_type;
    JOB *job;

    if (!ft_curckt || !ft_curckt->ci_defTask)
        return NULL;

    tran_type = ft_find_analysis("TRAN");   /* registered name is upper-case */
    if (tran_type < 0)
        return NULL;

    for (job = ft_curckt->ci_defTask->jobs; job; job = job->JOBnextJob)
        if (job->JOBtype == tran_type)
            return job;

    return NULL;
}


void
com_savestate(wordlist *wl)
{
    CKTcircuit *ckt;
    FILE *fp;
    const char *fname;
    int i, present, matSize;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: savestate: there is no circuit loaded.\n");
        return;
    }
    ckt = ft_curckt->ci_ckt;

    if (!wl || !wl->wl_word || !*wl->wl_word) {
        fprintf(cp_err, "Usage: savestate <file>\n");
        return;
    }
    fname = wl->wl_word;

    if (ckt->CKTstates[0] == NULL || ckt->CKTmaxEqNum <= 0) {
        fprintf(cp_err,
                "Error: savestate: no simulation state to save; run a transient first.\n");
        return;
    }
    if (ckt->CKTtime <= 0.0) {
        fprintf(cp_err,
                "Error: savestate: transient time is 0; nothing to checkpoint.\n");
        return;
    }
    if (ckt->CKTmatrix && ckt->CKTmatrix->CKTkluMODE) {
        fprintf(cp_err,
                "Error: savestate: checkpoint/restart is only supported with the "
                "default Sparse solver, not KLU (remove '.option klu').\n");
        return;
    }

    fp = fopen(fname, "wb");
    if (!fp) {
        perror(fname);
        return;
    }

    /* The solution vectors CKTrhsOld/CKTirhsOld are allocated with
       SMPmatSize(matrix)+1 entries (see NIreinit), which is NOT the same as
       CKTmaxEqNum+1 -- store the matrix size and use it for those vectors. */
    matSize = ckt->CKTmatrix ? SMPmatSize(ckt->CKTmatrix) : ckt->CKTmaxEqNum;

    /* header: magic + topology signature */
    WR(CKPT_MAGIC, 8);
    WR(&ckt->CKTmaxEqNum, 1);
    WR(&matSize, 1);
    WR(&ckt->CKTnumStates, 1);
    WR(&ckt->CKTmaxOrder, 1);
    WR(&ckt->CKTbreakSize, 1);

    /* scalar integration state */
    WR(&ckt->CKTorder, 1);
    WR(&ckt->CKTmode, 1);
    WR(&ckt->CKTtime, 1);
    WR(&ckt->CKTdelta, 1);
    WR(&ckt->CKTsaveDelta, 1);
    WR(&ckt->CKTdelmin, 1);
    WR(&ckt->CKTminBreak, 1);
    WR(ckt->CKTdeltaOld, 7);

    /* pending breakpoints */
    if (ckt->CKTbreakSize > 0 && ckt->CKTbreaks)
        WR(ckt->CKTbreaks, ckt->CKTbreakSize);

    /* solution vectors (size matSize + 1) */
    present = (ckt->CKTrhsOld != NULL);
    WR(&present, 1);
    if (present)
        WR(ckt->CKTrhsOld, matSize + 1);

    present = (ckt->CKTirhsOld != NULL);
    WR(&present, 1);
    if (present)
        WR(ckt->CKTirhsOld, matSize + 1);

    /* integration history: the (up to eight) device state vectors */
    for (i = 0; i < 8; i++) {
        present = (ckt->CKTstates[i] != NULL);
        WR(&present, 1);
        if (present && ckt->CKTnumStates > 0)
            WR(ckt->CKTstates[i], ckt->CKTnumStates);
    }

    (void) fclose(fp);
    fprintf(cp_out, "Checkpoint written to \"%s\" at t = %g s.\n",
            fname, ckt->CKTtime);
    return;

io_err:
    fprintf(cp_err, "Error: savestate: write failed on \"%s\".\n", fname);
    (void) fclose(fp);
}


void
com_loadstate(wordlist *wl)
{
    CKTcircuit *ckt;
    FILE *fp;
    const char *fname;
    char magic[8];
    int maxEqNum, matSize, numStates, maxOrder, breakSize;
    int i, present, err;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "Error: loadstate: there is no circuit loaded.\n");
        return;
    }
    ckt = ft_curckt->ci_ckt;

    if (!wl || !wl->wl_word || !*wl->wl_word) {
        fprintf(cp_err, "Usage: loadstate <file>\n");
        return;
    }
    fname = wl->wl_word;

    if (find_tran_job() == NULL) {
        fprintf(cp_err,
                "Error: loadstate: the deck needs a '.tran' line to define the "
                "continuation (its tstop is the new end time).\n");
        return;
    }

    fp = fopen(fname, "rb");
    if (!fp) {
        perror(fname);
        return;
    }

    RD(magic, 8);
    if (memcmp(magic, CKPT_MAGIC, 8) != 0) {
        fprintf(cp_err, "Error: loadstate: \"%s\" is not an ngspice checkpoint.\n", fname);
        (void) fclose(fp);
        return;
    }
    RD(&maxEqNum, 1);
    RD(&matSize, 1);
    RD(&numStates, 1);
    RD(&maxOrder, 1);
    RD(&breakSize, 1);

    /* Make sure the circuit is built so the state vectors exist to fill.  A
       freshly sourced deck has not been set up yet; do it now (once). */
    if (ckt->CKTstates[0] == NULL || ckt->CKTmaxEqNum <= 0) {
        if ((err = CKTsetup(ckt)) != OK) {
            fprintf(cp_err, "Error: loadstate: circuit setup failed.\n");
            (void) fclose(fp);
            return;
        }
        if ((err = CKTtemp(ckt)) != OK) {
            fprintf(cp_err, "Error: loadstate: circuit temperature setup failed.\n");
            (void) fclose(fp);
            return;
        }
    }

    if (ckt->CKTmatrix && ckt->CKTmatrix->CKTkluMODE) {
        fprintf(cp_err,
                "Error: loadstate: checkpoint/restart is only supported with the "
                "default Sparse solver, not KLU (remove '.option klu').\n");
        (void) fclose(fp);
        return;
    }

    /* signature check: the checkpoint must belong to this exact circuit */
    if (maxEqNum != ckt->CKTmaxEqNum ||
        numStates != ckt->CKTnumStates ||
        maxOrder != ckt->CKTmaxOrder) {
        fprintf(cp_err,
                "Error: loadstate: checkpoint does not match this circuit "
                "(equations %d vs %d, states %d vs %d).\n",
                maxEqNum, ckt->CKTmaxEqNum, numStates, ckt->CKTnumStates);
        (void) fclose(fp);
        return;
    }

    /* The solution vectors (CKTrhsOld/CKTirhsOld) are sized SMPmatSize(matrix)+1
       by CKTsetup/NIreinit.  That length must equal the one stored in the file,
       or the reads below would overrun the buffers.  (It matches whenever the
       restore circuit is identical to the saved one.)  */
    if (ckt->CKTmatrix && SMPmatSize(ckt->CKTmatrix) != matSize) {
        fprintf(cp_err,
                "Error: loadstate: checkpoint matrix size %d does not match "
                "this circuit (%d).\n", matSize, SMPmatSize(ckt->CKTmatrix));
        (void) fclose(fp);
        return;
    }

    /* scalar integration state */
    RD(&ckt->CKTorder, 1);
    RD(&ckt->CKTmode, 1);
    RD(&ckt->CKTtime, 1);
    RD(&ckt->CKTdelta, 1);
    RD(&ckt->CKTsaveDelta, 1);
    RD(&ckt->CKTdelmin, 1);
    RD(&ckt->CKTminBreak, 1);
    RD(ckt->CKTdeltaOld, 7);

    /* pending breakpoints */
    if (breakSize > 0) {
        if (ckt->CKTbreaks)
            FREE(ckt->CKTbreaks);
        ckt->CKTbreaks = TMALLOC(double, breakSize);
        if (ckt->CKTbreaks == NULL) {
            fprintf(cp_err, "Error: loadstate: out of memory.\n");
            (void) fclose(fp);
            return;
        }
        ckt->CKTbreakSize = breakSize;
        RD(ckt->CKTbreaks, breakSize);
    }

    /* solution vectors (size matSize + 1) */
    RD(&present, 1);
    if (present) {
        if (ckt->CKTrhsOld == NULL)
            goto mismatch;
        RD(ckt->CKTrhsOld, matSize + 1);
    }
    RD(&present, 1);
    if (present) {
        if (ckt->CKTirhsOld == NULL)
            goto mismatch;
        RD(ckt->CKTirhsOld, matSize + 1);
    }

    /* integration history */
    for (i = 0; i < 8; i++) {
        RD(&present, 1);
        if (present) {
            if (ckt->CKTstates[i] == NULL)
                goto mismatch;
            if (ckt->CKTnumStates > 0)
                RD(ckt->CKTstates[i], ckt->CKTnumStates);
        }
    }

    (void) fclose(fp);

    /* Drive the continuation.  DCtran() sees CKTcheckpoint and keeps the state
       we just restored while opening a fresh output plot for the new run; the
       deck's .tran tstop (read by TRANinit) becomes the end time. */
    ckt->CKTcheckpoint = 1;
    ft_curckt->ci_curTask = ft_curckt->ci_defTask;
    ft_curckt->ci_curOpt  = ft_curckt->ci_defOpt;
    ft_curckt->ci_inprogress = TRUE;

    fprintf(cp_out, "Restored checkpoint from \"%s\"; continuing transient from t = %g s.\n",
            fname, ckt->CKTtime);

    err = if_run(ckt, "resume", NULL, ft_curckt->ci_symtab);

    ckt->CKTcheckpoint = 0;   /* defensive: clear if the run never consumed it */
    if (err == 1) {
        fprintf(cp_err, "loadstate: simulation interrupted.\n");
    } else if (err == 2) {
        fprintf(cp_err, "loadstate: simulation aborted.\n");
        ft_curckt->ci_inprogress = FALSE;
    } else {
        ft_curckt->ci_inprogress = FALSE;
    }
    return;

mismatch:
    fprintf(cp_err, "Error: loadstate: checkpoint state layout does not match circuit.\n");
    (void) fclose(fp);
    return;

io_err:
    fprintf(cp_err, "Error: loadstate: read failed on \"%s\".\n", fname);
    (void) fclose(fp);
}
