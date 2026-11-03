/**********
Enhancement-130: a built-in Nelder-Mead optimizer.

`optimize` varies a set of circuit/device parameters, re-runs a user-chosen
analysis, and minimizes a user-supplied objective expression -- a derivative-free
downhill-simplex search implemented in normalized [0,1] parameter space (so it is
scale-invariant across parameters that span orders of magnitude).

Syntax (in a .control block, after the circuit is loaded):

  optimize -param <name> <init> <lo> <hi>  [-param ...]
           -analysis <command ...>
           -minimize <expression ...>
           [-maxiter <N>] [-tol <T>] [-verbose]

Each <name> is an `alter` target -- a device instance (e.g. R1, C1) or a
parameter (e.g. @m1[w]). For every candidate the optimizer applies each value in
place with `alter <name>=<value>`, runs the `-analysis` command, and evaluates the
`-minimize` expression (its last value is the scalar cost). `-analysis` and
`-minimize` collect every following token up to the next `-<letter>` flag, so
multi-word commands/expressions need no quoting; negative bounds (e.g. `-5`) are
`-`+digit and are not mistaken for flags. Console chatter from the hundreds of
inner analyses is suppressed (via ft_optimizing) unless `-verbose`.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"

#include "com_optimize.h"

#define OPT_MAXP     16          /* max parameters to optimize */
#define OPT_PENALTY  1e30        /* cost returned for a failed / non-finite eval */

struct optctx {
    int np;
    char *name[OPT_MAXP];
    double lo[OPT_MAXP], hi[OPT_MAXP], x0[OPT_MAXP];
    char *analysis;              /* analysis command, e.g. "tran 1u 1m" */
    char *objective;             /* expression to minimize */
    int maxiter;
    double tol;
    int verbose;
    int nevals;
};


static double clamp01(double u)
{
    return u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
}


/* Run one command SYNCHRONOUSLY by dispatching straight through the command
 * table. Unlike cp_evloop(), which (called re-entrantly) defers the command to
 * the outer interpreter loop -- so it would run after the optimizer returns, with
 * the quiet flag already cleared -- this executes it now, inside opt_eval. */
static void opt_run_cmd(const char *cmdstr)
{
    wordlist *wl = cp_lexer((char *) cmdstr);   /* tokenize on whitespace */
    int i;

    if (!wl || !wl->wl_word) {
        if (wl) wl_free(wl);
        return;
    }
    for (i = 0; cp_coms[i].co_comname; i++)
        if (strcasecmp(cp_coms[i].co_comname, wl->wl_word) == 0)
            break;
    if (cp_coms[i].co_comname && cp_coms[i].co_func)
        cp_coms[i].co_func(wl->wl_next);
    else
        fprintf(cp_err, "optimize: unknown command '%s'\n", wl->wl_word);
    wl_free(wl);
}


/* parse a SPICE-style number (understands k / meg / u / n / p ... suffixes) */
static double optnum(const char *w)
{
    char *s = (char *) w;
    double v = 0.0;
    if (ft_numparse(&s, FALSE, &v) < 0)
        v = atof(w);
    return v;
}


/* Evaluate the objective at a normalized point u in [0,1]^np: alter each param
 * in place, run the analysis, evaluate the objective expression. */
static double opt_eval(struct optctx *c, const double *u)
{
    int k;
    char cmd[512];
    struct pnode *pn;
    double f = OPT_PENALTY;

    /* Silence the per-iteration console chatter (alter's re-setup banner, the
     * analysis banner, row count, reference-value progress) unless -verbose.
     * ft_optimizing gates those prints at their source -- the analyses write to
     * stdout directly, and docommand's cp_ioreset() would undo an external fd
     * redirect. `alter` changes the value in place (no re-source), so the flag
     * set here survives through to the analysis. */
    ft_optimizing = !c->verbose;

    for (k = 0; k < c->np; k++) {
        double val = c->lo[k] + clamp01(u[k]) * (c->hi[k] - c->lo[k]);
        (void) snprintf(cmd, sizeof cmd, "alter %s=%.10g", c->name[k], val);
        opt_run_cmd(cmd);
    }
    opt_run_cmd(c->analysis);

    c->nevals++;

    /* evaluate the objective while still quiet -- reading a result vector can
     * re-trigger the analysis (which would print its banner) */
    pn = ft_getpnames_from_string(c->objective, TRUE);
    if (pn) {
        struct dvec *v = ft_evaluate(pn);
        if (v && v->v_length >= 1) {
            if (isreal(v))
                f = v->v_realdata[v->v_length - 1];
            else
                f = hypot(v->v_compdata[v->v_length - 1].cx_real,
                          v->v_compdata[v->v_length - 1].cx_imag);
            if (!finite(f))
                f = OPT_PENALTY;
        }
        /* garbage-collect the temporary vector ft_evaluate may have created
         * (mirrors com_let), so hundreds of evaluations do not leak */
        if (!pn->pn_value && v)
            vec_free(v);
        free_pnode(pn);
    }

    ft_optimizing = FALSE;
    return f;
}


/* Nelder-Mead downhill simplex over the np normalized parameters. On entry
 * ubest holds the normalized starting point; on exit it holds the best point
 * and *fbest its cost. */
static void nelder_mead(struct optctx *c, double *ubest, double *fbest)
{
    const double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
    const int n = c->np;
    double s[OPT_MAXP + 1][OPT_MAXP], fv[OPT_MAXP + 1];
    double cent[OPT_MAXP], xr[OPT_MAXP], xe[OPT_MAXP], xc[OPT_MAXP];
    int i, j, iter, lo;

    /* build the initial simplex: the start point plus one point per dimension
     * nudged by 0.1 in normalized space */
    for (j = 0; j < n; j++)
        s[0][j] = clamp01(ubest[j]);
    fv[0] = opt_eval(c, s[0]);
    for (i = 1; i <= n; i++) {
        for (j = 0; j < n; j++)
            s[i][j] = s[0][j];
        double b = s[0][i - 1] + 0.1;
        if (b > 1.0)
            b = s[0][i - 1] - 0.1;
        s[i][i - 1] = clamp01(b);
        fv[i] = opt_eval(c, s[i]);
    }

    for (iter = 0; iter < c->maxiter; iter++) {
        int hi, nh;
        double fr;

        lo = hi = 0;
        for (i = 1; i <= n; i++) {
            if (fv[i] < fv[lo]) lo = i;
            if (fv[i] > fv[hi]) hi = i;
        }
        nh = (hi == 0) ? 1 : 0;
        for (i = 0; i <= n; i++)
            if (i != hi && fv[i] > fv[nh]) nh = i;

        if (fv[hi] - fv[lo] <= c->tol * (fabs(fv[lo]) + c->tol))
            break;                               /* converged */

        for (j = 0; j < n; j++) {                /* centroid of all but worst */
            double sum = 0.0;
            for (i = 0; i <= n; i++)
                if (i != hi) sum += s[i][j];
            cent[j] = sum / n;
        }

        for (j = 0; j < n; j++)                  /* reflect */
            xr[j] = clamp01(cent[j] + alpha * (cent[j] - s[hi][j]));
        fr = opt_eval(c, xr);

        if (fr < fv[lo]) {                        /* expand */
            double fe;
            for (j = 0; j < n; j++)
                xe[j] = clamp01(cent[j] + gamma * (xr[j] - cent[j]));
            fe = opt_eval(c, xe);
            if (fe < fr) {
                for (j = 0; j < n; j++) s[hi][j] = xe[j];
                fv[hi] = fe;
            } else {
                for (j = 0; j < n; j++) s[hi][j] = xr[j];
                fv[hi] = fr;
            }
        } else if (fr < fv[nh]) {                 /* accept reflection */
            for (j = 0; j < n; j++) s[hi][j] = xr[j];
            fv[hi] = fr;
        } else {                                  /* contract */
            double fc;
            for (j = 0; j < n; j++)
                xc[j] = clamp01(cent[j] + rho * (s[hi][j] - cent[j]));
            fc = opt_eval(c, xc);
            if (fc < fv[hi]) {
                for (j = 0; j < n; j++) s[hi][j] = xc[j];
                fv[hi] = fc;
            } else {                              /* shrink toward the best */
                for (i = 0; i <= n; i++)
                    if (i != lo) {
                        for (j = 0; j < n; j++)
                            s[i][j] = clamp01(s[lo][j] + sigma * (s[i][j] - s[lo][j]));
                        fv[i] = opt_eval(c, s[i]);
                    }
            }
        }
        if (c->verbose)
            fprintf(cp_out, "  iter %-3d  best cost %.6g  (%d evals)\n",
                    iter + 1, fv[lo], c->nevals);
    }

    lo = 0;
    for (i = 1; i <= n; i++)
        if (fv[i] < fv[lo]) lo = i;
    for (j = 0; j < n; j++)
        ubest[j] = s[lo][j];
    *fbest = fv[lo];
}


static int is_flag(const char *w)
{
    return w && w[0] == '-' && isalpha((unsigned char) w[1]);
}


/* collect tokens from *pwl up to the next flag, joined with single spaces */
static char *collect_until_flag(wordlist **pwl)
{
    char *acc = NULL;
    wordlist *wl = *pwl;
    while (wl && !is_flag(wl->wl_word)) {
        if (!acc) {
            acc = copy(wl->wl_word);
        } else {
            char *j = tprintf("%s %s", acc, wl->wl_word);
            tfree(acc);
            acc = j;
        }
        wl = wl->wl_next;
    }
    *pwl = wl;
    return acc;
}


void com_optimize(wordlist *wl)
{
    struct optctx c;
    double ubest[OPT_MAXP], fbest = OPT_PENALTY;
    int k;

    memset(&c, 0, sizeof c);
    c.maxiter = 100;
    c.tol = 1e-6;

    while (wl) {
        const char *w = wl->wl_word;
        if (eq(w, "-param") || eq(w, "-p")) {
            if (c.np >= OPT_MAXP) {
                fprintf(cp_err, "optimize: too many -param (max %d)\n", OPT_MAXP);
                return;
            }
            wordlist *a = wl->wl_next, *b = a ? a->wl_next : NULL;
            wordlist *d = b ? b->wl_next : NULL, *e = d ? d->wl_next : NULL;
            if (!a || !b || !d || !e) {
                fprintf(cp_err, "optimize: -param needs <name> <init> <lo> <hi>\n");
                return;
            }
            c.name[c.np] = copy(a->wl_word);
            c.x0[c.np]   = optnum(b->wl_word);
            c.lo[c.np]   = optnum(d->wl_word);
            c.hi[c.np]   = optnum(e->wl_word);
            if (c.hi[c.np] <= c.lo[c.np]) {
                fprintf(cp_err, "optimize: param '%s' needs hi > lo\n", c.name[c.np]);
                return;
            }
            c.np++;
            wl = e->wl_next;
        } else if (eq(w, "-analysis") || eq(w, "-a")) {
            wl = wl->wl_next;
            tfree(c.analysis);
            c.analysis = collect_until_flag(&wl);
        } else if (eq(w, "-minimize") || eq(w, "-min") || eq(w, "-o")) {
            wl = wl->wl_next;
            tfree(c.objective);
            c.objective = collect_until_flag(&wl);
        } else if (eq(w, "-maxiter") || eq(w, "-n")) {
            if (wl->wl_next) { c.maxiter = atoi(wl->wl_next->wl_word); wl = wl->wl_next->wl_next; }
            else wl = NULL;
        } else if (eq(w, "-tol") || eq(w, "-t")) {
            if (wl->wl_next) { c.tol = atof(wl->wl_next->wl_word); wl = wl->wl_next->wl_next; }
            else wl = NULL;
        } else if (eq(w, "-verbose") || eq(w, "-v")) {
            c.verbose = 1;
            wl = wl->wl_next;
        } else {
            fprintf(cp_err, "optimize: unrecognized token '%s'\n", w);
            wl = wl->wl_next;
        }
    }

    if (c.np < 1 || !c.analysis || !c.objective) {
        fprintf(cp_err, "usage: optimize -param <name> <init> <lo> <hi> [-param ...] "
                        "-analysis <cmd> -minimize <expr> [-maxiter N] [-tol T] [-verbose]\n");
        goto cleanup;
    }
    if (c.maxiter < 1) c.maxiter = 1;
    if (c.tol <= 0.0) c.tol = 1e-6;

    if (!ft_curckt || !ft_curckt->ci_ckt) {
        fprintf(cp_err, "optimize: no circuit loaded\n");
        goto cleanup;
    }

    fprintf(cp_out, "optimize: %d parameter%s, analysis '%s', minimizing '%s'\n",
            c.np, c.np == 1 ? "" : "s", c.analysis, c.objective);

    for (k = 0; k < c.np; k++)
        ubest[k] = clamp01((c.x0[k] - c.lo[k]) / (c.hi[k] - c.lo[k]));

    nelder_mead(&c, ubest, &fbest);

    /* leave the circuit at the optimum (verbose final run) and report */
    c.verbose = 1;
    (void) opt_eval(&c, ubest);

    fprintf(cp_out, "optimize: converged, objective = %.6g after %d evaluations\n",
            fbest, c.nevals);
    for (k = 0; k < c.np; k++) {
        double val = c.lo[k] + ubest[k] * (c.hi[k] - c.lo[k]);
        fprintf(cp_out, "    %s = %.6g\n", c.name[k], val);
    }

cleanup:
    for (k = 0; k < c.np; k++)
        tfree(c.name[k]);
    tfree(c.analysis);
    tfree(c.objective);
}
