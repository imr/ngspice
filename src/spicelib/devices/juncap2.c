/*************************************************************************
 *
 * JUNCAP2 200.6 junction model (current and charge), shared by the device
 * models that offer it (see ngspice/juncap2.h)
 *
 * Transcription of JUNCAP200_InitModel.include and JUNCAP200_macrodefs.include
 * (JUNCAP2 200.6.2, (c) NXP Semiconductors / CEA-Leti, see PSP 103
 * release).  Variable names follow the Verilog-A source.  The voltage
 * derivatives are obtained by forward-mode differentiation with a
 * value/derivative pair, so dI/dV and dQ/dV are consistent with I and Q.
 *
 *************************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/sperror.h"
#include "ngspice/juncap2.h"

/* physical constants as in Common103_macrodefs.include */
#define JC2_KELVINCONVERSION  273.15
#define JC2_KBOL              1.3806505e-23
#define JC2_QELE              1.6021918e-19
#define JC2_HBAR              1.05457168e-34
#define JC2_MELE              9.1093826e-31
#define JC2_EPSO              8.8541878176e-12
#define JC2_EPSRSI            11.8
#define JC2_SQRTPI            1.77245385090551603
#define JC2_SE05              2.3025850929940458e+02
#define JC2_KE05              1.0e-100
#define JC2_KE05INV           1.0e100
#define JC2_ONETHIRD          3.3333333333333333e-01

/* constants as in JUNCAP200_macrodefs.include */
#define JC2_MINTEMP          -2.5e2
#define JC2_VBILOW            5.0e-2
#define JC2_A                 2.0
#define JC2_EPSCH             0.1
#define JC2_DVBI              5.0e-2
#define JC2_EPSAV             1.0e-6
#define JC2_VBRMAX            1.0e3
#define JC2_VMAXLARGE         1.0e8
#define JC2_AERFC             0.29214664
#define JC2_TWOTHIRDS         0.666666666666667
#define JC2_RELERR            1.0e-3

/* ---------------------------------------------------------------------
 * value / derivative pair
 * --------------------------------------------------------------------- */
typedef struct { double v, d; } jd;

static inline jd jc(double v)               { jd r; r.v = v; r.d = 0.0; return r; }
static inline jd jadd(jd a, jd b)           { jd r; r.v = a.v + b.v; r.d = a.d + b.d; return r; }
static inline jd jsub(jd a, jd b)           { jd r; r.v = a.v - b.v; r.d = a.d - b.d; return r; }
static inline jd jaddc(jd a, double c)      { jd r; r.v = a.v + c; r.d = a.d; return r; }
static inline jd jcsub(double c, jd a)      { jd r; r.v = c - a.v; r.d = -a.d; return r; }
static inline jd jmulc(jd a, double c)      { jd r; r.v = a.v * c; r.d = a.d * c; return r; }
static inline jd jmul(jd a, jd b)           { jd r; r.v = a.v * b.v; r.d = a.d * b.v + a.v * b.d; return r; }
static inline jd jdiv(jd a, jd b)           { jd r; r.v = a.v / b.v; r.d = (a.d - r.v * b.d) / b.v; return r; }
static inline jd jcdiv(double c, jd b)      { jd r; r.v = c / b.v; r.d = -r.v * b.d / b.v; return r; }
static inline jd jexp(jd a)                 { jd r; r.v = exp(a.v); r.d = r.v * a.d; return r; }
static inline jd jlog(jd a)                 { jd r; r.v = log(a.v); r.d = a.d / a.v; return r; }
static inline jd jsqrt(jd a)                { jd r; r.v = sqrt(a.v); r.d = 0.5 * a.d / r.v; return r; }
static inline jd jpowc(jd a, double p)      { jd r; r.v = pow(a.v, p); r.d = p * pow(a.v, p - 1.0) * a.d; return r; }

/* `mypower: sqrt for power 0.5 */
static inline jd jmypower(jd x, double p)   { return (p == 0.5) ? jsqrt(x) : jpowc(x, p); }

/* `mypower2: 1/x for power -1 */
static inline jd jmypower2(jd x, double p)  { return (p == -1.0) ? jcdiv(1.0, x) : jpowc(x, p); }

/* `mypower3: x^4 for power 4, else |x|^power */
static inline jd jmypower3(jd x, double p)
{
    jd r;
    if (p == 4.0) {
        r = jmul(jmul(x, x), jmul(x, x));
    } else {
        double ax = fabs(x.v);
        r.v = pow(ax, p);
        r.d = (ax > 0.0) ? p * pow(ax, p - 1.0) * ((x.v > 0.0) ? 1.0 : -1.0) * x.d : 0.0;
    }
    return r;
}

/* `P3: third order polynomial expansion of exp() */
static inline jd jP3(jd u)
{
    /* 1 + u*(1 + 0.5*(u*(1 + u/3))) */
    jd t = jaddc(jmulc(u, JC2_ONETHIRD), 1.0);
    t = jaddc(jmulc(jmul(u, t), 0.5), 1.0);
    return jaddc(jmul(u, t), 1.0);
}

/* `expl */
static inline jd jexpl(jd x)
{
    if (fabs(x.v) < JC2_SE05)
        return jexp(x);
    if (x.v < 0.0)
        return jcdiv(JC2_KE05, jP3(jcsub(-JC2_SE05, x)));
    return jmulc(jP3(jaddc(x, -JC2_SE05)), JC2_KE05INV);
}

/* `expl_low */
static inline jd jexpl_low(jd x)
{
    if (x.v > -JC2_SE05)
        return jexp(x);
    return jcdiv(JC2_KE05, jP3(jcsub(-JC2_SE05, x)));
}

/* `expll(x, xlow, expxlow, xhigh, expxhigh) with P1(x) = x + 1 */
static inline jd jexpll(jd x, double xlow, double expxlow, double xhigh, double expxhigh)
{
    if (x.v < xlow)
        return jcdiv(expxlow, jcsub(xlow + 1.0, x));
    if (x.v > xhigh)
        return jmulc(jaddc(x, 1.0 - xhigh), expxhigh);
    return jexp(x);
}

/* `hypfunction2 (x0 may depend on the voltage) */
static inline jd jhyp2(jd x, jd x0, double eps)
{
    jd dx = jsub(x, x0);
    jd s = jsqrt(jaddc(jmul(dx, dx), 4.0 * eps * eps));
    return jmulc(jsub(jadd(x, x0), s), 0.5);
}

/* `hypfunction5 */
static inline jd jhyp5(jd x, double x0, double eps)
{
    double h1 = 4.0 * eps * eps;
    double h2 = eps / x0;
    jd h2d = jaddc(x, eps * h2);
    jd h3 = jaddc(h2d, x0);
    jd h4 = jcsub(x0, h2d);
    jd h5 = jsqrt(jaddc(jmul(h4, h4), h1));
    return jmulc(jdiv(jmulc(x, x0), jadd(h3, h5)), 2.0);
}

/* `calcerfcexpmtat */
static jd
jc2_erfcexpmtat(const JC2glob *g, jd y, jd m)
{
    jd ysq = jmul(y, y);
    jd terfc, tmp, erfcpos, poly;

    if (y.v > 0.0)
        terfc = jcdiv(1.0, jaddc(jmulc(y, g->perfc), 1.0));
    else
        terfc = jcdiv(1.0, jcsub(1.0, jmulc(y, g->perfc)));
    tmp = jexpl_low(jsub(m, ysq));
    poly = jadd(jadd(jmulc(terfc, JC2_AERFC), jmulc(jmul(terfc, terfc), g->berfc)),
                jmulc(jmul(jmul(terfc, terfc), terfc), g->cerfc));
    erfcpos = jmul(poly, tmp);
    if (y.v > 0.0)
        return erfcpos;
    tmp = jexpl_low(m);
    return jsub(jmulc(tmp, 2.0), erfcpos);
}

/* voltage dependent quantities shared by the three components */
typedef struct {
    jd vj, idmult, zinv, two_psistar, vjsrh, vbbt, vav;
} jc2Bias;

/* `juncapfunction for component k */
static void
jc2_function(const JC2glob *g, const JC2tvals *t, const JC2inst *in, int k,
             const jc2Bias *b, jd VAK, jd *Ij, jd *Qj)
{
    double P = t->P[k];
    jd tmp, id, isrh, itat, ibbt, fbreakdown;
    jd vbi_minus_vjsrh = jc(0.0), wsrh = jc(0.0), wdep = jc(0.0), asrh = jc(0.0);

    tmp = jmypower(jcsub(1.0, jmulc(b->vj, t->vbiinv[k])), t->one_minus_P[k]);
    *Qj = jmulc(jadd(jmulc(jcsub(1.0, tmp), t->qpref[k]),
                     jmulc(jsub(VAK, b->vj), t->qpref2[k])), in->cfactor);

    id = jmulc(b->idmult, t->idsat[k]);

    if ((t->CSRH[k] == 0.0) && (t->CTAT[k] == 0.0)) {
        isrh = jc(0.0);
    } else {
        jd wsrhstep, dwsrh;
        vbi_minus_vjsrh = jcsub(t->vbi[k], b->vjsrh);
        wsrhstep = jcsub(1.0, jsqrt(jcsub(1.0, jdiv(b->two_psistar, vbi_minus_vjsrh))));
        if (P == 0.5) {
            dwsrh = jc(0.0);
        } else {
            jd w2 = jmul(wsrhstep, wsrhstep);
            dwsrh = jmulc(jadd(jdiv(jmul(w2, jlog(wsrhstep)), jcsub(1.0, wsrhstep)), wsrhstep),
                          1.0 - 2.0 * P);
        }
        wsrh = jadd(wsrhstep, dwsrh);
        tmp = jmypower(jmulc(vbi_minus_vjsrh, t->VBIRinv[k]), P);
        wdep = jmulc(tmp, t->wdepnulr[k]);
        asrh = jmulc(jmul(jaddc(b->zinv, -1.0), wdep), t->ftd[k]);
        isrh = jmulc(jmul(asrh, wsrh), t->CSRH[k]);
    }

    if (t->CTAT[k] == 0.0) {
        itat = jc(0.0);
    } else {
        jd btat, t3b, ub, umax, sqrtumax, u15, wgamma, wtat, ktat, ltat, mtat, xerfc, emt, gammamax;
        double atat = t->atat[k];
        btat = jmulc(jdiv(jmulc(wdep, t->one_minus_P[k]), vbi_minus_vjsrh), t->btatpart[k]);
        t3b = jcdiv(JC2_TWOTHIRDS * atat, btat);
        ub = jmul(t3b, t3b);
        umax = jsqrt(jdiv(jmul(ub, ub), jaddc(jmul(ub, ub), 1.0)));
        sqrtumax = jsqrt(umax);
        u15 = jmul(umax, sqrtumax);
        wgamma = jmypower2(jaddc(jmul(btat, u15), 1.0), -P * t->one_over_one_minus_P[k]);
        wtat = jdiv(jmul(wsrh, wgamma), jadd(wsrh, wgamma));
        ktat = jsqrt(jmulc(jdiv(btat, sqrtumax), 0.375));
        ltat = jsub(jmulc(jmul(t3b, sqrtumax), 2.0), umax);
        mtat = jadd(jsub(jmulc(jmul(t3b, sqrtumax), atat), jmulc(umax, atat)),
                    jmulc(jmul(btat, u15), 0.5));
        xerfc = jmul(jaddc(ltat, -1.0), ktat);
        emt = jc2_erfcexpmtat(g, xerfc, mtat);
        gammamax = jmulc(jdiv(jmulc(emt, atat), ktat), JC2_SQRTPI * 0.5);
        itat = jmulc(jmul(jmul(asrh, gammamax), wtat), t->CTAT[k]);
    }

    if (t->CBBT[k] == 0.0) {
        ibbt = jc(0.0);
    } else {
        jd vdiff = jcsub(t->VBIR[k], b->vbbt);
        jd Fmaxr;
        tmp = jmypower(jmulc(vdiff, t->VBIRinv[k]), P);
        Fmaxr = jmulc(jdiv(jmulc(vdiff, t->wdepnulrinv[k]), tmp), t->one_over_one_minus_P[k]);
        tmp = jexpl(jcdiv(-t->fbbt[k], Fmaxr));
        ibbt = jmulc(jmul(jmul(jmul(VAK, Fmaxr), Fmaxr), tmp), t->CBBT[k]);
    }

    if (t->VBR[k] > JC2_VBRMAX) {
        fbreakdown = jc(1.0);
    } else if (b->vav.v > -g->alphaav * t->VBR[k]) {
        tmp = jmypower3(jmulc(b->vav, t->VBRinv[k]), t->PBR[k]);
        fbreakdown = jcdiv(1.0, jcsub(1.0, tmp));
    } else {
        fbreakdown = jaddc(jmulc(jaddc(b->vav, g->alphaav * t->VBR[k]), t->slope[k]), t->fstop[k]);
    }

    *Ij = jmulc(jmul(jadd(jadd(id, isrh), jadd(itat, ibbt)), fbreakdown), in->ifactor);
}

/* `juncapcommon: returns the total current and charge of one junction */
static void
jc2_common(const JC2glob *g, const JC2tvals *t, const JC2inst *in,
           jd V, jd *Ijun, jd *Qjun)
{
    jc2Bias b;
    double geo[3];
    jd Ij, Qj;
    int k;

    geo[0] = in->AB; geo[1] = in->LS; geo[2] = in->LG;

    b.vj = b.idmult = b.zinv = b.two_psistar = b.vjsrh = b.vbbt = b.vav = jc(0.0);
    *Ijun = *Qjun = jc(0.0);

    if (!((in->AB == 0.0) && (in->LS == 0.0) && (in->LG == 0.0))) {
        jd z, vjlim;
        b.vj = jhyp5(V, in->vfmin, in->vch);
        if (V.v < in->VMAX) {
            b.zinv = jexpl(jmulc(V, 0.5 * g->phitdinv));
            b.idmult = jmul(b.zinv, b.zinv);
        } else {
            b.idmult = jmulc(jaddc(jmulc(jaddc(V, -in->VMAX), g->phitdinv), 1.0),
                             in->exp_VMAX_over_phitd);
            b.zinv = jsqrt(b.idmult);
        }
        b.idmult = jaddc(b.idmult, -1.0);
        z = jcdiv(1.0, b.zinv);
        if (V.v > 0.0) {
            jd s = jsqrt(jmul(jaddc(z, 1.0), jaddc(z, 3.0)));
            b.two_psistar = jmulc(jlog(jadd(jaddc(z, 2.0), s)), 2.0 * g->phitd);
        } else {
            jd s = jsqrt(jmul(jaddc(b.zinv, 1.0), jaddc(jmulc(b.zinv, 3.0), 1.0)));
            b.two_psistar = jsub(jmulc(jlog(jadd(jaddc(jmulc(b.zinv, 2.0), 1.0), s)), 2.0 * g->phitd), V);
        }
        vjlim = jcsub(in->vbimin, b.two_psistar);
        b.vjsrh = jhyp2(V, vjlim, g->phitd);
        b.vbbt = jhyp2(V, jc(in->vbbtlim), g->phitr);
        b.vav = jhyp2(V, jc(0.0), JC2_EPSAV);
    }

    for (k = 0; k < 3; k++) {
        if (geo[k] == 0.0)
            continue;
        jc2_function(g, t, in, k, &b, V, &Ij, &Qj);
        *Ijun = jadd(*Ijun, jmulc(Ij, geo[k]));
        *Qjun = jadd(*Qjun, jmulc(Qj, geo[k]));
    }
}

/* `JuncapExpressCurrent and `JuncapExpressCharge */
static void
jc2_express(const JC2glob *g, const JC2tvals *t, const JC2inst *in,
            jd V, jd *Ijun, jd *Qjun)
{
    jd tm1, ijunfor1, ijunfor2, ijunrev, vjv;
    double geo[3];
    int k;

    tm1 = jexpll(jmulc(V, g->phitdinv * in->MFOR1), -JC2_SE05, JC2_KE05, in->xhighf1, in->expxhf1);
    ijunfor1 = jmulc(jaddc(tm1, -1.0), in->ISATFOR1);
    tm1 = jexpll(jmulc(V, g->phitdinv * in->MFOR2), -JC2_SE05, JC2_KE05, in->xhighf2, in->expxhf2);
    ijunfor2 = jmulc(jaddc(tm1, -1.0), in->ISATFOR2);
    if (in->m0flag > 0.0) {
        ijunrev = jmul(V, jaddc(jmulc(V, in->MREV), in->ISATREV));
    } else {
        tm1 = jexpll(jmulc(V, -g->phitdinv * in->MREV), -JC2_SE05, JC2_KE05, in->xhighr, in->expxhr);
        ijunrev = jmulc(jaddc(tm1, -1.0), -in->ISATREV);
    }
    *Ijun = jadd(jadd(ijunfor1, ijunfor2), ijunrev);

    geo[0] = in->AB; geo[1] = in->LS; geo[2] = in->LG;
    *Qjun = jc(0.0);
    vjv = jhyp5(V, in->vfmin, in->vch);
    for (k = 0; k < 3; k++) {
        if (in->zflag[k] > 0.5) {
            jd tmpv = jmypower(jcsub(1.0, jmulc(vjv, t->vbiinv[k])), t->one_minus_P[k]);
            jd q = jadd(jmulc(jcsub(1.0, tmpv), t->qpref[k]), jmulc(jsub(V, vjv), t->qpref2[k]));
            *Qjun = jadd(*Qjun, jmulc(q, geo[k]));
        }
    }
}

/* total current and charge of junction 'side' (0 source, 1 drain) at voltage vak */
void
JC2eval(const JC2tdata *td, int side, const JC2inst *in, double vak,
        double *ij, double *gj, double *qj, double *cj)
{
    jd V, I, Q;

    V.v = vak;
    V.d = 1.0;
    if (td->g.express)
        jc2_express(&td->g, &td->t[side], in, V, &I, &Q);
    else
        jc2_common(&td->g, &td->t[side], in, V, &I, &Q);
    *ij = I.v;
    *gj = I.d;
    *qj = Q.v;
    *cj = Q.d;
}

/* ---------------------------------------------------------------------
 * parameters
 * --------------------------------------------------------------------- */

static const double jc2_default[JC2_NPAR] = {
#define X(a, name, def, lo, hi, desc) def,
    JC2_SIDE_PARAMS(X, 0)
#undef X
};
static const double jc2_lower[JC2_NPAR] = {
#define X(a, name, def, lo, hi, desc) lo,
    JC2_SIDE_PARAMS(X, 0)
#undef X
};
static const double jc2_upper[JC2_NPAR] = {
#define X(a, name, def, lo, hi, desc) hi,
    JC2_SIDE_PARAMS(X, 0)
#undef X
};
static const char *const jc2_name[JC2_NPAR] = {
#define X(a, name, def, lo, hi, desc) #name,
    JC2_SIDE_PARAMS(X, 0)
#undef X
};

/* reference temperature [K]: TRJ if given, else the default trdef [K] of
   the host model */
static double
jc2_tref(const JC2model *jm, double trdef)
{
    if (jm->trjGiven)
        return JC2_KELVINCONVERSION + jm->trj;
    return trdef;
}

/* id relative to the ID base of the host model */
int
JC2mParam(JC2model *jm, int id, IFvalue *value)
{
    int k;

    switch (id) {
    case JC2_MOD_VERSION:
        jm->version = value->rValue;
        jm->versionGiven = TRUE;
        return OK;
    case JC2_MOD_IMAX:
        jm->imax = value->rValue;
        jm->imaxGiven = TRUE;
        return OK;
    case JC2_MOD_TRJ:
        jm->trj = value->rValue;
        jm->trjGiven = TRUE;
        return OK;
    case JC2_MOD_FREV:
        jm->frev = value->rValue;
        jm->frevGiven = TRUE;
        return OK;
    case JC2_MOD_SWJUNEXP:
        jm->swjunexp = value->rValue;
        jm->swjunexpGiven = TRUE;
        return OK;
    case JC2_MOD_SWJUNASYM:
        jm->swjunasym = value->rValue;
        jm->swjunasymGiven = TRUE;
        return OK;
    default:
        break;
    }

    k = id - JC2_MOD_SIDE;
    if (k >= 0 && k < 2 * JC2_NPAR) {
        jm->par[k / JC2_NPAR][k % JC2_NPAR] = value->rValue;
        jm->parGiven[k / JC2_NPAR][k % JC2_NPAR] = 1;
        return OK;
    }
    return E_BADPARM;
}

/* id relative to the ID base of the host model;
   trdef: default reference temperature of the host model [K] */
int
JC2mAsk(const JC2model *jm, double trdef, int id, IFvalue *value)
{
    int k;

    switch (id) {
    case JC2_MOD_VERSION:
        value->rValue = jm->version;
        return OK;
    case JC2_MOD_IMAX:
        value->rValue = jm->imax;
        return OK;
    case JC2_MOD_TRJ:
        value->rValue = jc2_tref(jm, trdef) - JC2_KELVINCONVERSION;
        return OK;
    case JC2_MOD_FREV:
        value->rValue = jm->frev;
        return OK;
    case JC2_MOD_SWJUNEXP:
        value->rValue = jm->swjunexp;
        return OK;
    case JC2_MOD_SWJUNASYM:
        value->rValue = jm->swjunasym;
        return OK;
    default:
        break;
    }

    k = id - JC2_MOD_SIDE;
    if (k >= 0 && k < 2 * JC2_NPAR) {
        int side = k / JC2_NPAR, idx = k % JC2_NPAR;
        if (side == 1 && !(jm->swjunasym != 0.0 && jm->parGiven[1][idx]))
            side = 0;   /* drain uses the source parameter */
        value->rValue = jm->par[side][idx];
        return OK;
    }
    return E_BADPARM;
}

/* defaults, called from the setup function of the host model;
   name: model name for warnings, active: JUNCAP2 is used by the model */
void
JC2setup(JC2model *jm, char *name, int active)
{
    int idx;

    jm->name = name;

    if (!jm->versionGiven)
        jm->version = 200.6;
    if (active && !jm->warned && !(jm->version > 200.59999 && jm->version < 200.7))
        printf("Warning: jcap2version = %g is not implemented, JUNCAP2 200.6 is used.\n",
               jm->version);

    if (!jm->imaxGiven)
        jm->imax = 1000.0;
    if (!jm->frevGiven)
        jm->frev = 1.0e3;
    if (!jm->swjunexpGiven)
        jm->swjunexp = 0.0;
    if (!jm->swjunasymGiven)
        jm->swjunasym = 0.0;

    for (idx = 0; idx < JC2_NPAR; idx++)
        if (!jm->parGiven[0][idx])
            jm->par[0][idx] = jc2_default[idx];
}

/* range check as in the Verilog-A parameter declarations; clips, warns only
   at the first temperature update of the model (warned) */
static double
jc2_clip(const JC2model *jm, const char *name, const char *suffix, double val, double lo, double hi)
{
    if (val < lo || val > hi) {
        double nval = (val < lo) ? lo : hi;
        if (!jm->warned)
            printf("Warning: JUNCAP2 parameter %s%s = %g of model %s out of range, set to %g.\n",
                   name, suffix, val, jm->name, nval);
        return nval;
    }
    return val;
}

/* ---------------------------------------------------------------------
 * temperature dependent values (JUNCAP200_InitModel.include)
 * --------------------------------------------------------------------- */

/* temperature dependent values td of model jm at temperature temp [K];
   trdef: default reference temperature of the host model [K] */
void
JC2temp(JC2model *jm, JC2tdata *td, double temp, double trdef)
{
    JC2glob *g = &td->g;
    double tkr, tkd, auxt, KBOL_over_QELE, phitrinv, deltaphigr, deltaphigd, frev, epssi;
    int side, k, idx, asym;

    epssi = JC2_EPSO * JC2_EPSRSI;

    g->imax    = jc2_clip(jm, "imax", "", jm->imax, 1.0e-12, JC2_NOLIMIT);
    frev       = jc2_clip(jm, "frev", "", jm->frev, 1.0e1, 1.0e10);
    g->express = (jm->swjunexp > 0.5);
    asym = (jc2_clip(jm, "swjunasym", "", jm->swjunasym, 0.0, 1.0) != 0.0);

    if (!asym && !jm->warned)
        for (idx = 0; idx < JC2_NPAR; idx++)
            if (jm->parGiven[1][idx]) {
                printf("Warning: JUNCAP2 drain parameters of model %s are ignored, "
                       "they need swjunasym = 1.\n", jm->name);
                break;
            }

    /* reference temperature */
    tkr = jc2_tref(jm, trdef);
    if (jm->trjGiven)
        tkr = JC2_KELVINCONVERSION
            + jc2_clip(jm, "trj", "", jm->trj, JC2_MINTEMP, JC2_NOLIMIT);
    tkd = MAX(temp, JC2_KELVINCONVERSION + JC2_MINTEMP);

    auxt = tkd / tkr;
    KBOL_over_QELE = JC2_KBOL / JC2_QELE;
    g->phitr = KBOL_over_QELE * tkr;
    phitrinv = 1.0 / g->phitr;
    g->phitd = KBOL_over_QELE * tkd;
    g->phitdinv = 1.0 / g->phitd;

    deltaphigr = -(7.02e-4 * tkr * tkr) / (1108.0 + tkr);
    deltaphigd = -(7.02e-4 * tkd * tkd) / (1108.0 + tkd);

    g->perfc = (JC2_SQRTPI * JC2_AERFC);
    g->berfc = ((-5.0 * (JC2_AERFC) + 6.0 - pow((g->perfc), -2.0)) / 3.0);
    g->cerfc = (1.0 - (JC2_AERFC) - (g->berfc));

    g->alphaav = 1.0 - 1.0 / frev;

    for (side = 0; side < 2; side++) {
        JC2tvals *t = &td->t[side];
        const char *sfx = side ? "d" : "";
        double p[JC2_NPAR];

        for (idx = 0; idx < JC2_NPAR; idx++) {
            if (side == 1 && !(asym && jm->parGiven[1][idx]))
                p[idx] = td->t[0].par[idx];    /* drain takes the source value */
            else
                p[idx] = jc2_clip(jm, jc2_name[idx], sfx, jm->par[side][idx],
                                  jc2_lower[idx], jc2_upper[idx]);
            t->par[idx] = p[idx];
        }

        for (k = 0; k < 3; k++) {
            double CJOR, VBIR, P, PHIG, IDSATR, MEFFTAT, FBBTR, STFBBT, PBR;
            double phigr, phigd, ubi, deltaE;

            CJOR    = p[JC2P_cjorbot    + k];
            VBIR    = p[JC2P_vbirbot    + k];
            P       = p[JC2P_pbot       + k];
            PHIG    = p[JC2P_phigbot    + k];
            IDSATR  = p[JC2P_idsatrbot  + k];
            MEFFTAT = p[JC2P_mefftatbot + k];
            FBBTR   = p[JC2P_fbbtrbot   + k];
            STFBBT  = p[JC2P_stfbbtbot  + k];
            PBR     = p[JC2P_pbrbot     + k];

            t->P[k]    = P;
            t->VBIR[k] = VBIR;
            t->CSRH[k] = p[JC2P_csrhbot + k];
            t->CTAT[k] = p[JC2P_ctatbot + k];
            t->CBBT[k] = p[JC2P_cbbtbot + k];
            t->VBR[k]  = p[JC2P_vbrbot  + k];
            t->PBR[k]  = PBR;

            /* bandgap voltages at reference and device temperature */
            phigr = PHIG + deltaphigr;
            phigd = PHIG + deltaphigd;

            /* factor ftd and saturation current for ideal-current model */
            t->ftd[k] = (pow(auxt, 1.5)) * exp(0.5 * ((phigr * phitrinv) - (phigd * g->phitdinv)));
            t->idsat[k] = IDSATR * t->ftd[k] * t->ftd[k];

            /* built-in voltage, limited to phitd */
            ubi = VBIR * auxt - 2.0 * g->phitd * log(t->ftd[k]);
            t->vbi[k] = ubi + g->phitd * log(1 + exp((JC2_VBILOW - ubi) * g->phitdinv));
            t->vbiinv[k] = 1.0 / t->vbi[k];

            t->one_minus_P[k] = 1.0 - P;
            t->one_over_one_minus_P[k] = 1.0 / t->one_minus_P[k];

            /* temperature-scaled zero-bias capacitance and charge prefactors */
            t->cjo[k] = CJOR * pow((VBIR * t->vbiinv[k]), P);
            t->qpref[k] = t->cjo[k] * t->vbi[k] * t->one_over_one_minus_P[k];
            t->qpref2[k] = JC2_A * t->cjo[k];

            /* zero-bias depletion width at reference temperature */
            if (k == JC2_BOT)
                t->wdepnulr[k] = epssi / CJOR;
            else if (k == JC2_STI)
                t->wdepnulr[k] = p[JC2P_xjunsti] * epssi / CJOR;
            else
                t->wdepnulr[k] = p[JC2P_xjungat] * epssi / CJOR;
            t->wdepnulrinv[k] = 1.0 / t->wdepnulr[k];

            t->VBIRinv[k] = 1.0 / VBIR;

            /* trap-assisted tunneling */
            deltaE = MAX(0.5 * phigd, g->phitd);
            t->atat[k] = deltaE * g->phitdinv;
            t->btatpart[k] = sqrt(32.0 * MEFFTAT * JC2_MELE * JC2_QELE * (deltaE * deltaE * deltaE))
                           / (3.0 * JC2_HBAR);

            /* band-to-band tunneling */
            t->fbbt[k] = FBBTR * (1.0 + STFBBT * (tkd - tkr));
            if (t->fbbt[k] < 0.0)
                t->fbbt[k] = 0.0;

            /* avalanche / breakdown */
            t->fstop[k] = 1.0 / (1.0 - pow(g->alphaav, PBR));
            t->VBRinv[k] = 1.0 / t->VBR[k];
            t->slope[k] = -(t->fstop[k] * t->fstop[k] * pow(g->alphaav, (PBR - 1.0))) * PBR * t->VBRinv[k];
        }
        t->vjunref = p[JC2P_vjunref];
        t->fjunq   = p[JC2P_fjunq];
    }
    jm->warned = 1;
}


/* ---------------------------------------------------------------------
 * geometry dependent values (`JuncapInitInstance, `JuncapExpressInit1..5)
 * --------------------------------------------------------------------- */

static double
jc2_itotal(const JC2glob *g, const JC2tvals *t, const JC2inst *in, double V)
{
    jd I, Q;
    jc2_common(g, t, in, jc(V), &I, &Q);
    return I.v;
}

static void
jc2_init_instance(const JC2glob *g, const JC2tvals *t, JC2inst *in,
                  double AB, double LS, double LG, double ifactor, double cfactor)
{
    double vmax[3], geo[3], vbi2[3], p2[3], vbi2r[3], pmax;
    int k;

    in->ifactor = ifactor;
    in->cfactor = cfactor;
    in->AB = MAX(AB, 0.0);
    in->LS = MAX(LS, 0.0);
    in->LG = MAX(LG, 0.0);
    geo[0] = in->AB; geo[1] = in->LS; geo[2] = in->LG;

    for (k = 0; k < 3; k++) {
        if (t->idsat[k] * geo[k] > 0.0)
            vmax[k] = g->phitd * log(g->imax / (t->idsat[k] * geo[k]) + 1.0);
        else
            vmax[k] = JC2_VMAXLARGE;
    }
    in->VMAX = MIN(MIN(vmax[0], vmax[1]), vmax[2]);
    in->exp_VMAX_over_phitd = jexpl(jc(in->VMAX * g->phitdinv)).v;

    /* minimum of the relevant built-in voltages and limit for the BBT calculation */
    for (k = 0; k < 3; k++) {
        vbi2[k]  = t->vbi[k];
        p2[k]    = t->P[k];
        vbi2r[k] = t->VBIR[k];
    }
    if (in->AB == 0.0) {
        vbi2[0]  = t->vbi[1] + t->vbi[2];
        p2[0]    = 0.9 * MIN(t->P[1], t->P[2]);
        vbi2r[0] = t->VBIR[1] + t->VBIR[2];
    }
    if (in->LS == 0.0) {
        vbi2[1]  = t->vbi[0] + t->vbi[2];
        p2[1]    = 0.9 * MIN(t->P[0], t->P[2]);
        vbi2r[1] = t->VBIR[0] + t->VBIR[2];
    }
    if (in->LG == 0.0) {
        vbi2[2]  = t->vbi[0] + t->vbi[1];
        p2[2]    = 0.9 * MIN(t->P[0], t->P[1]);
        vbi2r[2] = t->VBIR[0] + t->VBIR[1];
    }
    in->vbimin = MIN(MIN(vbi2[0], vbi2[1]), vbi2[2]);
    in->vch = in->vbimin * JC2_EPSCH;
    pmax = MAX(MAX(p2[0], p2[1]), p2[2]);
    in->vfmin = in->vbimin * (1.0 - (pow(JC2_A, (-1.0 / (pmax)))));
    in->vbbtlim = MIN(MIN(vbi2r[0], vbi2r[1]), vbi2r[2]) - JC2_DVBI;

    /* JUNCAP-express defaults */
    in->ISATFOR1 = 0.0;
    in->MFOR1    = 1.0;
    in->MFOR2    = 1.0;
    in->ISATFOR2 = 0.0;
    in->MREV     = 1.0;
    in->ISATREV  = 0.0;
    in->m0flag   = 0.0;
    in->xhighf1  = 0.0;
    in->expxhf1  = 0.0;
    in->xhighf2  = 0.0;
    in->expxhf2  = 0.0;
    in->xhighr   = 0.0;
    in->expxhr   = 0.0;
    in->zflag[0] = in->zflag[1] = in->zflag[2] = 1.0;

    if (g->express) {
        const double FRACNA = 0.4, FRACNB = 0.65, FRACI = 0.8;
        double V1, V2, V3, V4, V5, I1, I2, I3, I4, I5;
        double I1_cor, I2_cor, I3_cor, I4_cor, I5_cor, alphaje, zfrac;
        int allzero = (in->AB == 0.0) && (in->LS == 0.0) && (in->LG == 0.0);

        /* part 1 and 2: evaluate the full model at five voltages */
        V1 = -FRACNA * t->vjunref;
        V2 = -FRACNB * t->vjunref;
        V3 = -FRACI * t->vjunref;
        V4 =  0.1;
        V5 =  0.2;
        I1 = jc2_itotal(g, t, in, V1);
        I2 = jc2_itotal(g, t, in, V2);
        I3 = jc2_itotal(g, t, in, V3);
        I4 = jc2_itotal(g, t, in, V4);
        I5 = jc2_itotal(g, t, in, V5);

        /* part 3: internal parameters from these five (I,V)-values */
        in->ISATFOR1 = in->AB * t->idsat[0] + in->LS * t->idsat[1] + in->LG * t->idsat[2];
        I4_cor = I4 - in->ISATFOR1 * (exp(V4 * g->phitdinv * in->MFOR1) - 1.0);
        I5_cor = I5 - in->ISATFOR1 * (exp(V5 * g->phitdinv * in->MFOR1) - 1.0);
        if (!allzero) {
            if ((I4 > 0.0) && (I5 > 0.0)) {
                if ((((I4_cor / I4) > JC2_RELERR) || ((I5_cor / I5) > JC2_RELERR))
                    && (I4_cor > 0.0) && (I5_cor > 0.0) && (I5_cor > I4_cor)) {
                    alphaje = I4_cor / I5_cor;
                    in->MFOR2 = g->phitd * log(alphaje) / (V4 - V5);
                    in->ISATFOR2 = I4_cor / (exp(V4 * g->phitdinv * in->MFOR2) - 1.0);
                }
            }
            I1_cor = I1 - in->ISATFOR1 * (exp(V1 * g->phitdinv * in->MFOR1) - 1.0)
                        - in->ISATFOR2 * (exp(V1 * g->phitdinv * in->MFOR2) - 1.0);
            I2_cor = I2 - in->ISATFOR1 * (exp(V2 * g->phitdinv * in->MFOR1) - 1.0)
                        - in->ISATFOR2 * (exp(V2 * g->phitdinv * in->MFOR2) - 1.0);
            I3_cor = I3 - in->ISATFOR1 * (exp(V3 * g->phitdinv * in->MFOR1) - 1.0)
                        - in->ISATFOR2 * (exp(V3 * g->phitdinv * in->MFOR2) - 1.0);
            if ((I1 < 0.0) && (I2 < 0.0) && (I3 < 0.0)) {
                if ((((I1_cor / I1) > JC2_RELERR) || ((I2_cor / I2) > JC2_RELERR)
                     || ((I3_cor / I3) > JC2_RELERR))
                    && (I1_cor < 0.0) && (I2_cor < 0.0) && (I3_cor < 0.0)) {
                    double m0_rev, mcor_rev, tt0, tt1, tt2;
                    alphaje = I1_cor / I2_cor;
                    m0_rev = -g->phitd * log(alphaje) / (V1 - V2);   /* zeroth order approximation */
                    tt0 = V2 / (V2 - V1);
                    tt1 = g->phitd * (alphaje - 1.0) * (pow(alphaje, tt0) - 1.0);
                    tt0 = V1 / (V1 - V2);
                    tt2 = pow(alphaje, tt0) * (V2 - V1) + alphaje * V1 - V2;
                    mcor_rev = tt1 / tt2;                            /* first order Newton correction */
                    in->MREV = m0_rev + mcor_rev;
                    if (fabs(V3 * g->phitdinv * in->MREV) < 1.0e-6) {
                        /* Taylor approximation; ISATREV and MREV have a different meaning here */
                        in->m0flag = 1.0;
                        in->ISATREV = I3_cor * (1.0 / V3 + 0.5 * g->phitdinv * in->MREV);
                        in->MREV = -0.5 * I3_cor * in->MREV * g->phitdinv / V3;
                    } else {
                        in->m0flag = 0.0;
                        in->ISATREV = -I3_cor / (exp(-V3 * g->phitdinv * in->MREV) - 1.0);
                    }
                }
            }
        }

        /* part 4: charge model initialization */
        zfrac = t->fjunq * (in->AB * t->cjo[0] + in->LS * t->cjo[1] + in->LG * t->cjo[2]);
        for (k = 0; k < 3; k++)
            if ((geo[k] * t->cjo[k]) <= zfrac)
                in->zflag[k] = 0.0;

        /* part 5: limits beyond which exponentials are linearly extrapolated */
        if (!allzero) {
            in->xhighf1 = log(0.5 * g->imax / (in->ISATFOR1 + 1.0e-21));
            in->xhighf2 = log(0.5 * g->imax / (in->ISATFOR2 + 1.0e-21));
            in->xhighr  = log(0.5 * g->imax / (fabs(in->ISATREV) + 1.0e-21));
        }
        in->xhighf1 = MIN(in->xhighf1, JC2_SE05);
        in->expxhf1 = exp(in->xhighf1);
        in->xhighf2 = MIN(in->xhighf2, JC2_SE05);
        in->expxhf2 = exp(in->xhighf2);
        in->xhighr  = MIN(in->xhighr, JC2_SE05);
        in->expxhr  = exp(in->xhighr);
    }
}

/* geometry of junction 'side' (0 source, 1 drain) of an instance:
   AB area, LS STI-edge perimeter, LG gate-edge length (geometry of the host model) */
void
JC2initInst(const JC2tdata *td, int side, JC2inst *in,
            double AB, double LS, double LG, double ifactor, double cfactor)
{
    jc2_init_instance(&td->g, &td->t[side], in, AB, LS, LG, ifactor, cfactor);
}
