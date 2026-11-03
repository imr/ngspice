/* New routines to evaluate the .measure cards.
   Entry point is function get_measure2(), called by fcn do_measure()
   from measure.c.
   Patches by Bill Swartz from 2009-05-18 and 2009-08-21 are included.
*/

#include "ngspice/ngspice.h"
#include "ngspice/memory.h"

#include "ngspice/fteext.h"
#include "ngspice/wordlist.h"

#include "vectors.h"
#include <math.h>
#include "dotcards.h"
#include "com_measure2.h"
#include "breakp2.h"

int measure_precision = -1;

typedef enum {
    MEASUREMENT_OK = 0,
    MEASUREMENT_FAILURE = 1
} MEASURE_VAL_T;

#define MEASURE_DEFAULT (-1)
/* Enhancement-225: measure error buffer. Was errbuf[100]; a long measure
 * expression (its whole string is sprintf'd as the vector name in
 * "no such vector as '%s'") overran it -> heap/stack smash. Bounded via
 * snprintf(.., MEAS_ERRBUF_SIZE, ..) at every write below. */
#define MEAS_ERRBUF_SIZE 512
#define MEASURE_LAST_TRANSITION (-2)

typedef struct measure
{
    char *result;

    char *m_vec;          // name of the output variable which determines the beginning of the measurement
    char *m_vec2;         // second output variable to measure if applicable
    char *m_analysis;     // analysis type (tran, dc or ac)
    char m_vectype;       // type of vector m_vec (vm, vi, vr, vp, vdb)
    char m_vectype2;      // type of vector m_vec2 (vm, vi, vr, vp, vdb)
    int m_rise;           // count number of rise events
    int m_fall;           // count number of fall events
    int m_cross;          // count number of rise/fall aka cross  events
    double m_val;         // value of the m_ver at which the counter for crossing, rises or falls is incremented by one
    double m_td;          // amount of delay before the measurement should start
    double m_from;        // measure only in a time window - starting time of window
    double m_to;          // measurement window - ending time
    double m_at;          // measure at the specified time
    double m_measured;    // what we measured
    double m_measured_at; // what we measured at the given time

} MEASURE, *MEASUREPTR;

typedef enum AnalysisType {
    AT_UNKNOWN, AT_DELAY, AT_TRIG,
    AT_FIND, AT_WHEN,
    AT_AVG, AT_MIN, AT_MAX, AT_RMS, AT_PP,
    AT_INTEG, AT_DERIV,
    AT_ERR, AT_ERR1, AT_ERR2, AT_ERR3, AT_MIN_AT, AT_MAX_AT,
    AT_PHASE_MARGIN, AT_GAIN_MARGIN
} ANALYSIS_TYPE_T;

static void measure_errMessage(const char *mName, const char *mFunction,
        const char *trigTarg, const char *errMsg, int chk_only);

/* like in string.c, here special without () */
static bool
is_arith_char2(char c)
{
    return c != '\0' && strchr("*/<>?:|&^!%\\", c);
}

static bool
str_has_arith_char2(char* s)
{
    if (*s == '+' || *s == '-')
        s++;
    for (; *s; s++)
        if (is_arith_char2(*s))
            return TRUE;

    return FALSE;
}

/** return precision (either 5 or value of environment variable NGSPICE_MEAS_PRECISION) */
int
measure_get_precision(void)
{
    char *env_ptr;
    int  precision = 5;

    if ((env_ptr = getenv("NGSPICE_MEAS_PRECISION")) != NULL)
        precision = atoi(env_ptr);

    if (measure_precision > 0)
        precision = measure_precision;

    return precision;
}


static void measure_errMessage(const char *mName, const char *mFunction,
        const char *trigTarg, const char *errMsg, int chk_only)
{
    if (!chk_only) {
        fprintf(stderr, "\nError: measure  %s  %s(%s) : ", mName, mFunction, trigTarg);
        fprintf(stderr, "%s", errMsg);
    }
}


/* If you have a vector vm(out), extract 'm' to meas->m_vectype
   and v(out) to meas->m_vec (without 'm') */

static void
correct_vec(MEASUREPTR meas)
{
    char *vec = meas->m_vec;

    /* return if not of type VM() etc */
    if ((*vec != 'v') || (!strchr(vec, '(')))
        return;

    if (vec[1] != '(') {
        meas->m_vectype = vec[1];
        meas->m_vec = tprintf("%c%s", vec[0], strchr(vec, '('));
        tfree(vec);
    }

    vec = meas->m_vec2;

    if (vec && (vec[1] != '(')) {
        meas->m_vectype2 = vec[1];
        meas->m_vec2 = tprintf("%c%s", vec[0], strchr(vec, '('));
        tfree(vec);
    }
}


/* Returns a value from a complex vector *values, depending on meas->m_vectype */
static double
get_value(
    MEASUREPTR meas,     /*in: pointer to mesurement structure */
    struct dvec *values, /*in: vector of complex values */
    int idx              /*in: index of vector value to be read out */
    )
{
    double ar, bi;

    ar = values->v_compdata[idx].cx_real;
    bi = values->v_compdata[idx].cx_imag;

    if ((meas->m_vectype == 'm') || (meas->m_vectype == 'M')) {
        return hypot(ar, bi); /* magnitude */
    } else if ((meas->m_vectype == 'r') || (meas->m_vectype == 'R')) {
        return ar;  /* real value */
    } else if ((meas->m_vectype == 'i') || (meas->m_vectype == 'I')) {
        return bi;  /* imaginary value */
    } else if ((meas->m_vectype == 'p') || (meas->m_vectype == 'P')) {
        return radtodeg(atan2(bi, ar)); /* phase (in degrees) */
    } else if ((meas->m_vectype == 'd') || (meas->m_vectype == 'D')) {
        return 20.0 * log10(hypot(ar, bi));  /* dB of magnitude */
    } else {
        return ar;  /* default: real value */
    }
}


/* Returns interpolated value. If ac simulation, exploit vector type with complex data for y */
static double
measure_interpolate(
    struct dvec *xScale, /* in: vector of independent variables, if ac: complex vector,
                            but only real part used */
    struct dvec *values, /* in: vector of dependent variables, if ac: complex vector */
    int i,               /* in: index of first interpolation value */
    int j,               /* in: index of second interpolation value */
    double var_value,    /* in: variable, whose counterpart is sought by interpolation */
    char x_or_y ,        /* in: if 'x', then look for y, if 'y' then look for x */
    MEASUREPTR meas      /* pointer to measurement structure */
    )
{
    double slope;
    double yint;
    double result;

    if (cieq (meas->m_analysis, "ac")) {
        /* get values from complex y vector according to meas->m_vectype,
           x vector uses only real part of complex data (frequency).*/
        slope = (get_value(meas, values, j)  - get_value(meas, values, i)) /
            (xScale->v_compdata[j].cx_real - xScale->v_compdata[i].cx_real);
        yint  = get_value(meas, values, i) - slope*xScale->v_compdata[i].cx_real;
    } else {
        slope = (values->v_realdata[j] - values->v_realdata[i]) /
            (xScale->v_realdata[j] - xScale->v_realdata[i]);
        yint  = values->v_realdata[i] - slope*xScale->v_realdata[i];
    }

    if (x_or_y == 'x')
        result = (var_value - yint)/slope;
    else
        result = slope*var_value + yint;

    return result;
}


/* -----------------------------------------------------------------
 * Function: Given an operation string returns back the measure type -
 * one of the enumerated type ANALSYS_TYPE_T.
 * ----------------------------------------------------------------- */
static ANALYSIS_TYPE_T
measure_function_type(char *operation)
{
    char *mFunction;               /* operation */
    ANALYSIS_TYPE_T mFunctionType; /* type of requested function */

    mFunction = cp_unquote(operation);
    // Functions
    if (strcasecmp(mFunction, "DELAY") == 0)
        mFunctionType = AT_DELAY;
    else if (strcasecmp(mFunction, "TRIG") == 0)
        mFunctionType = AT_DELAY;
    else if (strcasecmp(mFunction, "TARG") == 0)
        mFunctionType = AT_DELAY;
    else if (strcasecmp(mFunction, "FIND") == 0)
        mFunctionType = AT_FIND;
    else if (strcasecmp(mFunction, "WHEN") == 0)
        mFunctionType = AT_WHEN;
    else if (strcasecmp(mFunction, "AVG") == 0)
        mFunctionType = AT_AVG;
    else if (strcasecmp(mFunction, "MIN") == 0)
        mFunctionType = AT_MIN;
    else if (strcasecmp(mFunction, "MAX") == 0)
        mFunctionType = AT_MAX;
    else if (strcasecmp(mFunction, "MIN_AT") == 0)
        mFunctionType = AT_MIN_AT;
    else if (strcasecmp(mFunction, "MAX_AT") == 0)
        mFunctionType = AT_MAX_AT;
    else if (strcasecmp(mFunction, "RMS") == 0)
        mFunctionType = AT_RMS;
    else if (strcasecmp(mFunction, "PP") == 0)
        mFunctionType = AT_PP;
    else if (strcasecmp(mFunction, "INTEG") == 0 ||
             strcasecmp(mFunction, "INTEGRAL") == 0)
        mFunctionType = AT_INTEG;
    else if (strcasecmp(mFunction, "DERIV") == 0 ||
             strcasecmp(mFunction, "DERIVATIVE") == 0)
        mFunctionType = AT_DERIV;
    else if (strcasecmp(mFunction, "ERR") == 0)
        mFunctionType = AT_ERR;
    else if (strcasecmp(mFunction, "ERR1") == 0)
        mFunctionType = AT_ERR1;
    else if (strcasecmp(mFunction, "ERR2") == 0)
        mFunctionType = AT_ERR2;
    else if (strcasecmp(mFunction, "ERR3") == 0)
        mFunctionType = AT_ERR3;
    else if (strcasecmp(mFunction, "PHASE_MARGIN") == 0 ||
             strcasecmp(mFunction, "PHASEMARGIN") == 0)
        mFunctionType = AT_PHASE_MARGIN;
    else if (strcasecmp(mFunction, "GAIN_MARGIN") == 0 ||
             strcasecmp(mFunction, "GAINMARGIN") == 0)
        mFunctionType = AT_GAIN_MARGIN;
    else
        mFunctionType = AT_UNKNOWN;

    tfree(mFunction);

    return (mFunctionType);
}


/* For AC/SP, a measured item like vdb(out) / vp(out) / vm(out) is a post-processed
 * output function; only the base vector v(out) can be .save'd. Strip the magnitude/
 * phase suffix down to v(...)/i(...) so the auto-save pre-pass saves the right thing
 * (otherwise batch `.meas ac ... vdb(out)' left the node unsaved -> "no data saved
 * for AC, analysis not run"). Plain v(out) and non-parenthesised tokens pass through.
 * Caller frees. */
static char *
strip_ac_vec(const char *var, const char *analysis)
{
    if (cieq((char *) analysis, "ac") || cieq((char *) analysis, "sp")) {
        char c0 = var[0];
        const char *paren = strchr(var, '(');
        if ((c0 == 'v' || c0 == 'V' || c0 == 'i' || c0 == 'I') && paren && paren > var + 1)
            return tprintf("%c%s", c0, paren);
    }
    return copy(var);
}

/* -----------------------------------------------------------------
 * Function: Parse the measurement line and extract any variables in
 * the statement and call com_save2 to instantiate the variable as a
 * measurement vector in the transient analysis.
 * ----------------------------------------------------------------- */
int
measure_extract_variables(char *line)
{
    /* Various formats for measure statement:
     * .MEASURE {DC|AC|TRAN} result TRIG trig_variable VAL=val
     * + <TD=td> <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     * + <TRIG AT=time>
     * + TARG targ_variable VAL=val
     * + <TD=td> <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     * + <TRIG AT=time>
     *
     * .MEASURE {DC|AC|TRAN} result WHEN out_variable=val
     * + <TD=td> <FROM=val> <TO=val>
     * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     *
     * .MEASURE {DC|AC|TRAN} result WHEN out_variable=out_variable2
     * + <TD=td> <FROM=val> <TO=val>
     * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     *
     * .MEASURE {DC|AC|TRAN} result FIND out_variable WHEN out_variable2=val
     * + <TD=td> <FROM=val> <TO=val>
     * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     *
     * .MEASURE {DC|AC|TRAN} result FIND out_variable WHEN out_variable2=out_variable3
     * + <TD=td>
     * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     *
     * .MEASURE {DC|AC|TRAN} result FIND out_variable AT=val
     * + <FROM=val> <TO=val>
     *
     * .MEASURE {DC|AC|TRAN} result {AVG|MIN|MAX|MIN_AT|MAX_AT|PP|RMS} out_variable
     * + <TD=td> <FROM=val> <TO=val>
     *
     * .MEASURE {DC|AC|TRAN} result INTEG<RAL> out_variable
     * + <TD=td> <FROM=val> <TO=val>
     *
     * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable AT=val
     *
     * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable WHEN out_variable2=val
     * + <TD=td>
     * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     *
     * .MEASURE {DC|AC|TRAN} result DERIV<ATIVE> out_variable WHEN out_variable2=out_variable3
     * + <TD=td>
     * + <CROSS=# | CROSS=LAST> <RISE=#|RISE=LAST> <FALL=#|FALL=LAST>
     * ----------------------------------------------------------------- */

    int status;                 /* return status */
    char *item;                 /* parsing item */
    char *measure;              /* measure keyword */
    char *analysis;             /* analysis option */
    char *variable, *variable2; /* variable to trace */
    wordlist *measure_var;      /* wordlist of measurable */
    ANALYSIS_TYPE_T op;         /* measure function type */

    status = TRUE;
    measure = gettok(&line);
    if (!measure)
        return (status);

    analysis = gettok(&line);
    if (!analysis)
        return (status);

    if ((strcasecmp(analysis, "DC") == 0) ||
        (strcasecmp(analysis, "AC") == 0) ||
        (strcasecmp(analysis, "TRAN") == 0)) {
        analysis = copy(analysis);
    } else {
        /* sometimes operation is optional - for now just pick trans */
        analysis = copy("TRAN");
    }

    do {
        item = gettok(&line);
        if (item) {
            op = measure_function_type(item);
            if (op != AT_UNKNOWN) {
                /* We have a variable/complex variable coming next */
                variable = gettok_iv(&line);
                variable2 = NULL;
                if (*line == '=')
                    variable2 = gettok_iv(&line);
                if (variable) {
                    size_t len = strlen(item);
                    if (item[len-1] == '=') {
                    } else {
                        /* We may have something like V(n1)=1
                           or v(n1)=2 , same with i() */
                        char *sv = strip_ac_vec(variable, analysis);
                        measure_var = gettoks(sv);
                        com_save2(measure_var, analysis);
                        tfree(sv);
                        status = FALSE;
                    }
                }
                if (variable2) {
                    /* We may have something like  v(n1)=v(n2)
                       v(n2) is handled here, same with i() */
                    char *sv2 = strip_ac_vec(variable2, analysis);
                    measure_var = gettoks(sv2);
                    com_save2(measure_var, analysis);
                    tfree(sv2);
                    status = FALSE;
                }
            }
        }
    } while (*line);

    return status;
}


/* -----------------------------------------------------------------
 * Function: process a WHEN measurement statement which has been
 * parsed into a measurement structure.
 * ----------------------------------------------------------------- */
static int
com_measure_when(
    MEASUREPTR meas     /* in : parsed measurement structure */
    )
{
    int i, first;
    int riseCnt = 0;
    int fallCnt = 0;
    int crossCnt = 0;
    int section = -1;
    int measurement_pending;
    int init_measured_value;
    bool ac_check = FALSE, sp_check = FALSE, dc_check = FALSE, tran_check = FALSE;
    bool has_d2 = FALSE;
    double value, prevValue, value2, prevValue2;
    double scaleValue, prevScaleValue;

    enum ValSide { S_ABOVE_VAL, S_BELOW_VAL };
    enum ValEdge { E_RISING, E_FALLING };

    struct dvec *d, *d2, *dScale;

    d = vec_get(meas->m_vec);

    if (meas->m_vec2) {
        d2 = vec_get(meas->m_vec2);
        has_d2 = TRUE;
    } else {
        d2 = NULL;
    }

    dScale = plot_cur->pl_scale;

    if (d == NULL) {
        fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }

    if (has_d2 && (d2 == NULL)) {
        fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec2);
        return MEASUREMENT_FAILURE;
    }
    if (dScale == NULL) {
        fprintf(cp_err, "Error: no scale vector.\n");
        return MEASUREMENT_FAILURE;
    }

    if (dScale->v_realdata ==NULL && dScale->v_compdata == NULL) {
        fprintf(cp_err, "Error: scale vector time, frequency or dc has no data.\n");
        return MEASUREMENT_FAILURE;
    }

    prevValue = 0.;
    prevValue2 = 0.;
    prevScaleValue = 0.;
    first = 0;
    measurement_pending = 0;
    init_measured_value = 1;


    /* -----------------------------------------------------------------
     * Take the string tests outside of the loop for speed.
     * ----------------------------------------------------------------- */
    if (cieq (meas->m_analysis, "ac"))
        ac_check = TRUE;
    else if (cieq (meas->m_analysis, "sp"))
        sp_check = TRUE;
    else if (cieq (meas->m_analysis, "dc"))
        dc_check = TRUE;
    else {
        tran_check = TRUE;
        if (!d->v_realdata) {
            fprintf(stderr, "Error: no real data available for measurement (no tran simulation?)\n");
            return MEASUREMENT_FAILURE;
        }
    }

    for (i = 0; i < d->v_length; i++) {

        if (ac_check) {
            if (d->v_compdata)
                value = get_value(meas, d, i); //d->v_compdata[i].cx_real;
            else
                value = d->v_realdata[i];
            if (dScale->v_compdata)
                scaleValue = dScale->v_compdata[i].cx_real;
            else
                scaleValue = dScale->v_realdata[i];
        } else if (sp_check) {
            if (d->v_compdata)
                value = get_value(meas, d, i); //d->v_compdata[i].cx_real;
            else
                value = d->v_realdata[i];
            scaleValue = dScale->v_realdata[i];
        } else {
            value = d->v_realdata[i];
            scaleValue = dScale->v_realdata[i];
        }

        if (has_d2) {
            if (ac_check) {
                if (d2->v_compdata)
                    value2 = get_value(meas, d2, i); //d->v_compdata[i].cx_real;
                else
                    value2 = d2->v_realdata[i];
            } else if (sp_check) {
                if (d2->v_compdata)
                    value2 = get_value(meas, d2, i); //d->v_compdata[i].cx_real;
                else
                    value2 = d2->v_realdata[i];
            } else {
                value2 = d2->v_realdata[i];
            }
        } else {
            value2 = NAN;
        }

        /* 'dc' is special: it may start at an arbitrary scale value.
           Use m_td to store this value, as a delay TD does not make sense */
        if (dc_check && (i == 0))
            meas->m_td = scaleValue;
        /* if analysis tran, suppress values below TD */
        if (tran_check && (scaleValue < meas->m_td))
            continue;
        /* if analysis ac, sp, suppress values below 0 */
        else if ((ac_check || sp_check) && (scaleValue < 0))
            continue;

        if (dc_check) {
            /* dc: start from pos or neg scale value */
            if ((scaleValue < meas->m_from) || (scaleValue > meas->m_to))
                continue;
        } else {
            /* all others: start from neg scale value */
            if (scaleValue < meas->m_from)
                continue;

            if ((meas->m_to != 0.0e0) && (scaleValue > meas->m_to))
                break;
        }

        /* if 'dc': reset first if scale jumps back to origin */
        if ((first > 1) && (dc_check && (meas->m_td == scaleValue)))
            first = 1;

        if (first == 1) {
            if (has_d2) {
                // initialise
                crossCnt = 0;
                if (value < value2) {
                    section = S_BELOW_VAL;
                    if (prevValue >= prevValue2) {
                        fallCnt = 1;
                        crossCnt = 1;
                    }
                } else {
                    section = S_ABOVE_VAL;
                    if (prevValue < prevValue2) {
                        riseCnt = 1;
                        crossCnt = 1;
                    }
                }
                fflush(stdout);
            } else {
                // initialise
                crossCnt = 0;
                if (value < meas->m_val) {
                    section = S_BELOW_VAL;
                    if (prevValue >= meas->m_val) {
                        fallCnt = 1;
                        crossCnt = 1;
                    }
                } else {
                    section = S_ABOVE_VAL;
                    if (prevValue < meas->m_val) {
                        riseCnt = 1;
                        crossCnt = 1;
                    }
                }
                fflush(stdout);
            }
        }

        if (first > 1) {
            if (has_d2) {
                if ((section == S_BELOW_VAL) && (value >= value2)) {
                    section = S_ABOVE_VAL;
                    crossCnt++;
                    riseCnt++;
                    if (meas->m_fall != MEASURE_LAST_TRANSITION) {
                        /* we can measure rise/cross transition if the user
                         * has not requested a last fall transition */
                        measurement_pending = 1;
                    }

                } else if ((section == S_ABOVE_VAL) && (value <= value2)) {
                    section = S_BELOW_VAL;
                    crossCnt++;
                    fallCnt++;
                    if (meas->m_rise != MEASURE_LAST_TRANSITION) {
                        /* we can measure fall/cross transition if the user
                         * has not requested a last rise transition */
                        measurement_pending = 1;
                    }
                }

                if  ((crossCnt == meas->m_cross) || (riseCnt == meas->m_rise) || (fallCnt == meas->m_fall)) {
                    /* user requested an exact match of cross, rise, or fall
                     * exit when we meet condition */
//                meas->m_measured = prevScaleValue + (value2 - prevValue) * (scaleValue - prevScaleValue) / (value - prevValue);
                    meas->m_measured = prevScaleValue + (prevValue2 - prevValue) * (scaleValue - prevScaleValue) / (value - prevValue - value2 + prevValue2);
                    return MEASUREMENT_OK;
                }
                if  (measurement_pending) {
                    if ((meas->m_cross == MEASURE_DEFAULT) && (meas->m_rise == MEASURE_DEFAULT) && (meas->m_fall == MEASURE_DEFAULT)) {
                        /* user didn't request any option, return the first possible case */
                        meas->m_measured = prevScaleValue + (prevValue2 - prevValue) * (scaleValue - prevScaleValue) / (value - prevValue - value2 + prevValue2);
                        return MEASUREMENT_OK;
                    } else if ((meas->m_cross == MEASURE_LAST_TRANSITION) || (meas->m_rise == MEASURE_LAST_TRANSITION) || (meas->m_fall == MEASURE_LAST_TRANSITION)) {
                        meas->m_measured = prevScaleValue + (prevValue2 - prevValue) * (scaleValue - prevScaleValue) / (value - prevValue - value2 + prevValue2);
                        /* no return - look for last */
                        init_measured_value = 0;
                    }
                    measurement_pending = 0;
                }
            } else {
                if ((section == S_BELOW_VAL) && (value >= meas->m_val)) {
                    section = S_ABOVE_VAL;
                    crossCnt++;
                    riseCnt++;
                    if (meas->m_fall != MEASURE_LAST_TRANSITION) {
                        /* we can measure rise/cross transition if the user
                         * has not requested a last fall transition */
                        measurement_pending = 1;
                    }

                } else if ((section == S_ABOVE_VAL) && (value <= meas->m_val)) {
                    section = S_BELOW_VAL;
                    crossCnt++;
                    fallCnt++;
                    if (meas->m_rise != MEASURE_LAST_TRANSITION) {
                        /* we can measure fall/cross transition if the user
                         * has not requested a last rise transition */
                        measurement_pending = 1;
                    }
                }

                if  ((crossCnt == meas->m_cross) || (riseCnt == meas->m_rise) || (fallCnt == meas->m_fall)) {
                    /* user requested an exact match of cross, rise, or fall
                     * exit when we meet condition */
                    meas->m_measured = prevScaleValue + (meas->m_val - prevValue) * (scaleValue - prevScaleValue) / (value - prevValue);
                    return MEASUREMENT_OK;
                }
                if  (measurement_pending) {
                    if ((meas->m_cross == MEASURE_DEFAULT) && (meas->m_rise == MEASURE_DEFAULT) && (meas->m_fall == MEASURE_DEFAULT)) {
                        /* user didn't request any option, return the first possible case */
                        meas->m_measured = prevScaleValue + (meas->m_val - prevValue) * (scaleValue - prevScaleValue) / (value - prevValue);
                        return MEASUREMENT_OK;
                    } else if ((meas->m_cross == MEASURE_LAST_TRANSITION) || (meas->m_rise == MEASURE_LAST_TRANSITION) || (meas->m_fall == MEASURE_LAST_TRANSITION)) {
                        meas->m_measured = prevScaleValue + (meas->m_val - prevValue) * (scaleValue - prevScaleValue) / (value - prevValue);
                        /* no return - look for last */
                        init_measured_value = 0;
                    }
                    measurement_pending = 0;
                }
            }
        }
        first ++;

        prevValue = value;
        if (has_d2)
            prevValue2 = value2;
        prevScaleValue = scaleValue;
    }

    if (init_measured_value)
        meas->m_measured = NAN;

    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: process an AT measurement statement which has been
 * parsed into a measurement structure.  We make sure to interpolate
 * the value when appropriate.
 * ----------------------------------------------------------------- */
static int
measure_at(
    MEASUREPTR meas,            /* in : parsed "at" data */
    double at                   /* in: time to perform measurement */
    )
{
    int i;
    double value, pvalue, svalue, psvalue;
    bool ac_check = FALSE, sp_check = FALSE, dc_check = FALSE, tran_check = FALSE;
    struct dvec *d, *dScale;

    psvalue = pvalue = 0;

    if (meas->m_vec == NULL) {
        fprintf(stderr, "Error: Syntax error in meas line, missing vector\n");
        return MEASUREMENT_FAILURE;
    }

    d = vec_get(meas->m_vec);
    dScale = plot_cur->pl_scale;

    if (d == NULL) {
        fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }

    if (dScale == NULL) {
        fprintf(cp_err, "Error: no such vector time, frequency or dc.\n");
        return MEASUREMENT_FAILURE;
    }

    if (dScale->v_realdata == NULL && dScale->v_compdata == NULL) {
        fprintf(cp_err, "Error: scale vector time, frequency or dc has no data.\n");
        return MEASUREMENT_FAILURE;
    }

    /* -----------------------------------------------------------------
     * Take the string tests outside of the loop for speed.
     * ----------------------------------------------------------------- */
    if (cieq (meas->m_analysis, "ac"))
        ac_check = TRUE;
    else if (cieq (meas->m_analysis, "sp"))
        sp_check = TRUE;
    else if (cieq (meas->m_analysis, "dc"))
        dc_check = TRUE;
    else {
        tran_check = TRUE;
        if (!d->v_realdata) {
            fprintf(stderr, "Error: no real data available for measurement (no tran simulation?)\n");
            return MEASUREMENT_FAILURE;
        }
    }

    for (i = 0; i < d->v_length; i++) {
        if (ac_check) {
            if (d->v_compdata) {
                value = get_value(meas, d, i); //d->v_compdata[i].cx_real;
            } else {
                value = d->v_realdata[i];
                // fprintf(cp_err, "Warning: 'meas ac' input vector is real!\n");
            }
            if (dScale->v_compdata)
                svalue = dScale->v_compdata[i].cx_real;
            else
                svalue = dScale->v_realdata[i]; //prevent crash in case if buggy input
        } else if (sp_check) {
            if (d->v_compdata)
                value = get_value(meas, d, i); //d->v_compdata[i].cx_real;
            else
                value = d->v_realdata[i];
            if (dScale->v_compdata) // .sp, s-param.
                svalue = dScale->v_compdata[i].cx_real;
            else // fft
                svalue = dScale->v_realdata[i];
        } else {
            value = d->v_realdata[i];
            svalue = dScale->v_realdata[i];
        }

        if ((i > 0) && (psvalue <= at) && (svalue >= at)) {
            meas->m_measured = pvalue + (at - psvalue) * (value - pvalue) / (svalue - psvalue);
            return MEASUREMENT_OK;
        } else if  (dc_check && (i > 0) && (psvalue >= at) && (svalue <= at)) {
            meas->m_measured = pvalue + (at - psvalue) * (value - pvalue) / (svalue - psvalue);
            return MEASUREMENT_OK;
        }

        psvalue = svalue;
        pvalue = value;
    }

    meas->m_measured = NAN;
    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: measure the DERIVATIVE of a vector at a given scale
 * point (Enhancement-179 -- .meas DERIV was parsed but never
 * evaluated: it fell into the 'currently not supported' stub).
 * A quadratic through the three samples around `at` is
 * differentiated analytically, handling a nonuniform timestep; at
 * the interval ends it degrades to the two-point slope.
 * ----------------------------------------------------------------- */
static int
measure_deriv_at(
    MEASUREPTR meas,            /* in : parsed measurement data request */
    double at                   /* in : scale value to differentiate at */
    )
{
    int i, j;
    bool ac_check = FALSE, sp_check = FALSE;
    struct dvec *d, *dScale;

    if (meas->m_vec == NULL) {
        fprintf(stderr, "Error: Syntax error in meas line, missing vector\n");
        return MEASUREMENT_FAILURE;
    }

    d = vec_get(meas->m_vec);
    dScale = plot_cur->pl_scale;

    if (d == NULL) {
        fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }
    if (dScale == NULL || (dScale->v_realdata == NULL && dScale->v_compdata == NULL)) {
        fprintf(cp_err, "Error: scale vector time, frequency or dc has no data.\n");
        return MEASUREMENT_FAILURE;
    }

    if (cieq(meas->m_analysis, "ac"))
        ac_check = TRUE;
    else if (cieq(meas->m_analysis, "sp"))
        sp_check = TRUE;
    else if (!cieq(meas->m_analysis, "dc") && !d->v_realdata) {
        fprintf(stderr, "Error: no real data available for measurement (no tran simulation?)\n");
        return MEASUREMENT_FAILURE;
    }

    if (d->v_length < 2) {
        meas->m_measured = NAN;
        return MEASUREMENT_OK;
    }

#define DERIV_VAL(idx) (((ac_check || sp_check) && d->v_compdata) ? \
                        get_value(meas, d, (idx)) : d->v_realdata[(idx)])
#define DERIV_SCL(idx) (dScale->v_compdata ? dScale->v_compdata[(idx)].cx_real \
                                           : dScale->v_realdata[(idx)])

    /* find the interval [j-1, j] bracketing `at` */
    for (j = 1; j < d->v_length; j++) {
        double s0 = DERIV_SCL(j - 1), s1 = DERIV_SCL(j);
        if ((s0 <= at && at <= s1) || (s0 >= at && at >= s1))
            break;
    }
    if (j >= d->v_length) {
        meas->m_measured = NAN;
        return MEASUREMENT_OK;
    }

    /* three-point stencil around the bracket (shift at the ends) */
    i = j - 1;
    if (i == 0)
        i = 1;
    if (i > d->v_length - 2)
        i = d->v_length - 2;

    {
        double t0 = DERIV_SCL(i - 1), t1 = DERIV_SCL(i), t2 = DERIV_SCL(i + 1);
        double v0 = DERIV_VAL(i - 1), v1 = DERIV_VAL(i), v2 = DERIV_VAL(i + 1);
        double d01 = t1 - t0, d12 = t2 - t1, d02 = t2 - t0;

        if (d01 == 0.0 || d12 == 0.0 || d02 == 0.0) {
            /* repeated scale points (breakpoint duplicates): two-point slope */
            double sa = DERIV_SCL(j - 1), sb = DERIV_SCL(j);
            if (sb == sa) {
                meas->m_measured = NAN;
                return MEASUREMENT_OK;
            }
            meas->m_measured = (DERIV_VAL(j) - DERIV_VAL(j - 1)) / (sb - sa);
            return MEASUREMENT_OK;
        }
        /* derivative of the Lagrange quadratic through (t0,v0),(t1,v1),(t2,v2) */
        meas->m_measured =
            v0 * (2.0 * at - t1 - t2) / (d01 * d02)
          - v1 * (2.0 * at - t0 - t2) / (d01 * d12)
          + v2 * (2.0 * at - t0 - t1) / (d02 * d12);
    }

#undef DERIV_VAL
#undef DERIV_SCL

    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: process an MIN, MAX, or AVG statement which has been
 * parsed into a measurement structure.  We should make sure to interpolate
 * the value here when we have m_from and m_to constraints * so this
 * function is slightly wrong.   Need to fix in future rev.
 * ----------------------------------------------------------------- */
static int
measure_minMaxAvg(
    MEASUREPTR meas,                /* in : parsed measurement data request */
    ANALYSIS_TYPE_T mFunctionType   /* in: one of AT_AVG, AT_MIN, AT_MAX, AT_MIN_AT, AT_MAX_AT */
    )
{
    int i;
    struct dvec *d, *dScale;
    double value, svalue, mValue, mValueAt;
    double pvalue = 0.0, sprev = 0.0, Tsum = 0.0;
    /* Enhancement-303: the immediately preceding RAW sample (in or out of the window),
       needed to interpolate the exact point where a dc sweep crosses `from` or `to`. */
    double prev_s = 0.0, prev_v = 0.0;
    bool have_prev = FALSE;
    int first;
    bool ac_check = FALSE, sp_check = FALSE, dc_check = FALSE, tran_check = FALSE;

    mValue = 0;
    mValueAt = svalue = 0;
    meas->m_measured = NAN;
    meas->m_measured_at = NAN;
    first = 0;

    if (meas->m_vec == NULL) {
        fprintf(cp_err, "Syntax error in meas line\n");
        return MEASUREMENT_FAILURE;
    }

    d = vec_get(meas->m_vec);
    if (d == NULL) {
        fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }


    /* -----------------------------------------------------------------
     * Take the string tests outside of the loop for speed.
     * ----------------------------------------------------------------- */
    if (cieq (meas->m_analysis, "ac"))
        ac_check = TRUE;
    else if (cieq (meas->m_analysis, "sp"))
        sp_check = TRUE;
    else if (cieq (meas->m_analysis, "dc"))
        dc_check = TRUE;
    else
        tran_check = TRUE;

    if (ac_check || sp_check) {
        dScale = vec_get("frequency");
        if (dScale == NULL) {
            fprintf(cp_err, "Error: meas %s ...\n", meas->m_analysis);
            fprintf(cp_err, " no such scale vector as frequency.\n");
            return MEASUREMENT_FAILURE;
        }
    } else if (tran_check) {
        dScale = vec_get("time");
        if (dScale == NULL) {
            fprintf(cp_err, "Error: meas %s ...\n", meas->m_analysis);
            fprintf(cp_err, " no such scale vector as time.\n");
            return MEASUREMENT_FAILURE;
        }
    } else if (dc_check) {
        dScale = vec_get("v-sweep");
        if (!dScale) {
            dScale = vec_get("i-sweep");
            if (!dScale) {
                dScale = vec_get("temp-sweep");
                if (!dScale)
                    dScale = vec_get("res-sweep");
            }
        }
        if (dScale == NULL) {
            fprintf(cp_err, "Error: meas %s ...\n", meas->m_analysis);
            fprintf(cp_err, " no such scale vector as v-sweep, i-sweep, temp-sweep, or res-sweep.\n");
            return MEASUREMENT_FAILURE;
        }
    } else {                    /* error */
        fprintf(cp_err, "Error: no such analysis type as %s.\n", meas->m_analysis);
        return MEASUREMENT_FAILURE;
    }

    if (dScale->v_realdata == NULL && dScale->v_compdata == NULL) {
        fprintf(cp_err, "Error: scale vector %s has no data.\n", dScale->v_name);
        return MEASUREMENT_FAILURE;
    }

    if (d->v_length != dScale->v_length) {
        fprintf(cp_err, "Error: length of scale vector (%s) does not match length of data vector (%s).\n", dScale->v_name, d->v_name);
        fprintf(cp_err, "    Analysis is not possible.\n");
        return MEASUREMENT_FAILURE;
    }

    for (i = 0; i < d->v_length; i++) {
        if (ac_check) {
            if (d->v_compdata) {
                value = get_value(meas, d, i); //d->v_compdata[i].cx_real;
            } else {
                value = d->v_realdata[i];
                // fprintf(cp_err, "Warning: 'meas ac' input vector is real!\n");
            }
            svalue = dScale->v_compdata[i].cx_real;
        } else if (sp_check) {
            if (d->v_compdata)
                value = get_value(meas, d, i); //d->v_compdata[i].cx_real;
            else
                value = d->v_realdata[i];
            if (dScale->v_realdata)
                svalue = dScale->v_realdata[i];
            else
                /* may happen if you write an sp vector and load it again */
                svalue = dScale->v_compdata[i].cx_real;
        } else {
            value = d->v_realdata[i];
            svalue = dScale->v_realdata[i];
        }

        if (dc_check) {
            /* dc: start from pos or neg scale value */
            bool in_win = !((svalue < meas->m_from) || (svalue > meas->m_to));

            /* Enhancement-303: clip an AVG window to exactly [from, to] on a dc sweep
               too. Unlike time, a dc sweep may DESCEND (`dc v1 2 0 -0.001`), so the
               boundary cannot be assumed to be the lower one: work from the actual
               crossing between the previous raw sample and this one, which is
               direction-agnostic and simply never fires on a sweep that does not
               cross. Without this the window silently began at the first sweep point
               inside it and ended at the last -- the same O(step) truncation fixed for
               tran/ac/sp in Enhancement-302. */
            if (mFunctionType == AT_AVG && have_prev && svalue != prev_s) {
                bool prev_in = !((prev_s < meas->m_from) || (prev_s > meas->m_to));

                if (in_win && !prev_in && first == 0) {
                    /* crossing INTO the window: open it at the exact boundary */
                    double b = (prev_s < meas->m_from) ? meas->m_from : meas->m_to;
                    double t = (b - prev_s) / (svalue - prev_s);
                    mValue = 0.0;
                    Tsum = 0.0;
                    pvalue = prev_v + t * (value - prev_v);
                    sprev = b;
                    mValueAt = b;
                    first = 1;          /* this sample now accumulates against `b` */
                } else if (!in_win && prev_in && first != 0) {
                    /* crossing OUT of it: close at the exact boundary and stop */
                    double b = (svalue > meas->m_to) ? meas->m_to : meas->m_from;
                    double t = (b - prev_s) / (svalue - prev_s);
                    double bv = prev_v + t * (value - prev_v);
                    mValue += 0.5 * (bv + pvalue) * (b - sprev);
                    Tsum += (b - sprev);
                    sprev = b;
                    break;
                }
            }

            prev_s = svalue;
            prev_v = value;
            have_prev = TRUE;

            if (!in_win)
                continue;
        } else {
            /* all others: start from neg scale value */
            if (svalue < meas->m_from)
                continue;

            if ((meas->m_to != 0.0e0) && (svalue > meas->m_to)) {
                /* Enhancement-302: AVG integrates a trapezoid over the window and
                   divides by its span, so the window has to END AT EXACTLY `to`.
                   Without this it ended at the last sample BEFORE `to`, dropping the
                   final partial interval -- an O(dt) error that made `avg` disagree
                   with `integ`/(to-from) computed over the same window, even though
                   they are the same quantity. RMS and INTEG already clip this way in
                   measure_rms_integral(); do the same here. MIN/MAX keep their
                   whole-sample semantics. */
                if (mFunctionType == AT_AVG && first != 0 && i > 0) {
                    /* Enhancement-316: always add the final partial interval up to `to`.
                       The !AlmostEqualUlps guard used to gate the whole accumulation, so
                       when the first out-of-window sample fell within 100 ULPs of `to`
                       the entire final trapezoid [sprev, to] -- a full timestep -- was
                       dropped, ending AVG's window one step short of `to`. That made
                       `avg` disagree with `integ`/(to-from) over the same window (they
                       are the same quantity -- the Enhancement-302 contract). INTEG/RMS
                       in measure_rms_integral() ALWAYS add the final point, interpolating
                       only when the sample overshoots `to`; match that here -- interpolate
                       to `to` only when the sample is more than 100 ULPs beyond it, else
                       use it as-is, but add the trapezoid either way. */
                    if (!AlmostEqualUlps(svalue, meas->m_to, 100)) {
                        value = measure_interpolate(dScale, d, i - 1, i, meas->m_to,
                                                    'y', meas);
                        svalue = meas->m_to;
                    }
                    mValue += 0.5 * (value + pvalue) * (svalue - sprev);
                    Tsum += (svalue - sprev);
                    pvalue = value;
                    sprev = svalue;
                }
                break;
            }
        }

        if (first == 0) {
            first = 1;

            switch (mFunctionType) {
            case AT_MIN:
            case AT_MIN_AT:
            case AT_MAX_AT:
            case AT_MAX:
                mValue = value;
                mValueAt = svalue;
                break;
            case AT_AVG:
                mValue = 0.0;
                mValueAt = svalue;
                Tsum = 0.0;
                /* Enhancement-302: and START at exactly `from`, for the same reason --
                   otherwise the first partial interval is dropped as well.
                   NOT for a dc sweep: only time/frequency scales are guaranteed to
                   ascend. A descending `dc v1 2 0 -0.001` enters the window at its
                   HIGH end, where forcing svalue to `from` would extrapolate far
                   outside [i-1, i] and poison every later trapezoid. */
                if (!dc_check && meas->m_from != 0.0e0 && i > 0 &&
                        !AlmostEqualUlps(svalue, meas->m_from, 100)) {
                    value = measure_interpolate(dScale, d, i - 1, i, meas->m_from,
                                                'y', meas);
                    svalue = meas->m_from;
                    mValueAt = svalue;
                }
                pvalue = value;
                sprev = svalue;
                break;
            default:
                fprintf(cp_err, "Error: improper min/max/avg call.\n");
                return MEASUREMENT_FAILURE;
            }
        } else {
            switch (mFunctionType) {
            case AT_MIN:
            case AT_MIN_AT: {
                if (value <= mValue) {
                    mValue = value;
                    mValueAt = svalue;
                }
                break;
            }
            case AT_MAX_AT:
            case AT_MAX: {
                if (value >= mValue) {
                    mValue = value;
                    mValueAt = svalue;
                }
                break;
            }
            case AT_AVG: {
                mValue += 0.5 * (value + pvalue) * (svalue - sprev);
                Tsum += (svalue - sprev);
                pvalue = value;
                sprev = svalue;
                break;
            }
            default :
                fprintf(cp_err, "Error: improper min/max/avg call.\n");
                return MEASUREMENT_FAILURE;
            }

        }
    }

    switch (mFunctionType)
    {
    case AT_AVG: {
        meas->m_measured = mValue / (first ? Tsum : 1.0);
        /* Enhancement-302: report the END OF THE WINDOW, not `svalue` -- which, when
           the loop broke on the first out-of-window sample, was a time the average
           never covered (a 250us window was echoed as "to=2.50280e-04"). `sprev` is
           the last point actually accumulated, i.e. the true window end.
           Enhancement-303: this now holds for dc as well -- but a dc sweep may DESCEND,
           and then the accumulation ends at the LOW bound, which would print as
           "from= 0.25 to= 0.25". The covered window is [mValueAt, sprev] in traversal
           order, so report its upper end and the echo reads the same either way. */
        meas->m_measured_at = first ? ((mValueAt > sprev) ? mValueAt : sprev) : svalue;
        break;
    }
    case AT_MIN:
    case AT_MAX:
    case AT_MIN_AT:
    case AT_MAX_AT: {
        meas->m_measured = mValue;
        meas->m_measured_at = mValueAt;
        break;
    }
    default :
        fprintf(cp_err, "Error: improper min/max/avg call.\n");
        return MEASUREMENT_FAILURE;
    }
    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: gain margin / phase margin from an AC loop-gain response.
 *   phase_margin: PM = 180 + phase, evaluated at the (first) gain
 *                 crossover where |gain| = 1 (0 dB).
 *   gain_margin:  GM = -gain(dB), evaluated at the (first) phase
 *                 crossover where the phase = -180 deg.
 * The phase is UNWRAPPED first -- ngspice's vp() is wrapped to
 * (-180,180], so the raw phase never actually equals -180 and a
 * `when vp=-180' measure cannot find the phase crossover. m_measured
 * holds the margin, m_measured_at the crossover frequency.
 * ----------------------------------------------------------------- */
static int
measure_margin(
    MEASUREPTR meas,              /* in : parsed measurement (m_vec = loop gain) */
    ANALYSIS_TYPE_T mFunctionType /* in: AT_PHASE_MARGIN or AT_GAIN_MARGIN */
    )
{
    struct dvec *d, *dScale;
    int i, n;
    double *gdb, *ph, *frq;
    double prev, offset;

    meas->m_measured = NAN;
    meas->m_measured_at = NAN;

    if (meas->m_vec == NULL) {
        fprintf(cp_err, "Syntax error in meas line\n");
        return MEASUREMENT_FAILURE;
    }
    if (!cieq(meas->m_analysis, "ac") && !cieq(meas->m_analysis, "sp")) {
        fprintf(cp_err, "Error: gain_margin/phase_margin require an 'ac' measurement.\n");
        return MEASUREMENT_FAILURE;
    }
    d = vec_get(meas->m_vec);
    if (d == NULL) {
        fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }
    if (d->v_compdata == NULL) {
        fprintf(cp_err, "Error: %s has no complex AC data for a margin measurement.\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }
    dScale = vec_get("frequency");
    if (dScale == NULL) {
        fprintf(cp_err, "Error: no such scale vector as frequency.\n");
        return MEASUREMENT_FAILURE;
    }

    n = d->v_length;
    if (n < 2)
        return MEASUREMENT_FAILURE;
    gdb = TMALLOC(double, n);
    ph  = TMALLOC(double, n);
    frq = TMALLOC(double, n);
    prev = 0.0;
    offset = 0.0;
    for (i = 0; i < n; i++) {
        double re = d->v_compdata[i].cx_real;
        double im = d->v_compdata[i].cx_imag;
        /* phase in DEGREES always -- margins are conventionally in degrees, and the
         * radtodeg() macro only converts when the runtime cx_degrees flag is set
         * (otherwise it returns radians). */
        double p = atan2(im, re) * (180.0 / M_PI);
        gdb[i] = 20.0 * log10(hypot(re, im));
        if (i > 0) {                 /* unwrap: undo the +-360 jumps of atan2 */
            double dp = p - prev;
            if (dp > 180.0) offset -= 360.0;
            else if (dp < -180.0) offset += 360.0;
        }
        prev = p;
        ph[i] = p + offset;
        frq[i] = dScale->v_compdata ? dScale->v_compdata[i].cx_real : dScale->v_realdata[i];
    }

    for (i = 1; i < n; i++) {
        double t, fc;
        if (mFunctionType == AT_PHASE_MARGIN) {
            /* first 0 dB gain crossover */
            if ((gdb[i-1] >= 0.0 && gdb[i] < 0.0) || (gdb[i-1] < 0.0 && gdb[i] >= 0.0)) {
                t = (0.0 - gdb[i-1]) / (gdb[i] - gdb[i-1]);
                meas->m_measured = 180.0 + (ph[i-1] + t * (ph[i] - ph[i-1]));
            } else {
                continue;
            }
        } else {
            /* first -180 deg phase crossover (unwrapped) */
            if ((ph[i-1] >= -180.0 && ph[i] < -180.0) || (ph[i-1] < -180.0 && ph[i] >= -180.0)) {
                t = (-180.0 - ph[i-1]) / (ph[i] - ph[i-1]);
                meas->m_measured = -(gdb[i-1] + t * (gdb[i] - gdb[i-1]));
            } else {
                continue;
            }
        }
        /* interpolate the crossover frequency in log space (log-spaced sweep) */
        if (frq[i-1] > 0.0 && frq[i] > 0.0)
            fc = exp(log(frq[i-1]) + t * (log(frq[i]) - log(frq[i-1])));
        else
            fc = frq[i-1] + t * (frq[i] - frq[i-1]);
        meas->m_measured_at = fc;
        break;
    }

    tfree(gdb);
    tfree(ph);
    tfree(frq);
    return isnan(meas->m_measured) ? MEASUREMENT_FAILURE : MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: process an RMS or INTEG statement which has been
 * parsed into a measurement structure.  Here we do interpolate
 * the starting and stopping time window so the answer is correct.
 * ----------------------------------------------------------------- */
static int
measure_rms_integral(
    MEASUREPTR meas,              /* in : parsed measurement data request */
    ANALYSIS_TYPE_T mFunctionType /* in: one of AT_RMS, or AT_INTEG */
    )
{
    int i;                      /* counter */
    /* Enhancement-304: `n` walks the samples in order of increasing scale (which is
       index order unless a dc sweep descends); `i` is the index it maps to and `iprev`
       the one before it in that order. */
    int n, iprev;
    bool ascending = TRUE;
    int xy_size;                /* # of temp array elements */
    struct dvec *d, *xScale;    /* value and indpendent (x-axis) vectors */
    double value, xvalue;       /* current value and independent value */
    double *x;                  /* temp x array */
    double *y;                  /* temp y array */
    double toVal;               /* to time value */
    double *width;              /* temp width array */
    double sum1;                /* first sum */
    double sum2;                /* second sum */
    double sum3;                /* third sum */
    int first;
    bool ac_check = FALSE, sp_check = FALSE, dc_check = FALSE, tran_check = FALSE;

    xvalue = 0;
    meas->m_measured = NAN;
    meas->m_measured_at = NAN;
    first = 0;

    if (cieq (meas->m_analysis, "ac"))
        ac_check = TRUE;
    else if (cieq (meas->m_analysis, "sp"))
        sp_check = TRUE;
    else if (cieq (meas->m_analysis, "dc"))
        dc_check = TRUE;
    else
        tran_check = TRUE;

    d = vec_get(meas->m_vec);
    if (d == NULL) {
        fprintf(cp_err, "Error: no such vector as %s.\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }

    if (ac_check || sp_check) {
        xScale = vec_get("frequency");
        if (xScale == NULL) {
            fprintf(cp_err, "Error: meas %s ...\n", meas->m_analysis);
            fprintf(cp_err, " no such scale vector as frequency.\n");
            return MEASUREMENT_FAILURE;
        }
    } else if (tran_check) {
        xScale = vec_get("time");
        if (xScale == NULL) {
            fprintf(cp_err, "Error: meas %s ...\n", meas->m_analysis);
            fprintf(cp_err, " no such scale vector as time.\n");
            return MEASUREMENT_FAILURE;
        }
    } else if (dc_check) {
        xScale = vec_get("v-sweep");
        if (!xScale) {
            xScale = vec_get("i-sweep");
            if (!xScale) {
                xScale = vec_get("temp-sweep");
                if (!xScale)
                    xScale = vec_get("res-sweep");
            }
        }
        if (xScale == NULL) {
            fprintf(cp_err, "Error: meas %s ...\n", meas->m_analysis);
            fprintf(cp_err, " no such scale vector as v-sweep, i-sweep, temp-sweep, or res-sweep.\n");
            return MEASUREMENT_FAILURE;
        }
    } else {                      /* error */
        fprintf(cp_err, "Error: no such analysis type as %s.\n", meas->m_analysis);
        return MEASUREMENT_FAILURE;
    }

    if (xScale->v_realdata == NULL && xScale->v_compdata == NULL) {
        fprintf(cp_err, "Error: scale vector time, frequency or ?-sweep has no data.\n");
        return MEASUREMENT_FAILURE;
    }

    /* Allocate buffers for calculation. */
    x     = TMALLOC(double, xScale->v_length);
    y     = TMALLOC(double, xScale->v_length);
    width = TMALLOC(double, xScale->v_length + 1);

    xy_size = 0;
    toVal = -1;

    /* Enhancement-304: this loop assumes the scale ASCENDS -- it clips at `from` on the
       way in and at `to` on the way out, and the segment widths x[i+1]-x[i] it builds
       must come out positive for the Simpson / trapezoid sums below. A dc sweep may
       DESCEND (`dc v1 2 0 -0.001`), and then the very first sample is already above
       `to`: the clip fired at once, interpolated against index i-1 == -1 (an OUT-OF-
       BOUNDS read) and broke with a single point, so the sums ran zero times and the
       measurement came back 0.0 with `from= nan`. Walk the samples in order of
       increasing scale instead of index order; for ascending data this is the identity
       and every step below is unchanged. */
    {
        double s0, sN;
        if (ac_check) {
            s0 = xScale->v_compdata[0].cx_real;
            sN = xScale->v_compdata[d->v_length - 1].cx_real;
        } else if (xScale->v_realdata) {
            s0 = xScale->v_realdata[0];
            sN = xScale->v_realdata[d->v_length - 1];
        } else {
            s0 = xScale->v_compdata[0].cx_real;
            sN = xScale->v_compdata[d->v_length - 1].cx_real;
        }
        if (d->v_length > 1 && sN < s0)
            ascending = FALSE;
    }

    /* create new set of values over interval [from, to] -- interpolate if necessary */
    for (n = 0; n < d->v_length; n++) {
        /* index of this sample, and of the one before it IN TRAVERSAL ORDER */
        i = ascending ? n : (d->v_length - 1 - n);
        iprev = ascending ? (i - 1) : (i + 1);

        if (ac_check) {
            if (d->v_compdata) {
                value = get_value(meas, d, i); //d->v_compdata[i].cx_real;
            } else {
                value = d->v_realdata[i];
                // fprintf(cp_err, "Warning: 'meas ac' input vector is real!\n");
            }
            xvalue = xScale->v_compdata[i].cx_real;
        } else {
            value = d->v_realdata[i];
            xvalue = xScale->v_realdata[i];
        }

        if (xvalue < meas->m_from)
            continue;

        if ((meas->m_to != 0.0e0) && (xvalue > meas->m_to)) {
            /* interpolate ending value if necessary. `n > 0` is what keeps `iprev` in
               range -- it is the guard the old `i-1` lacked. With no preceding sample
               there is nothing to interpolate against, so the window simply has not
               been entered yet. */
            if (n > 0 && !AlmostEqualUlps(xvalue, meas->m_to, 100)) {
                value = measure_interpolate(xScale, d, iprev, i, meas->m_to, 'y', meas);
                xvalue = meas->m_to;
            } else if (n == 0) {
                continue;
            }
            x[xy_size] = xvalue;
            if (mFunctionType == AT_RMS)
                y[xy_size++] = value * value;
            else
                y[xy_size++] = value;
            toVal = xvalue;
            break;
        }

        if (first == 0) {
            if (meas->m_from != 0.0e0 && (n > 0)) {
                // interpolate starting value.
                if (!AlmostEqualUlps(xvalue, meas->m_from, 100)) {
                    value = measure_interpolate(xScale, d, iprev, i, meas->m_from, 'y' , meas);
                    xvalue = meas->m_from;
                }
            }
            meas->m_measured_at = xvalue;
            first = 1;
        }
        x[xy_size] = xvalue;
        if (mFunctionType == AT_RMS)
            y[xy_size++] = value * value;
        else
            y[xy_size++] = value;
    }

    // evaluate segment width
    for (i = 0; i < xy_size-1; i++)
        width[i] = x[i+1] - x[i];
    width[i++] = 0;
    width[i++] = 0;

    // Compute Integral (area under curve)
    i = 0;
    sum1 = sum2 = sum3 = 0.0;
    while (i < xy_size-1) {
        // Simpson's 3/8 Rule
        if (AlmostEqualUlps(width[i], width[i+1], 100) &&
            AlmostEqualUlps(width[i], width[i+2], 100)) {
            sum1 += 3*width[i] * (y[i] + 3*(y[i+1] + y[i+2]) + y[i+3]) / 8.0;
            i += 3;
        }
        // Simpson's 1/3 Rule
        else if (AlmostEqualUlps(width[i], width[i+1], 100)) {
            sum2 += width[i] * (y[i] + 4*y[i+1] + y[i+2]) / 3.0;
            i += 2;
        }
        // Trapezoidal Rule
        else {
            sum3 += width[i] * (y[i] + y[i+1]) / 2;
            i++;
        }
    }

    /* Now set the measurement values if not set.
       Enhancement-304: the window ran to the end of the data rather than hitting `to`,
       so the end of it is simply the last point appended. Reading
       `xScale->...[d->v_length-1]` was wrong whenever the scale descends (that is the
       SMALLEST scale value, not the last one traversed), and the `value`/`xvalue` reads
       that used to sit here indexed with the integration loop's counter -- past the end
       of the data -- into variables nothing reads again. Both are gone. */
    if (toVal < 0.0)
        toVal = (xy_size > 0) ? x[xy_size-1] : meas->m_measured_at;
    meas->m_from = meas->m_measured_at;
    meas->m_to = toVal;

    if (mFunctionType == AT_RMS) {
        meas->m_measured = (sum1 + sum2 + sum3)/ (toVal - meas->m_measured_at);
        meas->m_measured = sqrt(meas->m_measured);

    } else {
        meas->m_measured = (sum1 + sum2 + sum3);
    }

    txfree(x);
    txfree(y);
    txfree(width);
    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: Wrapper function to process a RMS measurement.
 * ----------------------------------------------------------------- */
#if 0
static void
measure_rms(
    MEASUREPTR meas             /* in : parsed measurement data request */
    )
{
    // RMS (root mean squared):
    // Calculates the square root of the area under the 'out_var2' curve
    //  divided be the period of interest
    measure_rms_integral(meas, AT_RMS);
}
#endif


/* -----------------------------------------------------------------
 * Function: Wrapper function to process a integration measurement.
 * ----------------------------------------------------------------- */
#if 0
static void
measure_integ(
    MEASUREPTR meas             /* in : parsed measurement data request */
    )
{
    // INTEGRAL INTEG
    measure_rms_integral(meas, AT_INTEG);
}
#endif


/* still some more work to do.... */
#if 0
static void
measure_deriv(void)
{
    // DERIVATIVE DERIV
}
#endif


// ERR Equations
#if 0
static void
measure_ERR(void)
{
}

static void
measure_ERR1(void)
{
}

static void
measure_ERR2(void)
{
}

static void
measure_ERR3(void)
{
}
#endif


void
com_dotmeasure(wordlist *wl)
{
    NG_IGNORE(wl);

    /* simulation info */
    // printf("*%s\n", plot_cur->pl_title);
    // printf("\t %s, %s\n", plot_cur->pl_name, plot_cur->pl_date); // missing temp
}


/* -----------------------------------------------------------------
 * Function: Given a measurement variable name, see if the analysis
 * has generated a measure vector for it.  Returns TRUE if it exists
 * or varname is NULL,  Return FALSE otherwise
 * ----------------------------------------------------------------- */
static int
measure_valid_vector(
    char *varname               /* in: requested variable name */
    )
{
    struct dvec *d;             /* measurement vector */
    char* ptr;
    long num;

    if (varname == NULL)
        return TRUE;

    /* If varname is a simple number, don't use this as a
    name of a vetor, but as a number */
    num = strtol(varname, &ptr, 10);
    if (*ptr == '\0')
        return FALSE;

    d = vec_get(varname);
    if (d == NULL)
        return FALSE;

    return TRUE;
}


/* -----------------------------------------------------------------
 * Function: Given a wordlist and measurement structure, parse the
 * standard parameters such as RISE, FALL, VAL, TD, FROM, TO, etc.
 * in a measurement statement.   We also check the appropriate
 * variables found in the measurement statement.
 * ----------------------------------------------------------------- */
static int
measure_parse_stdParams(
    MEASUREPTR meas,          /* in : measurement structure */
    wordlist *wl,             /* in : word list to parse */
    wordlist *wlBreak,        /* out: where we stopped parsing */
    char *errbuf              /* in/out: buffer where we write error messages */
    )
{
    int pCnt;
    char *p, *pName = NULL, *pValue;
    double engVal1;

    pCnt = 0;
    while (wl != wlBreak) {
        p = wl->wl_word;
        pName = strtok(p, "=");
        pValue = strtok(NULL, "=");

        if (pValue == NULL) {
            if (strcasecmp(pName, "LAST") == 0) {
                meas->m_cross = MEASURE_LAST_TRANSITION;
                meas->m_rise = -1;
                meas->m_fall = -1;
                pCnt ++;
                wl = wl->wl_next;
                continue;
            } else {
                snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax. equal sign missing ?\n");
                return MEASUREMENT_FAILURE;
            }
        }

        if (strcasecmp(pValue, "LAST") == 0) {
            engVal1 = MEASURE_LAST_TRANSITION;
        }
        else {
            if (ft_numparse(&pValue, FALSE, &engVal1) < 0) {
                snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax, cannot evaluate right hand side of %s=%s\n", pName, pValue);
                return MEASUREMENT_FAILURE;
            }
        }

        if (strcasecmp(pName, "RISE") == 0) {
            meas->m_rise = (int)floor(engVal1 + 0.5);
            meas->m_fall = -1;
            meas->m_cross = -1;
        } else if (strcasecmp(pName, "FALL") == 0) {
            meas->m_fall = (int)floor(engVal1 + 0.5);
            meas->m_rise = -1;
            meas->m_cross = -1;
        } else if (strcasecmp(pName, "CROSS") == 0) {
            meas->m_cross = (int)floor(engVal1 + 0.5);
            meas->m_rise = -1;
            meas->m_fall = -1;
        } else if (strcasecmp(pName, "VAL") == 0) {
            meas->m_val = engVal1;
        } else if (strcasecmp(pName, "TD") == 0) {
            meas->m_td = engVal1;
        } else if (strcasecmp(pName, "FROM") == 0) {
            meas->m_from = engVal1;
        } else if (strcasecmp(pName, "TO") == 0) {
            meas->m_to = engVal1;
        } else if (strcasecmp(pName, "AT") == 0) {
            meas->m_at = engVal1;
        } else {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "no such parameter as '%s'\n", pName);
            return MEASUREMENT_FAILURE;
        }

        pCnt ++;
        wl = wl->wl_next;
    }

    if (pCnt == 0) {
        if (pName)
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax of %s\n", pName);
        else
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax of\n");
        return MEASUREMENT_FAILURE;
    }

    // valid vector
    if (measure_valid_vector(meas->m_vec) == 0) {
        snprintf(errbuf, MEAS_ERRBUF_SIZE, "no such vector as '%s'\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }

    // valid vector2
    if (meas->m_vec2 != NULL)
        if (measure_valid_vector(meas->m_vec2) == 0) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "no such vector as '%s'\n", meas->m_vec2);
            return MEASUREMENT_FAILURE;
        }

    /* dc: make m_from always less than m_to */
    if (cieq("dc", meas->m_analysis))
        if (meas->m_to < meas->m_from) {
            SWAP(double, meas->m_from, meas->m_to);
        }

    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: Given a wordlist and measurement structure, parse a
 * FIND measurement statement.   Most of the work is done by calling
 * measure_parse_stdParams.
 * ----------------------------------------------------------------- */
static int
measure_parse_find(
    MEASUREPTR meas,          /* in : measurement structure */
    wordlist *wl,             /* in : word list to parse */
    wordlist *wlBreak,        /* out: where we stopped parsing */
    char *errbuf              /* in/out: buffer where we write error messages */
    )
{
    int pCnt;

    meas->m_vec = NULL;
    meas->m_vec2 = NULL;
    meas->m_val = 1e99;
    meas->m_cross = -1;
    meas->m_fall = -1;
    meas->m_rise = -1;
    meas->m_td = 0;
    meas->m_from = 0.0e0;
    meas->m_to = 0.0e0;
    meas->m_at = 1e99;

    /* for DC, set new outer limits for 'from' and 'to'
       because 0.0e0 may be valid inside of range */
    if (cieq("dc", meas->m_analysis)) {
        meas->m_to = 1.0e99;
        meas->m_from = -1.0e99;
    }

    pCnt = 0;
    while (wl != wlBreak) {
        char *p = wl->wl_word;

        if (pCnt == 0) {
            meas->m_vec = cp_unquote(wl->wl_word);
            /* correct for vectors like vm, vp etc. */
            if (cieq("ac", meas->m_analysis) || cieq("sp", meas->m_analysis))
                correct_vec(meas);
        } else if (pCnt == 1) {
            char * const pName = strtok(p, "=");
            char * const pVal = strtok(NULL, "=");

            if (pVal == NULL) {
                snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax of WHEN\n");
                return MEASUREMENT_FAILURE;
            }

            if (strcasecmp(pName, "AT") == 0) {
                if (ft_numparse((char **) &pVal, FALSE, &meas->m_at) < 0) {
                    snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax of WHEN\n");
                    return MEASUREMENT_FAILURE;
                }
            }
            else {
                snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax of WHEN\n");
                return MEASUREMENT_FAILURE;
            }
        } else {
            if (measure_parse_stdParams(meas, wl, NULL, errbuf) ==
                    MEASUREMENT_FAILURE)
                return MEASUREMENT_FAILURE;
        }

        wl = wl->wl_next;
        pCnt ++;
    }

    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: Given a wordlist and measurement structure, parse a
 * WHEN measurement statement.   Most of the work is done by calling
 * measure_parse_stdParams.
 * ----------------------------------------------------------------- */
static int
measure_parse_when(
    MEASUREPTR meas,          /* in : measurement structure */
    wordlist *wl,             /* in : word list to parse */
    char *errBuf              /* in/out: buffer where we write error messages */
    )
{
    int pCnt, err = 0;
    char *p, *pVar1, *pVar2;
    meas->m_vec = NULL;
    meas->m_vec2 = NULL;
    meas->m_val = 1e99;
    meas->m_cross = -1;
    meas->m_fall = -1;
    meas->m_rise = -1;
    meas->m_td = 0;
    meas->m_from = 0.0e0;
    meas->m_to = 0.0e0;
    meas->m_at = 1e99;


    /* for DC, set new outer limits for 'from' and 'to'
       because 0.0e0 may be valid inside of range */
    if (cieq("dc", meas->m_analysis)) {
        meas->m_to = 1.0e99;
        meas->m_from = -1.0e99;
    }

    pCnt = 0;
    while (wl) {
        p = wl->wl_word;

        if (pCnt == 0) {
            pVar1 = strtok(p, "=");
            pVar2 = strtok(NULL, "=");

            if (pVar2 == NULL) {
                snprintf(errBuf, MEAS_ERRBUF_SIZE, "bad syntax\n");
                return MEASUREMENT_FAILURE;
            }

            meas->m_vec = copy(pVar1);
            /* correct for vectors like vm, vp etc. */
            if (cieq("ac", meas->m_analysis) || cieq("sp", meas->m_analysis))
                correct_vec(meas);
            if (measure_valid_vector(pVar2) == 1) {
                meas->m_vec2 = copy(pVar2);
                /* correct for vectors like vm, vp etc. */
                if (cieq("ac", meas->m_analysis) || cieq("sp", meas->m_analysis))
                    correct_vec(meas);
            } else {
                if (str_has_arith_char2(pVar2)) {
                    snprintf(errBuf, MEAS_ERRBUF_SIZE, "Expressions like %s are not supported.\n", pVar2);
                    return MEASUREMENT_FAILURE;
                }
                meas->m_val = INPevaluate2(&pVar2, &err, 0);
                if (err || (*pVar2 && (!strchr("avfc", *pVar2) || *(pVar2 + 1) != '\0'))) {
                    snprintf(errBuf, MEAS_ERRBUF_SIZE, "Cannot evaluate %s \n", pVar2);
                    return MEASUREMENT_FAILURE;
                }
            }
        } else {
            if (measure_parse_stdParams(meas, wl, NULL, errBuf) == MEASUREMENT_FAILURE)
                return MEASUREMENT_FAILURE;
            break;
        }

        wl = wl->wl_next;
        pCnt ++;
    }
    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: Given a wordlist and measurement structure, parse a
 * TRIGGER or TARGET clause of a measurement statement.   Most of the
 * work is done by calling measure_parse_stdParams.
 * ----------------------------------------------------------------- */
static int
measure_parse_trigtarg(
    MEASUREPTR meas,          /* in : measurement structure */
    wordlist *words,          /* in : word list to parse */
    wordlist *wlTarg,         /* out : where we stopped parsing target clause */
    char *trigTarg,           /* in : type of clause */
    char *errbuf              /* in/out: buffer where we write error messages */
    )
{
    int pcnt;
    char *p;

    meas->m_vec = NULL;
    meas->m_vec2 = NULL;
    meas->m_cross = -1;
    meas->m_fall = -1;
    meas->m_rise = -1;
    meas->m_td = 0;
    meas->m_from = 0.0e0;
    meas->m_to = 0.0e0;
    meas->m_at = 1e99;

    /* for DC, set new outer limits for 'from' and 'to'
       because 0.0e0 may be valid inside of range */
    if (cieq("dc", meas->m_analysis)) {
        meas->m_to = 1.0e99;
        meas->m_from = -1.0e99;
    }

    pcnt = 0;

    while (words != wlTarg) {
        p = words->wl_word;

        if ((pcnt == 0) && !ciprefix("at", p)) {
            meas->m_vec = cp_unquote(words->wl_word);
            /* correct for vectors like vm, vp etc. */
            if (cieq("ac", meas->m_analysis) || cieq("sp", meas->m_analysis))
                correct_vec(meas);
        } else if (ciprefix("at", p)) {
            if (measure_parse_stdParams(meas, words, wlTarg, errbuf) ==
                    MEASUREMENT_FAILURE)
                return MEASUREMENT_FAILURE;
        } else {
            if (measure_parse_stdParams(meas, words, wlTarg, errbuf) ==
                    MEASUREMENT_FAILURE)
                return MEASUREMENT_FAILURE;
            break;
        }

        words = words->wl_next;
        pcnt ++;
    }

    if (pcnt == 0) {
        snprintf(errbuf, MEAS_ERRBUF_SIZE, "bad syntax of '%s'\n", trigTarg);
        return MEASUREMENT_FAILURE;
    }

    // valid vector
    if (measure_valid_vector(meas->m_vec) == 0) {
        snprintf(errbuf, MEAS_ERRBUF_SIZE, "no such vector as '%s'\n", meas->m_vec);
        return MEASUREMENT_FAILURE;
    }

    return MEASUREMENT_OK;
}


/* -----------------------------------------------------------------
 * Function: Given a wordlist, extract the measurement statement,
 * process it, and return a result.  If out_line is furnished, we
 * format and copy the result it this string buffer.  The autocheck
 * variable allows us to check for "autostop".  This function is
 * called from measure.c.    We use the functions in this file because
 * the parsing is much more complete and thorough.
 * ----------------------------------------------------------------- */
int
get_measure2(
    wordlist *wl,     /* in: a word list for us to process */
    double *result,   /* out : the result of the measurement */
    char *out_line,   /* out: formatted result - may be NULL */
    size_t max_out_line, /* in: size of out_line's buffer (0 if out_line is NULL) */
    bool autocheck    /* in: TRUE if checking for "autostop"; FALSE otherwise */
    )
{
    wordlist *words, *wlTarg, *wlWhen;
    char errbuf[MEAS_ERRBUF_SIZE];
    char *mAnalysis = NULL;     // analysis type
    char *mName = NULL;         // name given to the measured output
    char *mFunction = NULL;
    int precision;              // measurement precision
    ANALYSIS_TYPE_T mFunctionType = AT_UNKNOWN;
    int wl_cnt;
    char *p;
    int ret_val = MEASUREMENT_FAILURE;
    FILE *mout = cp_out;

    *result = 0.0e0;        /* default result */

    if (!wl) {
        fprintf(cp_err, "Error: no assignment found in command meas.\n");
        return MEASUREMENT_FAILURE;
    }

    if (!plot_cur || !plot_cur->pl_dvecs || !plot_cur->pl_scale) {
        fprintf(cp_err, "Error: no vectors available\n");
        return MEASUREMENT_FAILURE;
    }

    if (!ciprefix("tran", plot_cur->pl_typename) &&
        !ciprefix("ac", plot_cur->pl_typename) &&
        !ciprefix("dc", plot_cur->pl_typename) &&
        !ciprefix("sp", plot_cur->pl_typename))
    {
        fprintf(cp_err, "Error: measure limited to tran, dc, sp, or ac analysis\n");
        return MEASUREMENT_FAILURE;
    }

    words = wl;
    wlTarg = NULL;
    wlWhen = NULL;

    precision = measure_get_precision();
    wl_cnt = 0;
    while (words) {

        switch (wl_cnt)
        {
        case 0:
            mAnalysis = cp_unquote(words->wl_word);
            break;
        case 1:
            mName = cp_unquote(words->wl_word);
            break;
        case 2:
        {
            mFunctionType = measure_function_type(words->wl_word);
            if (mFunctionType == AT_UNKNOWN) {
                if (!autocheck) {
                    printf("\tmeasure '%s'  failed\n", mName);
                    printf("Error: measure  %s  :\n", mName);
                    printf("\tno such function as '%s'\n", words->wl_word);
                }
                tfree(mName);
                tfree(mAnalysis);
                return MEASUREMENT_FAILURE;
            }
            mFunction = copy(words->wl_word);
            break;
        }
        default:
        {
            p = words->wl_word;

            if (strcasecmp(p, "targ") == 0)
                wlTarg = words;

            if (strcasecmp(p, "when") == 0)
                wlWhen = words;

            break;
        }
        }
        wl_cnt ++;
        words = words->wl_next;
    }

    if (wl_cnt < 3) {
        fprintf(stderr, "\tmeasure '%s'  failed\n", mName);
        fprintf(stderr, "Error: measure  %s  :\n", mName);
        fprintf(stderr, "\tinvalid num params\n");
        tfree(mName);
        tfree(mAnalysis);
        tfree(mFunction);
        return MEASUREMENT_FAILURE;
    }

    //------------------------


    words = wl;

    if (words)
        words = words->wl_next; // skip
    if (words)
        words = words->wl_next; // skip results name
    if (words)
        words = words->wl_next; // Function

    // switch here
    switch (mFunctionType)
    {
    case AT_DELAY:
    case AT_TRIG:
    {
        // trig parameters
        MEASUREPTR measTrig, measTarg;
        measTrig = TMALLOC(struct measure, 1);
        measTarg = TMALLOC(struct measure, 1);

        measTrig->m_analysis = measTarg->m_analysis = mAnalysis;

        if (measure_parse_trigtarg(measTrig, words, wlTarg, "trig", errbuf) ==
                MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret1;
        }

        if ((measTrig->m_rise == -1) && (measTrig->m_fall == -1) &&
            (measTrig->m_cross == -1) && (measTrig->m_at == 1e99)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "at, rise, fall or cross must be given\n");
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret1;
        }

        while (words != wlTarg)
            words = words->wl_next; // hack

        if (words)
            words = words->wl_next; // skip targ

        if (measure_parse_trigtarg(measTarg, words, NULL, "targ", errbuf) ==
                MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "TARG", errbuf, autocheck);
            goto err_ret1;
        }

        if ((measTarg->m_rise == -1) && (measTarg->m_fall == -1) &&
            (measTarg->m_cross == -1)&& (measTarg->m_at == 1e99)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "at, rise, fall or cross must be given\n");
            measure_errMessage(mName, mFunction, "TARG", errbuf, autocheck);
            goto err_ret1;
        }

        // If there was a FROM propagate trig<->targ

        if (measTrig->m_from !=0.0 && measTarg->m_from == 0.0)
            measTarg->m_from = measTrig->m_from;
        else if (measTarg->m_from !=0.0 && measTrig->m_from == 0.0)
            measTrig->m_from = measTarg->m_from;

        // measure trig
        if (measTrig->m_at == 1e99)
            com_measure_when(measTrig);
        else
            measTrig->m_measured = measTrig->m_at;


        if (isnan(measTrig->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret1;
        }
        // measure targ
        if (measTarg->m_at == 1e99)
            com_measure_when(measTarg);
        else
            measTarg->m_measured = measTarg->m_at;

        if (isnan(measTarg->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "TARG", errbuf, autocheck);
            goto err_ret1;
        }

        // print results
        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=  %.*e targ=  %.*e trig=  %.*e\n",
                mName, precision, (measTarg->m_measured - measTrig->m_measured),
                precision, measTarg->m_measured, precision, measTrig->m_measured);
        else
            fprintf(mout,"%-20s=  %.*e targ=  %.*e trig=  %.*e\n", mName, precision,
                (measTarg->m_measured - measTrig->m_measured),
                precision, measTarg->m_measured, precision, measTrig->m_measured);

        *result = (measTarg->m_measured - measTrig->m_measured);

        ret_val = MEASUREMENT_OK;

err_ret1:
        tfree(mAnalysis);
        tfree(mName);
        tfree(measTarg->m_vec);
        tfree(measTarg);
        tfree(measTrig->m_vec);
        tfree(measTrig);
        tfree(mFunction);

        return ret_val;
    }
    case AT_FIND:
    {
        MEASUREPTR meas, measFind;
        meas = TMALLOC(struct measure, 1);
        measFind = TMALLOC(struct measure, 1);

        meas->m_analysis = measFind->m_analysis = mAnalysis;

        if (measure_parse_find(meas, words, wlWhen, errbuf) == MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "FIND", errbuf, autocheck);
            goto err_ret2;
        }

        if (meas->m_at == 1e99) {
            // find .. when statment
            while (words != wlWhen)
                words = words->wl_next; // hack

            if (words)
                words = words->wl_next; // skip targ

            if (measure_parse_when(measFind, words, errbuf) == MEASUREMENT_FAILURE) {
                measure_errMessage(mName, mFunction, "WHEN", errbuf, autocheck);
                goto err_ret2;
            }

            com_measure_when(measFind);

            if (isnan(measFind->m_measured)) {
                snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
                measure_errMessage(mName, mFunction, "AT", errbuf, autocheck);
                goto err_ret2;
            }

            if(measure_at(meas, measFind->m_measured) == MEASUREMENT_FAILURE){
                goto err_ret2;
            }

            meas->m_at = measFind->m_measured;

        } else {
            if (measure_at(meas, meas->m_at) == MEASUREMENT_FAILURE) {
                goto err_ret2;
            }
        }

        if (isnan(meas->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "AT", errbuf, autocheck);
            goto err_ret2;
        }

        // print results
        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=  %.*e\n", mName, precision, meas->m_measured);
        else
            fprintf(mout,"%-20s=  %.*e\n", mName, precision, meas->m_measured);

        *result = meas->m_measured;

        ret_val = MEASUREMENT_OK;

err_ret2:
        tfree(mAnalysis);
        tfree(mName);
        tfree(meas->m_vec);
        tfree(meas);
        tfree(measFind->m_vec);
        tfree(measFind);
        tfree(mFunction);

        return ret_val;
    }
    case AT_WHEN:
    {
        MEASUREPTR meas;
        meas = TMALLOC(struct measure, 1);
        meas->m_analysis = mAnalysis;
        if (measure_parse_when(meas, words, errbuf) == MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "WHEN", errbuf, autocheck);
            goto err_ret3;
        }

        com_measure_when(meas);

        if (isnan(meas->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "WHEN", errbuf, autocheck);
            goto err_ret3;
        }

        // print results
        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=   %.*e\n", mName, precision, meas->m_measured);
        else
            fprintf(mout, "%-20s=  %.*e\n", mName, precision, meas->m_measured);

        *result = meas->m_measured;

        ret_val = MEASUREMENT_OK;

err_ret3:
        tfree(mAnalysis);
        tfree(mName);
        tfree(meas->m_vec);
        tfree(meas->m_vec2);
        tfree(meas);
        tfree(mFunction);

        return ret_val;
    }
    case AT_RMS:
    case AT_INTEG:
    {
        // trig parameters
        MEASUREPTR meas;
        meas = TMALLOC(struct measure, 1);
        meas->m_analysis = mAnalysis;
        if (measure_parse_trigtarg(meas, words, NULL, "trig", errbuf) ==
                MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret4;
        }

        // measure
        measure_rms_integral(meas, mFunctionType);

        if (isnan(meas->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck); // ??
            goto err_ret4;
        }

        if (meas->m_at == 1e99)
            meas->m_at = 0.0e0;

        // print results
        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=   %.*e from=  %.*e to=  %.*e\n", mName, precision, meas->m_measured, precision, meas->m_from, precision, meas->m_to);
        else
            fprintf(mout, "%-20s=  %.*e from=  %.*e to=  %.*e\n", mName, precision, meas->m_measured, precision, meas->m_from, precision, meas->m_to);

        *result = meas->m_measured;

        ret_val = MEASUREMENT_OK;

err_ret4:
        tfree(mAnalysis);
        tfree(mName);
        tfree(meas->m_vec);
        tfree(meas);
        tfree(mFunction);

        return ret_val;

    }
    case AT_AVG:
    {
        // trig parameters
        MEASUREPTR meas;
        meas = TMALLOC(struct measure, 1);

        meas->m_analysis = mAnalysis;

        if (measure_parse_trigtarg(meas, words, NULL, "trig", errbuf) ==
                MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret5;
        }

        // measure
        measure_minMaxAvg(meas, mFunctionType);
        if (isnan(meas->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck); // ??
            goto err_ret5;
        }

        if (meas->m_at == 1e99)
            meas->m_at = meas->m_from;

        // print results
        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=  %.*e from=  %.*e to=  %.*e\n", mName,
                precision, meas->m_measured, precision, meas->m_at, precision, meas->m_measured_at);
        else
            fprintf(mout, "%-20s=  %.*e from=  %.*e to=  %.*e\n", mName,
                precision, meas->m_measured, precision, meas->m_at, precision, meas->m_measured_at);

        *result = meas->m_measured;

        ret_val = MEASUREMENT_OK;

err_ret5:
        tfree(mAnalysis);
        tfree(mName);
        tfree(meas->m_vec);
        tfree(meas);
        tfree(mFunction);

        return ret_val;
    }
    case AT_PHASE_MARGIN:
    case AT_GAIN_MARGIN:
    {
        MEASUREPTR meas;
        const char *unit = (mFunctionType == AT_PHASE_MARGIN) ? "deg" : "dB";
        meas = TMALLOC(struct measure, 1);
        meas->m_analysis = mAnalysis;

        if (measure_parse_trigtarg(meas, words, NULL, "trig", errbuf) ==
                MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret_margin;
        }

        measure_margin(meas, mFunctionType);
        if (isnan(meas->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, (mFunctionType == AT_PHASE_MARGIN)
                    ? "no unity-gain (0 dB) crossover found\n"
                    : "no -180 deg phase crossover found\n");
            measure_errMessage(mName, mFunction, "MARGIN", errbuf, autocheck);
            goto err_ret_margin;
        }

        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=  %.*e %s at=  %.*e\n", mName,
                precision, meas->m_measured, unit, precision, meas->m_measured_at);
        else
            fprintf(mout, "%-20s=  %.*e %s at=  %.*e\n", mName,
                precision, meas->m_measured, unit, precision, meas->m_measured_at);

        *result = meas->m_measured;
        ret_val = MEASUREMENT_OK;

err_ret_margin:
        tfree(mAnalysis);
        tfree(mName);
        tfree(meas->m_vec);
        tfree(meas);
        tfree(mFunction);

        return ret_val;
    }
    case AT_MIN:
    case AT_MAX:
    case AT_MIN_AT:
    case AT_MAX_AT:
    {
        // trig parameters
        MEASUREPTR measTrig;
        measTrig = TMALLOC(struct measure, 1);
        measTrig->m_analysis = mAnalysis;
        if (measure_parse_trigtarg(measTrig, words, NULL, "trig", errbuf) ==
                MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret6;
        }

        // measure
        if ((mFunctionType == AT_MIN) || (mFunctionType == AT_MIN_AT))
            measure_minMaxAvg(measTrig, AT_MIN);
        else
            measure_minMaxAvg(measTrig, AT_MAX);

        if (isnan(measTrig->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck); // ??
            goto err_ret6;
        }

        if ((mFunctionType == AT_MIN) || (mFunctionType == AT_MAX)) {
            // print results
            if (out_line)
                snprintf(out_line, max_out_line, "%-20s=  %.*e at=  %.*e\n",
                    mName, precision, measTrig->m_measured, precision, measTrig->m_measured_at);
            else
                fprintf(mout, "%-20s=  %.*e at=  %.*e\n", 
                    mName, precision, measTrig->m_measured, precision, measTrig->m_measured_at);

            *result = measTrig->m_measured;
        } else {
            // print results
            if (out_line)
                snprintf(out_line, max_out_line, "%-20s=  %.*e with=  %.*e\n",
                    mName, precision, measTrig->m_measured_at, precision, measTrig->m_measured);
            else
                fprintf(mout, "%-20s=  %.*e with=  %.*e\n", 
                    mName, precision, measTrig->m_measured_at, precision, measTrig->m_measured);

            *result = measTrig->m_measured_at;
        }

        ret_val = MEASUREMENT_OK;

err_ret6:
        tfree(mAnalysis);
        tfree(mName);
        tfree(measTrig->m_vec);
        tfree(measTrig);
        tfree(mFunction);

        return ret_val;
    }
    case AT_PP:
    {
        double minValue, maxValue;
        MEASUREPTR measTrig;
        measTrig = TMALLOC(struct measure, 1);
        measTrig->m_analysis = mAnalysis;
        if (measure_parse_trigtarg(measTrig, words, NULL, "trig", errbuf) ==
                MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck);
            goto err_ret7;
        }

        // measure min
        measure_minMaxAvg(measTrig, AT_MIN);
        if (isnan(measTrig->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck); // ??
            goto err_ret7;
        }
        minValue = measTrig->m_measured;

        // measure max
        measure_minMaxAvg(measTrig, AT_MAX);
        if (isnan(measTrig->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "TRIG", errbuf, autocheck); // ??
            goto err_ret7;
        }
        maxValue = measTrig->m_measured;

        // print results
        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=  %.*e from=  %.*e to=  %.*e\n",
                mName, precision, (maxValue - minValue), precision, measTrig->m_from, precision, measTrig->m_to);
        else
            fprintf(mout, "%-20s=  %.*e from=  %.*e to=  %.*e\n",
                mName, precision, (maxValue - minValue), precision, measTrig->m_from, precision, measTrig->m_to);

        *result = (maxValue - minValue);

        ret_val = MEASUREMENT_OK;

err_ret7:
        tfree(mAnalysis);
        tfree(mName);
        tfree(measTrig->m_vec);
        tfree(measTrig);
        tfree(mFunction);

        return ret_val;
    }

    case AT_DERIV:
    {
        /* Enhancement-179: DERIV{ATIVE} out_var AT=val | WHEN ... -- was
         * parsed but unimplemented ('currently not supported'). Same
         * parse/locate flow as FIND, with the derivative evaluator. */
        MEASUREPTR meas, measFind;
        meas = TMALLOC(struct measure, 1);
        measFind = TMALLOC(struct measure, 1);

        meas->m_analysis = measFind->m_analysis = mAnalysis;

        if (measure_parse_find(meas, words, wlWhen, errbuf) == MEASUREMENT_FAILURE) {
            measure_errMessage(mName, mFunction, "DERIV", errbuf, autocheck);
            goto err_ret_deriv;
        }

        if (meas->m_at == 1e99) {
            /* deriv .. when statement */
            while (words != wlWhen)
                words = words->wl_next;

            if (words)
                words = words->wl_next;

            if (measure_parse_when(measFind, words, errbuf) == MEASUREMENT_FAILURE) {
                measure_errMessage(mName, mFunction, "WHEN", errbuf, autocheck);
                goto err_ret_deriv;
            }

            com_measure_when(measFind);

            if (isnan(measFind->m_measured)) {
                snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
                measure_errMessage(mName, mFunction, "WHEN", errbuf, autocheck);
                goto err_ret_deriv;
            }

            if (measure_deriv_at(meas, measFind->m_measured) == MEASUREMENT_FAILURE)
                goto err_ret_deriv;

            meas->m_at = measFind->m_measured;

        } else {
            if (measure_deriv_at(meas, meas->m_at) == MEASUREMENT_FAILURE)
                goto err_ret_deriv;
        }

        if (isnan(meas->m_measured)) {
            snprintf(errbuf, MEAS_ERRBUF_SIZE, "out of interval\n");
            measure_errMessage(mName, mFunction, "AT", errbuf, autocheck);
            goto err_ret_deriv;
        }

        if (out_line)
            snprintf(out_line, max_out_line, "%-20s=  %.*e\n", mName, precision, meas->m_measured);
        else
            fprintf(mout, "%-20s=  %.*e\n", mName, precision, meas->m_measured);

        *result = meas->m_measured;

        ret_val = MEASUREMENT_OK;

err_ret_deriv:
        tfree(mAnalysis);
        tfree(mName);
        tfree(meas->m_vec);
        tfree(meas);
        tfree(measFind->m_vec);
        tfree(measFind);
        tfree(mFunction);

        return ret_val;
    }

    case AT_ERR:
    case AT_ERR1:
    case AT_ERR2:
    case AT_ERR3:
    {
        fprintf(stderr, "\nError: measure  %s failed:\n", mName);
        fprintf(stderr, "\tfunction '%s' currently not supported\n\n", mFunction);
        tfree(mFunction);
        break;
    }

    default:
    {
        fprintf(stderr, "ERROR: enumeration value `AT_UNKNOWN' not handled in get_measure2\nAborting...\n");
        controlled_exit(EXIT_FAILURE);
    }
    }

    return MEASUREMENT_FAILURE;
}
