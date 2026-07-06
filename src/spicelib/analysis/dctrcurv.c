/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 1999 Paolo Nenzi
**********/

#include "ngspice/ngspice.h"

#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#include "res/resdefs.h"

#include "ngspice/cktdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"
#include "ngspice/compatmode.h"

#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
#endif

#include "ngspice/devdefs.h"

#ifdef HAS_PROGREP
static double actval, actdiff;
#endif

/* Sweeping a .param: we update the numparam dictionary AND any V/I
 * source's DC value that was bound to this param at parse time.
 * Bindings come from inpdpar.c's dpar_register_binding() table; we
 * read them via the opaque accessors below.  nupa_get_real /
 * nupa_set_real live in frontend/numparam/spicenum.c. */
extern int nupa_get_real(const char *name, double *value);
extern int nupa_set_real(const char *name, double value);

typedef struct dpar_param_binding dpar_param_binding_t;
extern dpar_param_binding_t *dpar_first_param_binding(void);
extern const char *dpar_binding_param_name(const dpar_param_binding_t *b);
extern const char *dpar_binding_dev_name(const dpar_param_binding_t *b);
extern int dpar_binding_dev_type(const dpar_param_binding_t *b);
extern const dpar_param_binding_t *dpar_binding_next(const dpar_param_binding_t *b);

/* Push `value` into the DC field of every V/I source bound to the
 * named .param.  Lookup matches on the (param_name, dev_type) pair
 * recorded by dpar_register_binding at INPdevParse time, then finds
 * the actual instance by name in CKThead[dev_type].  Mirrors how
 * the existing source-name sweep path mutates VSRCdcValue/
 * ISRCdcValue directly. */
static void dctrcurv_push_param_to_bindings(
    CKTcircuit *ckt, const char *param_name, double value,
    int vcode, int icode)
{
    const dpar_param_binding_t *b;
    for (b = dpar_first_param_binding(); b; b = dpar_binding_next(b)) {
        if (strcmp(dpar_binding_param_name(b), param_name) != 0) continue;
        int dt = dpar_binding_dev_type(b);
        const char *devn = dpar_binding_dev_name(b);
        if (dt == vcode && vcode >= 0) {
            VSRCmodel *m;
            VSRCinstance *here;
            for (m = (VSRCmodel *)ckt->CKThead[vcode]; m; m = VSRCnextModel(m))
                for (here = VSRCinstances(m); here; here = VSRCnextInstance(here)) {
                    if (here->VSRCname && strcmp(here->VSRCname, devn) == 0) {
                        here->VSRCdcValue = value;
                        here->VSRCdcGiven = 1;
                        goto next_binding;
                    }
                }
        } else if (dt == icode && icode >= 0) {
            ISRCmodel *m;
            ISRCinstance *here;
            for (m = (ISRCmodel *)ckt->CKThead[icode]; m; m = ISRCnextModel(m))
                for (here = ISRCinstances(m); here; here = ISRCnextInstance(here))
                    if (here->ISRCname && strcmp(here->ISRCname, devn) == 0) {
                        here->ISRCdcValue = value;
                        here->ISRCdcGiven = 1;
                        goto next_binding;
                    }
        }
    next_binding:;
    }
}

/* Forward decl for the parse-time binding registry — defined in
 * src/spicelib/parser/inpdpar.c. */
extern void dpar_register_binding(const char *param_name,
                                  const char *dev_name, int dev_type);

/* Scan the original (un-substituted) deck for device lines that
 * mention `param_name` as a bare identifier in the value position
 * and register a binding for each match.  Numparam pre-substitutes
 * bare-identifier values in V/I/R device lines BEFORE INPdevParse
 * is called, so the parse-time bind path (inpeval.c +
 * inpdpar.c::INPdevParse) can't capture them — by the time the
 * parser sees the line, the original `.param` name has been
 * replaced with its numeric value.  This deck-text scan recovers
 * the binding at .dc setup time using the saved actualLine text
 * via `ft_curckt->ci_origdeck`.
 *
 * Match rules: the value field is the LAST whitespace-delimited
 * token before the end of the (trimmed) line.  We accept it as a
 * binding when it equals `param_name` exactly (case-insensitive)
 * and the device name's first letter is v/V/i/I (voltage or
 * current source).  No expression context — `.dc paramName` only
 * propagates to sources whose value field is literally
 * `paramName`, not to `2*paramName` or `paramName+0.1`. */
static void dctrcurv_scan_deck_for_bindings(const char *param_name,
                                            int vcode, int icode)
{
    if (!ft_curckt) return;
    struct card *c;
    for (c = ft_curckt->ci_origdeck ? ft_curckt->ci_origdeck : ft_curckt->ci_deck;
         c; c = c->nextcard) {
        const char *line = c->line;
        if (!line || !*line) continue;
        if (*line == '*' || *line == '.' || *line == '+') continue;
        /* Classify by first letter — only V/I sources are bindable. */
        int dev_type = -1;
        if (*line == 'v' || *line == 'V')      dev_type = vcode;
        else if (*line == 'i' || *line == 'I') dev_type = icode;
        else continue;
        if (dev_type < 0) continue;
        /* Extract dev name (first token, lowercased to match
         * GENname's canonical form). */
        const char *name_end = line;
        while (*name_end && *name_end != ' ' && *name_end != '\t')
            name_end++;
        size_t name_len = (size_t)(name_end - line);
        if (name_len == 0 || name_len > 63) continue;
        char dev_name_buf[64];
        for (size_t i = 0; i < name_len; i++) {
            char ch = line[i];
            if (ch >= 'A' && ch <= 'Z') ch = (char)(ch - 'A' + 'a');
            dev_name_buf[i] = ch;
        }
        dev_name_buf[name_len] = '\0';
        /* Trim trailing whitespace and grab the last token of the
         * line — that's the value field (ngspice / HSPICE put the
         * source value at end-of-line for simple DC sources). */
        const char *end = line + strlen(line);
        while (end > line && (end[-1] == ' ' || end[-1] == '\t' ||
                              end[-1] == '\n' || end[-1] == '\r'))
            end--;
        const char *last = end;
        while (last > line && last[-1] != ' ' && last[-1] != '\t')
            last--;
        /* Strip `{...}` braces if present — inpcom.c wraps bare
         * param refs in V/I source values into `{name}` for
         * numparam-driven substitution.  We want the inner name. */
        if (last < end && *last == '{' && end[-1] == '}') {
            last++;
            end--;
        }
        size_t tok_len = (size_t)(end - last);
        size_t plen = strlen(param_name);
        bool match = (tok_len == plen);
        if (match) {
            for (size_t i = 0; i < plen; i++) {
                char a = last[i], b = param_name[i];
                if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
                if (b >= 'A' && b <= 'Z') b = (char)(b - 'A' + 'a');
                if (a != b) { match = false; break; }
            }
        }
        if (!match) continue;
        dpar_register_binding(param_name, dev_name_buf, dev_type);
    }
}


int
DCtrCurv(CKTcircuit *ckt, int restart)
{
    TRCV *job = (TRCV *) ckt->CKTcurJob;

    int i;
    double *temp;
    int converged;
    int rcode;
    int vcode;
    int icode;
    int j;
    int error;
    IFuid varUid;
    IFuid *nameList;
    int numNames;
    int firstTime = 1;
    static runDesc *plot = NULL;

#ifdef WANT_SENSE2
    long save;
#ifdef SENSDEBUG
    if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & DCSEN)) {
        printf("\nDC Sensitivity Results\n\n");
        CKTsenPrint(ckt);
    }
#endif
#endif

    rcode = CKTtypelook("Resistor");
    vcode = CKTtypelook("Vsource");
    icode = CKTtypelook("Isource");

    if (!restart && job->TRCVnestState >= 0) {
        /* continuing */
        i = job->TRCVnestState;
        /* resume to work? saj*/
        error = SPfrontEnd->OUTpBeginPlot (NULL, NULL,
                                           NULL,
                                           NULL, 0,
                                           666, NULL, 666,
                                           &plot);
        goto resume;
    }

    ckt->CKTtime = 0;
    ckt->CKTdelta = job->TRCVvStep[0];
    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
    ckt->CKTorder = 1;

    /* Save the state of the circuit */
    for (j = 0; j < 7; j++)
        ckt->CKTdeltaOld[j] = ckt->CKTdelta;

    for (i = 0; i <= job->TRCVnestLevel; i++) {

        if (rcode >= 0) {
            /* resistances are in this version, so use them */
            RESinstance *here;
            RESmodel *model;

            for (model = (RESmodel *)ckt->CKThead[rcode]; model; model = RESnextModel(model))
                for (here = RESinstances(model); here; here = RESnextInstance(here))
                    if (here->RESname == job->TRCVvName[i]) {
                        job->TRCVvElt[i]  = (GENinstance *)here;
                        job->TRCVvSave[i] = here->RESresist;
                        job->TRCVgSave[i] = here->RESresGiven;
                        job->TRCVvType[i] = rcode;
                        here->RESresist   = job->TRCVvStart[i];
                        here->RESresGiven = 1;
                        CKTtemp(ckt);
                        goto found;
                    }
        }

        if (vcode >= 0) {
            /* voltage sources are in this version, so use them */
            VSRCinstance *here;
            VSRCmodel *model;

            for (model = (VSRCmodel *)ckt->CKThead[vcode]; model; model = VSRCnextModel(model))
                for (here = VSRCinstances(model); here; here = VSRCnextInstance(here))
                    if (here->VSRCname == job->TRCVvName[i]) {
                        job->TRCVvElt[i]  = (GENinstance *)here;
                        job->TRCVvSave[i] = here->VSRCdcValue;
                        job->TRCVgSave[i] = here->VSRCdcGiven;
                        job->TRCVvType[i] = vcode;
                        here->VSRCdcValue = job->TRCVvStart[i];
                        here->VSRCdcGiven = 1;
                        goto found;
                    }
        }

        if (icode >= 0) {
            /* current sources are in this version, so use them */
            ISRCinstance *here;
            ISRCmodel *model;

            for (model = (ISRCmodel *)ckt->CKThead[icode]; model; model = ISRCnextModel(model))
                for (here = ISRCinstances(model); here; here = ISRCnextInstance(here))
                    if (here->ISRCname == job->TRCVvName[i]) {
                        job->TRCVvElt[i]  = (GENinstance *)here;
                        job->TRCVvSave[i] = here->ISRCdcValue;
                        job->TRCVgSave[i] = here->ISRCdcGiven;
                        job->TRCVvType[i] = icode;
                        here->ISRCdcValue = job->TRCVvStart[i];
                        here->ISRCdcGiven = 1;
                        goto found;
                    }
        }

        if (cieq(job->TRCVvName[i], "temp")) {
            job->TRCVvSave[i] = ckt->CKTtemp; /* Saves the old circuit temperature */
            job->TRCVvType[i] = TEMP_CODE;    /* Set the sweep type code */
            ckt->CKTtemp = job->TRCVvStart[i] + CONSTCtoK; /* Set the new circuit temp */
            inp_evaluate_temper(ft_curckt);
            CKTtemp(ckt);
            goto found;
        }

        /* HSPICE-compat: sweep a .param.  If the name matches a
         * NUPA_REAL entry in the numparam global dictionary, push
         * the start value into the dictionary AND into any V/I
         * source whose DC field was bound to this .param.
         *
         * Bindings are recovered by scanning the original (un-
         * substituted) deck text in `ft_curckt->ci_origdeck` —
         * numparam pre-substitutes bare-identifier values in V/I/R
         * device lines BEFORE INPdevParse runs, so the parse-time
         * registry from inpdpar.c can't capture them. */
        {
            double cur_val;
            if (nupa_get_real(job->TRCVvName[i], &cur_val)) {
                dctrcurv_scan_deck_for_bindings(
                    job->TRCVvName[i], vcode, icode);
                job->TRCVvSave[i] = cur_val;
                job->TRCVvType[i] = PARAM_CODE;
                job->TRCVvElt[i]  = NULL;
                job->TRCVstepCount[i] = 0;
                nupa_set_real(job->TRCVvName[i], job->TRCVvStart[i]);
                dctrcurv_push_param_to_bindings(
                    ckt, job->TRCVvName[i], job->TRCVvStart[i],
                    vcode, icode);
                goto found;
            }
        }

        SPfrontEnd->IFerrorf (ERR_FATAL,
                "DC Transfer Function: Voltage source, current source, or "
                "resistor named \"%s\" is not in the circuit",
                job->TRCVvName[i]);
        return(E_NODEV);

    found:;
    }

#ifdef HAS_PROGREP
    actval = job->TRCVvStart[job->TRCVnestLevel];
    actdiff = job->TRCVvStart[job->TRCVnestLevel] - job->TRCVvStop[job->TRCVnestLevel];
#endif

#ifdef XSPICE

    /* Tell the code models what mode we're in */
    g_mif_info.circuit.anal_type = MIF_DC;

    g_mif_info.circuit.anal_init = MIF_TRUE;

#endif

    error = CKTnames(ckt, &numNames, &nameList);
    if (error)
        return(error);

    if (job->TRCVvType[0] == vcode)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "v-sweep", UID_OTHER, NULL);
    else if (job->TRCVvType[0] == icode)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "i-sweep", UID_OTHER, NULL);
    else if (job->TRCVvType[0] == TEMP_CODE)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "temp-sweep", UID_OTHER, NULL);
    else if (job->TRCVvType[0] == rcode)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "res-sweep", UID_OTHER, NULL);
    else if (job->TRCVvType[0] == PARAM_CODE)
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "param-sweep", UID_OTHER, NULL);
    else
        SPfrontEnd->IFnewUid (ckt, &varUid, NULL, "?-sweep", UID_OTHER, NULL);

    error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                       ckt->CKTcurJob->JOBname,
                                       varUid, IF_REAL,
                                       numNames, nameList, IF_REAL,
                                       &plot);
    tfree(nameList);

    if (error)
        return(error);

    /* initialize CKTsoaCheck `warn' counters */
    if (ckt->CKTsoaCheck)
        error = CKTsoaInit();

    /* now have finished the initialization - can start doing hard part */

    i = 0;

 resume:

    for (;;) {

        if (job->TRCVvType[i] == vcode) { /* voltage source */
            if (SGN(job->TRCVvStep[i]) *
                (((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcValue -
                 job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        } else if (job->TRCVvType[i] == icode) { /* current source */
            if (SGN(job->TRCVvStep[i]) *
                (((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcValue -
                 job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        } else if (job->TRCVvType[i] == rcode) { /* resistance */
            if (SGN(job->TRCVvStep[i]) *
                (((RESinstance*)(job->TRCVvElt[i]))->RESresist -
                 job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        } else if (job->TRCVvType[i] == TEMP_CODE) { /* temp sweep */
            if (SGN(job->TRCVvStep[i]) *
                ((ckt->CKTtemp - CONSTCtoK) - job->TRCVvStop[i]) >
                DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        } else if (job->TRCVvType[i] == PARAM_CODE) { /* param sweep */
            double cur_val = 0.0;
            nupa_get_real(job->TRCVvName[i], &cur_val);
            if (SGN(job->TRCVvStep[i]) *
                (cur_val - job->TRCVvStop[i]) > DBL_EPSILON * 1e+03)
            {
                i++;
                firstTime = 1;
                ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT;
                if (i > job->TRCVnestLevel)
                    break;
                goto nextstep;
            }
        }

        while (--i >= 0)
            if (job->TRCVvType[i] == vcode) { /* voltage source */
                ((VSRCinstance *)(job->TRCVvElt[i]))->VSRCdcValue =
                    job->TRCVvStart[i];
            } else if (job->TRCVvType[i] == icode) { /* current source */
                ((ISRCinstance *)(job->TRCVvElt[i]))->ISRCdcValue =
                    job->TRCVvStart[i];
            } else if (job->TRCVvType[i] == TEMP_CODE) {
                ckt->CKTtemp = job->TRCVvStart[i] + CONSTCtoK;
                inp_evaluate_temper(ft_curckt);
                CKTtemp(ckt);
            } else if (job->TRCVvType[i] == rcode) {
                ((RESinstance *)(job->TRCVvElt[i]))->RESresist =
                    job->TRCVvStart[i];
                RESupdate_conduct((RESinstance *)(job->TRCVvElt[i]), FALSE);
                DEVices[rcode]->DEVload(job->TRCVvElt[i]->GENmodPtr, ckt);
            } else if (job->TRCVvType[i] == PARAM_CODE) {
                job->TRCVstepCount[i] = 0;
                nupa_set_real(job->TRCVvName[i], job->TRCVvStart[i]);
                dctrcurv_push_param_to_bindings(
                    ckt, job->TRCVvName[i], job->TRCVvStart[i],
                    vcode, icode);
            }

        /* Rotate state vectors. */
        temp = ckt->CKTstates[ckt->CKTmaxOrder + 1];
        for (j = ckt->CKTmaxOrder; j >= 0; j--)
            ckt->CKTstates[j + 1] = ckt->CKTstates[j];
        ckt->CKTstate0 = temp;

        /* do operation */
#ifdef XSPICE
/* gtri - begin - wbk - Do EVTop if event instances exist */
        if (ckt->evt->counts.num_insts == 0) {
            /* If no event-driven instances, do what SPICE normally does */
#endif

            if (newcompat.hs) {
                converged = CKTop(ckt,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                                  ckt->CKTdcMaxIter);
                if (converged != 0)
                    return(converged);
            }
            else {
                converged = NIiter(ckt, ckt->CKTdcTrcvMaxIter);
                if (converged != 0) {
                    converged = CKTop(ckt,
                        (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                        (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                        ckt->CKTdcMaxIter);
                    if (converged != 0)
                        return(converged);
                }
            }
#ifdef XSPICE
        }
        else {
            /* else do new algorithm */

            /* first get the current step in the analysis */
            if (job->TRCVvType[0] == vcode) {
                g_mif_info.circuit.evt_step =
                    ((VSRCinstance *)(job->TRCVvElt[0]))->VSRCdcValue;
            } else if (job->TRCVvType[0] == icode) {
                g_mif_info.circuit.evt_step =
                    ((ISRCinstance *)(job->TRCVvElt[0]))->ISRCdcValue;
            } else if (job->TRCVvType[0] == rcode) {
                g_mif_info.circuit.evt_step =
                    ((RESinstance*)(job->TRCVvElt[0]->GENmodPtr))->RESresist;
            } else if (job->TRCVvType[0] == TEMP_CODE) {
                g_mif_info.circuit.evt_step =
                    ckt->CKTtemp - CONSTCtoK;
            }

            /* if first time through, call EVTop immediately and save event results */
            if (firstTime) {
                converged = EVTop(ckt,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                                  (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                                  ckt->CKTdcMaxIter,
                                  MIF_TRUE);
                EVTdump(ckt, IPC_ANAL_DCOP, g_mif_info.circuit.evt_step);
                EVTop_save(ckt, MIF_FALSE, g_mif_info.circuit.evt_step);
                if (converged != 0)
                    return(converged);
            }
            /* else, call NIiter first with mode = MODEINITPRED */
            /* to attempt quick analog solution.  Then call all hybrids and call */
            /* EVTop only if event outputs have changed, or if non-converged */
            else {
                converged = NIiter(ckt, ckt->CKTdcTrcvMaxIter);
                EVTcall_hybrids(ckt);
                if ((converged != 0) || (ckt->evt->queue.output.num_changed != 0)) {
                    converged = EVTop(ckt,
                                      (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITJCT,
                                      (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITFLOAT,
                                      ckt->CKTdcMaxIter,
                                      MIF_FALSE);
                    EVTdump(ckt, IPC_ANAL_DCTRCURVE, g_mif_info.circuit.evt_step);
                    EVTop_save(ckt, MIF_FALSE, g_mif_info.circuit.evt_step);
                    if (converged != 0)
                        return(converged);
                }
            }
        }
/* gtri - end - wbk - Do EVTop if event instances exist */
#endif

        ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCTRANCURVE | MODEINITPRED;
        if (job->TRCVvType[0] == vcode)
            ckt->CKTtime = ((VSRCinstance *)(job->TRCVvElt[0]))->VSRCdcValue;
        else if (job->TRCVvType[0] == icode)
            ckt->CKTtime = ((ISRCinstance *)(job->TRCVvElt[0]))->ISRCdcValue;
        else if (job->TRCVvType[0] == rcode)
            ckt->CKTtime = ((RESinstance *)(job->TRCVvElt[0]))->RESresist;
        else if (job->TRCVvType[0] == TEMP_CODE)
            ckt->CKTtime = ckt->CKTtemp - CONSTCtoK;
        else if (job->TRCVvType[0] == PARAM_CODE) {
            double v = 0.0;
            nupa_get_real(job->TRCVvName[0], &v);
            ckt->CKTtime = v;
        }

#ifdef XSPICE
        /* If first time through, call CKTdump to output Operating Point info */
        if (wantevtdata && firstTime) {
            CKTdump(ckt, 0.0, plot);
        }
#endif

#ifdef WANT_SENSE2
/*
  if (!ckt->CKTsenInfo) printf("sensitivity structure does not exist\n");
*/
        if (ckt->CKTsenInfo && (ckt->CKTsenInfo->SENmode & DCSEN)) {
            int senmode;

#ifdef SENSDEBUG
            if (job->TRCVvType[0] == vcode) { /* voltage source */
                printf("Voltage Source Value : %.5e V\n",
                       ((VSRCinstance*) (job->TRCVvElt[0]))->VSRCdcValue);
            }
            if (job->TRCVvType[0] == icode) { /* current source */
                printf("Current Source Value : %.5e A\n",
                       ((ISRCinstance*)(job->TRCVvElt[0]))->ISRCdcValue);
            }
            if (job->TRCVvType[0] == rcode) { /* resistance */
                printf("Current Resistance Value : %.5e Ohm\n",
                       ((RESinstance*)(job->TRCVvElt[0]->GENmodPtr))->RESresist);
            }
            if (job->TRCVvType[0] == TEMP_CODE) { /* Temperature */
                printf("Current Circuit Temperature : %.5e C\n",
                       ckt->CKTtemp - CONSTCtoK);
            }
#endif

            senmode = ckt->CKTsenInfo->SENmode;
            save = ckt->CKTmode;
            ckt->CKTsenInfo->SENmode = DCSEN;
            error = CKTsenDCtran(ckt);
            if (error)
                return(error);

            ckt->CKTmode = save;
            ckt->CKTsenInfo->SENmode = senmode;
        }
#endif

        CKTdump(ckt,ckt->CKTtime,plot);

        if (ckt->CKTsoaCheck)
            error = CKTsoaCheck(ckt);

        if (firstTime) {
            firstTime = 0;
            if (ckt->CKTstate1 && ckt->CKTstate0) {
                memcpy(ckt->CKTstate1, ckt->CKTstate0,
                       (size_t) ckt->CKTnumStates * sizeof(double));
            }
        }

        i = 0;

    nextstep:;

        if (job->TRCVvType[i] == vcode) { /* voltage source */
            ((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcValue +=
                job->TRCVvStep[i];
        } else if (job->TRCVvType[i] == icode) { /* current source */
            ((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcValue +=
                job->TRCVvStep[i];
        } else if (job->TRCVvType[i] == rcode) { /* resistance */
            ((RESinstance*)(job->TRCVvElt[i]))->RESresist +=
                job->TRCVvStep[i];
            RESupdate_conduct((RESinstance *)(job->TRCVvElt[i]), FALSE);
            DEVices[rcode]->DEVload(job->TRCVvElt[i]->GENmodPtr, ckt);
        } else if (job->TRCVvType[i] == TEMP_CODE) { /* temperature */
            ckt->CKTtemp += job->TRCVvStep[i];

            /* FIXME: Do the Temp check already here for the first time.
               If the stop criterion is fulfilled, discard Temp evaluation, because
               CKTtemp may report errors if a large extra Temp step is exercized. */
            if (SGN(job->TRCVvStep[i]) *
                ((ckt->CKTtemp - CONSTCtoK) - job->TRCVvStop[i]) > DBL_EPSILON * 1e+03) {
//                ckt->CKTtemp -= job->TRCVvStep[i]; // Undo the large step
//                ckt->CKTtemp += SGN(job->TRCVvStep[i]) * DBL_EPSILON * 2e+03; // Add just a small step
                continue; // Skip model evaluation
            }

            inp_evaluate_temper(ft_curckt);
            CKTtemp(ckt);
        } else if (job->TRCVvType[i] == PARAM_CODE) { /* param sweep */
            /* Exact arithmetic: cur_val = start + N*step instead of
             * accumulating += step.  Accumulation drifts by ~N ULP
             * over N iterations and prevents `.measure when X=stop`
             * from finding the endpoint.
             *
             * Even start+N*step isn't bit-exact when start/step
             * aren't representable in binary FP (start=-0.1,
             * step=0.01 → -0.1+340*0.01 ≈ 3.2999999... not 3.3).
             * On the LAST accepted iteration (the one whose NEXT
             * would trigger the stop check), snap cur_val to stop
             * exactly so the saved row's X-value is bit-exact.
             * GF55 bcd55 isoednfet's `.measure when v(n2)=3.3`
             * relies on this. */
            job->TRCVstepCount[i]++;
            double cur_val = job->TRCVvStart[i] +
                job->TRCVstepCount[i] * job->TRCVvStep[i];
            /* Snap to stop ONLY on the iter that lands within one
             * step of stop AND whose successor would overshoot.
             * Without the cur-side guard, every iter past stop
             * keeps getting snapped back to stop and the loop's
             * own stop-check (top of for(;;)) never triggers. */
            double next_val = job->TRCVvStart[i] +
                (job->TRCVstepCount[i] + 1) * job->TRCVvStep[i];
            double sign = SGN(job->TRCVvStep[i]);
            double cur_offset = sign * (cur_val - job->TRCVvStop[i]);
            double next_offset = sign * (next_val - job->TRCVvStop[i]);
            if (next_offset > DBL_EPSILON * 1e+03 &&
                cur_offset <= DBL_EPSILON * 1e+03) {
                /* Snap to stop exactly.  INPevaluate (the .dc stop and
                 * the measure `when X = <param>` RHS) and formula()'s
                 * fetchnumber (which stores the .param's value via
                 * sscanf "%lG") now BOTH parse numbers through strtod, so
                 * they round identically: the snapped TRCVvStop is already
                 * bit-equal to the m_val the measure compares against.
                 * (Previously the two paths differed by 1 ULP and this had
                 * to route the snap through INPevaluate's "% 23.15e"
                 * round-trip to match -- see inpeval.c.) */
                cur_val = job->TRCVvStop[i];
            }
            nupa_set_real(job->TRCVvName[i], cur_val);
            dctrcurv_push_param_to_bindings(
                ckt, job->TRCVvName[i], cur_val, vcode, icode);
        }

        if (SPfrontEnd->IFpauseTest()) {
            /* user asked us to pause, so save state */
            job->TRCVnestState = i;
            return(E_PAUSE);
        }

#ifdef HAS_PROGREP
        if (i == job->TRCVnestLevel) {
            actval += job->TRCVvStep[job->TRCVnestLevel];
            SetAnalyse("dc", abs((int)((actval - job->TRCVvStart[job->TRCVnestLevel]) * 1000. / actdiff)));
        }
#endif

    }

    /* all done, lets put everything back */

    for (i = 0; i <= job->TRCVnestLevel; i++)
        if (job->TRCVvType[i] == vcode) {   /* voltage source */
            ((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcValue = job->TRCVvSave[i];
            ((VSRCinstance*)(job->TRCVvElt[i]))->VSRCdcGiven = (job->TRCVgSave[i] != 0);
        } else  if (job->TRCVvType[i] == icode) { /*current source */
            ((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcValue = job->TRCVvSave[i];
            ((ISRCinstance*)(job->TRCVvElt[i]))->ISRCdcGiven = (job->TRCVgSave[i] != 0);
        } else  if (job->TRCVvType[i] == rcode) { /* Resistance */
            ((RESinstance*)(job->TRCVvElt[i]))->RESresist = job->TRCVvSave[i];
            ((RESinstance*)(job->TRCVvElt[i]))->RESresGiven = (job->TRCVgSave[i] != 0);
            RESupdate_conduct((RESinstance *)(job->TRCVvElt[i]), TRUE);
            DEVices[rcode]->DEVload(job->TRCVvElt[i]->GENmodPtr, ckt);
        } else if (job->TRCVvType[i] == TEMP_CODE) {
            ckt->CKTtemp = job->TRCVvSave[i];
            inp_evaluate_temper(ft_curckt);
            CKTtemp(ckt);
        } else if (job->TRCVvType[i] == PARAM_CODE) {
            nupa_set_real(job->TRCVvName[i], job->TRCVvSave[i]);
            dctrcurv_push_param_to_bindings(
                ckt, job->TRCVvName[i], job->TRCVvSave[i], vcode, icode);
        }

    SPfrontEnd->OUTendPlot (plot);

    return(OK);
}
