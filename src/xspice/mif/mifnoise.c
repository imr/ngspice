/*============================================================================
FILE    MIFnoise.c

MEMBER OF process XSPICE

Public Domain

AUTHORS

    20 Mar 2026     Seth Hillbrand

SUMMARY

    This file contains the generic noise callback for all XSPICE code models.
    It supports two noise source discovery mechanisms:

    Declarative: Models that define reserved parameter names (noise_voltage,
    noise_current, noise_corner, noise_exponent) get automatic noise sources
    bound to their first output or input port.

    Programmatic: Models that set noise_programmatic=TRUE have their cm_func
    called with MIF_NOI, allowing them to register arbitrary noise sources
    via cm_noise_add_source() and set densities via NOISE_DENSITY().

INTERFACES

    MIFnoise()

============================================================================*/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/noisedef.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/devdefs.h"

#include "ngspice/mif.h"
#include "ngspice/mifdefs.h"
#include "ngspice/mifproto.h"
#include "ngspice/mifparse.h"
#include "ngspice/mifcmdat.h"

#include "ngspice/suffix.h"


/* Declarative noise source layout: groups of 3 (white, flicker, total) */
#define DECL_SRCS_PER_GROUP 3
#define DECL_WHITE  0
#define DECL_FLICKER 1
#define DECL_TOTAL   2

/* Indices for reserved parameter names, -1 if not present */
typedef struct {
    int  nv_idx;          /* noise_voltage parameter index */
    int  nc_idx;          /* noise_current parameter index */
    int  corner_idx;      /* noise_corner parameter index */
    int  exp_idx;         /* noise_exponent parameter index */
    int  prog_idx;        /* noise_programmatic parameter index */
} Mif_Noise_Param_Indices_t;


static void
find_noise_params(int mod_type, Mif_Noise_Param_Indices_t *idx)
{
    int i;
    int num_param = DEVices[mod_type]->DEVpublic.num_param;
    Mif_Param_Info_t *pinfo;

    idx->nv_idx = -1;
    idx->nc_idx = -1;
    idx->corner_idx = -1;
    idx->exp_idx = -1;
    idx->prog_idx = -1;

    for (i = 0; i < num_param; i++) {
        pinfo = &(DEVices[mod_type]->DEVpublic.param[i]);

        if (strcmp(pinfo->name, "noise_voltage") == 0)
            idx->nv_idx = i;
        else if (strcmp(pinfo->name, "noise_current") == 0)
            idx->nc_idx = i;
        else if (strcmp(pinfo->name, "noise_corner") == 0)
            idx->corner_idx = i;
        else if (strcmp(pinfo->name, "noise_exponent") == 0)
            idx->exp_idx = i;
        else if (strcmp(pinfo->name, "noise_programmatic") == 0)
            idx->prog_idx = i;
    }
}


/* Find first output port that is a voltage-type (has branch equation) */
static int
find_first_output_conn(MIFinstance *here, int *out_port)
{
    int i, j;

    for (i = 0; i < here->num_conn; i++) {
        if (here->conn[i]->is_null || !here->conn[i]->is_output)
            continue;

        for (j = 0; j < here->conn[i]->size; j++) {
            if (here->conn[i]->port[j]->is_null)
                continue;

            *out_port = j;
            return i;
        }
    }

    return -1;
}


/* Find first input port */
static int
find_first_input_conn(MIFinstance *here, int *in_port)
{
    int i, j;

    for (i = 0; i < here->num_conn; i++) {
        if (here->conn[i]->is_null || !here->conn[i]->is_input)
            continue;

        for (j = 0; j < here->conn[i]->size; j++) {
            if (here->conn[i]->port[j]->is_null)
                continue;

            *in_port = j;
            return i;
        }
    }

    return -1;
}


/*
 * Resolve noise source nodes from a connection/port and source type.
 * Returns 0 on success, -1 on invalid combination.
 */
static int
resolve_noise_nodes(MIFinstance *here, int conn, int port,
                    Mif_Noise_Src_Type_t type, int *node1, int *node2)
{
    Mif_Port_Data_t *pdata = here->conn[conn]->port[port];
    Mif_Smp_Ptr_t *smp = &(pdata->smp_data);

    if (type == MIF_NOISE_CURRENT) {
        *node1 = smp->pos_node;
        *node2 = smp->neg_node;
        return 0;
    }

    if (type == MIF_NOISE_CURRENT_POS) {
        *node1 = smp->pos_node;
        *node2 = 0;
        return 0;
    }

    if (type == MIF_NOISE_CURRENT_NEG) {
        *node1 = smp->neg_node;
        *node2 = 0;
        return 0;
    }

    /* MIF_NOISE_VOLTAGE */
    switch (pdata->type) {
    case MIF_VOLTAGE:
    case MIF_DIFF_VOLTAGE:
    case MIF_RESISTANCE:
    case MIF_DIFF_RESISTANCE:
        /* Voltage-type output ports have a branch equation */
        *node1 = smp->branch;
        *node2 = 0;
        return 0;

    case MIF_CURRENT:
    case MIF_DIFF_CURRENT:
    case MIF_VSOURCE_CURRENT:
        /* Current-type input ports have an ibranch equation */
        *node1 = smp->ibranch;
        *node2 = 0;
        return 0;

    default:
        return -1;
    }
}


static Mif_Boolean_t
is_voltage_noise_port(Mif_Port_Type_t type)
{
    switch (type) {
    case MIF_VOLTAGE:
    case MIF_DIFF_VOLTAGE:
    case MIF_RESISTANCE:
    case MIF_DIFF_RESISTANCE:
        return MIF_TRUE;

    default:
        return MIF_FALSE;
    }
}


/*
 * Allocate noise state arrays for an instance.
 * Must be called after num_noise_srcs and noise_prog_offset have been set.
 */
static void
alloc_noise_state(MIFinstance *here)
{
    int nsrcs = here->num_noise_srcs;

    if (nsrcs <= 0)
        return;

    here->MIFnVar = TMALLOC(double, NSTATVARS * nsrcs);
    here->noise_node1 = TMALLOC(int, nsrcs);
    here->noise_node2 = TMALLOC(int, nsrcs);
    here->noise_src_names = TMALLOC(char *, nsrcs);

    if (!here->MIFnVar || !here->noise_node1 || !here->noise_node2 || !here->noise_src_names) {
        here->num_noise_srcs = 0;
        return;
    }

    memset(here->MIFnVar, 0, (size_t)(NSTATVARS * nsrcs) * sizeof(double));
    memset(here->noise_node1, 0, (size_t)nsrcs * sizeof(int));
    memset(here->noise_node2, 0, (size_t)nsrcs * sizeof(int));
    memset(here->noise_src_names, 0, (size_t)nsrcs * sizeof(char *));

    if (here->noise_prog_offset < nsrcs) {
        int num_prog = nsrcs - here->noise_prog_offset;

        here->noise_prog_density = TMALLOC(double, num_prog);
    }
}


/*
 * Set up declarative noise sources during N_OPEN.
 * When count_only is TRUE, just counts eligible sources without writing arrays.
 * Always sets noise_decl_nv_base and noise_decl_nc_base on the instance.
 * Returns the number of declarative sources.
 */
static int
setup_declarative_sources(MIFinstance *here,
                          Mif_Noise_Param_Indices_t *idx,
                          int base_idx,
                          Mif_Boolean_t count_only)
{
    int count = 0;
    int out_conn, out_port, in_conn, in_port;
    int node1, node2;

    here->noise_decl_nv_base = -1;
    here->noise_decl_nc_base = -1;

    /* noise_voltage: 3 sources attached to first output port */
    if (idx->nv_idx >= 0 && !here->param[idx->nv_idx]->is_null) {
        out_conn = find_first_output_conn(here, &out_port);

        if (out_conn >= 0 &&
            is_voltage_noise_port(here->conn[out_conn]->port[out_port]->type) &&
            resolve_noise_nodes(here, out_conn, out_port,
                                MIF_NOISE_VOLTAGE, &node1, &node2) == 0) {

            here->noise_decl_nv_base = base_idx + count;

            if (!count_only) {
                int s = base_idx + count;

                here->noise_node1[s + DECL_WHITE] = node1;
                here->noise_node2[s + DECL_WHITE] = node2;
                here->noise_src_names[s + DECL_WHITE] = tprintf("_nv_white");

                here->noise_node1[s + DECL_FLICKER] = node1;
                here->noise_node2[s + DECL_FLICKER] = node2;
                here->noise_src_names[s + DECL_FLICKER] = tprintf("_nv_flicker");

                here->noise_node1[s + DECL_TOTAL] = node1;
                here->noise_node2[s + DECL_TOTAL] = node2;
                here->noise_src_names[s + DECL_TOTAL] = tprintf("_nv_total");
            }

            count += DECL_SRCS_PER_GROUP;
        }
    }

    /* noise_current: 3 sources attached to first input port */
    if (idx->nc_idx >= 0 && !here->param[idx->nc_idx]->is_null) {
        in_conn = find_first_input_conn(here, &in_port);

        if (in_conn >= 0 &&
            resolve_noise_nodes(here, in_conn, in_port,
                                MIF_NOISE_CURRENT, &node1, &node2) == 0) {

            here->noise_decl_nc_base = base_idx + count;

            if (!count_only) {
                int s = base_idx + count;

                here->noise_node1[s + DECL_WHITE] = node1;
                here->noise_node2[s + DECL_WHITE] = node2;
                here->noise_src_names[s + DECL_WHITE] = tprintf("_nc_white");

                here->noise_node1[s + DECL_FLICKER] = node1;
                here->noise_node2[s + DECL_FLICKER] = node2;
                here->noise_src_names[s + DECL_FLICKER] = tprintf("_nc_flicker");

                here->noise_node1[s + DECL_TOTAL] = node1;
                here->noise_node2[s + DECL_TOTAL] = node2;
                here->noise_src_names[s + DECL_TOTAL] = tprintf("_nc_total");
            }

            count += DECL_SRCS_PER_GROUP;
        }
    }

    return count;
}


/*
 * Evaluate declarative noise for one group (white + flicker + total).
 */
static void
eval_declarative_group(MIFinstance *here, CKTcircuit *ckt, Ndata *data,
                       NOISEAN *job, double *OnDens,
                       int param_idx, int corner_idx, int exp_idx,
                       int src_base, int nsrcs)
{
    double Sv, fc, n, f;
    double noizDens[3], lnNdens[3];
    double tempOutNoise, tempInNoise;
    int i;

    Sv = here->param[param_idx]->element[0].rvalue;
    fc = (corner_idx >= 0) ? here->param[corner_idx]->element[0].rvalue : 0.0;
    n = (exp_idx >= 0) ? here->param[exp_idx]->element[0].rvalue : 1.0;
    f = data->freq;

    /* White noise */
    NevalSrc(&noizDens[DECL_WHITE], &lnNdens[DECL_WHITE], ckt,
             N_GAIN, here->noise_node1[src_base + DECL_WHITE],
             here->noise_node2[src_base + DECL_WHITE], 0.0);
    noizDens[DECL_WHITE] *= Sv * Sv;
    lnNdens[DECL_WHITE] = log(MAX(noizDens[DECL_WHITE], N_MINLOG));

    /* Flicker noise */
    NevalSrc(&noizDens[DECL_FLICKER], NULL, ckt,
             N_GAIN, here->noise_node1[src_base + DECL_FLICKER],
             here->noise_node2[src_base + DECL_FLICKER], 0.0);
    noizDens[DECL_FLICKER] *= (fc > 0 && f > 0) ? Sv * Sv * pow(fc / f, n) : 0.0;
    lnNdens[DECL_FLICKER] = log(MAX(noizDens[DECL_FLICKER], N_MINLOG));

    /* Total */
    noizDens[DECL_TOTAL] = noizDens[DECL_WHITE] + noizDens[DECL_FLICKER];
    lnNdens[DECL_TOTAL] = log(MAX(noizDens[DECL_TOTAL], N_MINLOG));

    *OnDens += noizDens[DECL_TOTAL];

    /* Integration, following resnoise.c pattern */
    if (data->delFreq == 0.0) {

        for (i = 0; i < DECL_SRCS_PER_GROUP; i++)
            here->MIFnVar[LNLSTDENS * nsrcs + src_base + i] = lnNdens[i];

        if (data->freq == job->NstartFreq) {

            for (i = 0; i < DECL_SRCS_PER_GROUP; i++) {
                here->MIFnVar[OUTNOIZ * nsrcs + src_base + i] = 0.0;
                here->MIFnVar[INNOIZ * nsrcs + src_base + i] = 0.0;
            }
        }
    }
    else {

        for (i = 0; i < DECL_SRCS_PER_GROUP; i++) {
            if (i == DECL_TOTAL)
                continue;

            tempOutNoise = Nintegrate(noizDens[i], lnNdens[i],
                here->MIFnVar[LNLSTDENS * nsrcs + src_base + i], data);
            tempInNoise = Nintegrate(noizDens[i] * data->GainSqInv,
                lnNdens[i] + data->lnGainInv,
                here->MIFnVar[LNLSTDENS * nsrcs + src_base + i] + data->lnGainInv,
                data);

            here->MIFnVar[LNLSTDENS * nsrcs + src_base + i] = lnNdens[i];
            data->outNoiz += tempOutNoise;
            data->inNoise += tempInNoise;

            if (job->NStpsSm != 0) {
                here->MIFnVar[OUTNOIZ * nsrcs + src_base + i] += tempOutNoise;
                here->MIFnVar[OUTNOIZ * nsrcs + src_base + DECL_TOTAL] += tempOutNoise;
                here->MIFnVar[INNOIZ * nsrcs + src_base + i] += tempInNoise;
                here->MIFnVar[INNOIZ * nsrcs + src_base + DECL_TOTAL] += tempInNoise;
            }
        }
    }

    if (data->prtSummary) {

        for (i = 0; i < DECL_SRCS_PER_GROUP; i++)
            data->outpVector[data->outNumber++] = noizDens[i];
    }
}


/*
 * Set up XSPICE context for calling cm_func during noise analysis.
 * Uses the global g_mif_noise_cm_data (defined in mif.c) so that
 * cm_noise_add_source() in cm.c can access the noise data.
 */
static void
setup_noise_cm_context(MIFinstance *here, CKTcircuit *ckt,
                       Mif_Noise_Data_t *noise_data)
{
    g_mif_info.ckt = ckt;
    g_mif_info.instance = here;
    g_mif_info.errmsg = "";
    g_mif_info.circuit.call_type = MIF_ANALOG;
    g_mif_info.circuit.init = MIF_FALSE;
    g_mif_info.circuit.anal_init = MIF_FALSE;
    g_mif_info.circuit.anal_type = MIF_NOI;

    g_mif_noise_cm_data.circuit.anal_type = MIF_NOI;
    g_mif_noise_cm_data.circuit.anal_init = MIF_FALSE;
    g_mif_noise_cm_data.circuit.init = MIF_FALSE;
    g_mif_noise_cm_data.circuit.call_type = MIF_ANALOG;
    g_mif_noise_cm_data.circuit.frequency = ckt->CKTomega;
    g_mif_noise_cm_data.circuit.temperature = ckt->CKTtemp - 273.15;
    g_mif_noise_cm_data.circuit.time = 0.0;
    memset(g_mif_noise_cm_data.circuit.t, 0, sizeof(g_mif_noise_cm_data.circuit.t));
    g_mif_noise_cm_data.num_conn = here->num_conn;
    g_mif_noise_cm_data.conn = here->conn;
    g_mif_noise_cm_data.num_param = here->num_param;
    g_mif_noise_cm_data.param = here->param;
    g_mif_noise_cm_data.num_inst_var = here->num_inst_var;
    g_mif_noise_cm_data.inst_var = here->inst_var;
    g_mif_noise_cm_data.callback = &(here->callback);
    g_mif_noise_cm_data.noise = noise_data;
}


static void
restore_after_cm_func(void)
{
    g_mif_info.circuit.anal_type = MIF_AC;
    g_mif_noise_cm_data.noise = NULL;
}


/*
 * MIFnoise - Generic noise callback for all XSPICE code models.
 */
int
MIFnoise(int mode, int operation, GENmodel *genmodel, CKTcircuit *ckt,
          Ndata *data, double *OnDens)
{
    NOISEAN *job = (NOISEAN *) ckt->CKTcurJob;
    MIFmodel *model;
    MIFinstance *here;
    int mod_type;
    Mif_Noise_Param_Indices_t parm_idx;
    int i;

    model = (MIFmodel *) genmodel;
    mod_type = model->MIFmodType;

    /* Scan parameter names once per model type */
    find_noise_params(mod_type, &parm_idx);

    for (; model != NULL; model = MIFnextModel(model)) {

        if (!model->analog)
            continue;

        for (here = MIFinstances(model); here != NULL; here = MIFnextInstance(here)) {

            if (!here->analog)
                continue;

            switch (operation) {

            case N_OPEN:
            {
                int decl_count = 0;
                int prog_count = 0;
                Mif_Boolean_t has_programmatic = MIF_FALSE;

                /* Check if model has noise_programmatic parameter.
                 * If present but null (user didn't specify), defaults to TRUE.
                 * If present and not null, use the user's value. */
                if (parm_idx.prog_idx >= 0) {

                    if (here->param[parm_idx.prog_idx]->is_null ||
                        here->param[parm_idx.prog_idx]->element[0].bvalue) {
                        has_programmatic = MIF_TRUE;
                    }
                }

                if (!here->noise_initialized) {
                    /* First N_OPEN (N_DENS pass): discover and allocate.
                     * Count pass determines eligible sources and caches base indices. */
                    decl_count = setup_declarative_sources(here, &parm_idx, 0, MIF_TRUE);
                    here->noise_prog_offset = decl_count;

                    /* Count programmatic sources by calling cm_func */
                    if (has_programmatic) {
                        Mif_Noise_Data_t noise_data;
                        memset(&noise_data, 0, sizeof(noise_data));
                        noise_data.registering = MIF_TRUE;

                        setup_noise_cm_context(here, ckt, &noise_data);
                        DEVices[mod_type]->DEVpublic.cm_func(&g_mif_noise_cm_data);
                        restore_after_cm_func();

                        prog_count = noise_data.num_prog_srcs;
                        here->num_noise_srcs = decl_count + prog_count;
                        alloc_noise_state(here);
                        setup_declarative_sources(here, &parm_idx, 0, MIF_FALSE);

                        /* Resolve programmatic source nodes.
                         * Sources with conn == -1 were rejected during registration
                         * and get node1=node2=0 (zero noise contribution). */
                        for (i = 0; i < prog_count; i++) {
                            int si = decl_count + i;
                            int n1 = 0, n2 = 0;

                            if (noise_data.prog_conn[i] >= 0 &&
                                resolve_noise_nodes(here,
                                    noise_data.prog_conn[i],
                                    noise_data.prog_port[i],
                                    noise_data.prog_types[i],
                                    &n1, &n2) == 0) {
                                here->noise_node1[si] = n1;
                                here->noise_node2[si] = n2;
                            }

                            here->noise_src_names[si] = noise_data.prog_names[i];
                            noise_data.prog_names[i] = NULL;
                        }

                        /* Free temporary registration arrays */
                        if (noise_data.prog_types)
                            FREE(noise_data.prog_types);

                        if (noise_data.prog_conn)
                            FREE(noise_data.prog_conn);

                        if (noise_data.prog_port)
                            FREE(noise_data.prog_port);

                        if (noise_data.prog_names) {

                            for (i = 0; i < prog_count; i++) {
                                if (noise_data.prog_names[i])
                                    FREE(noise_data.prog_names[i]);
                            }

                            FREE(noise_data.prog_names);
                        }
                    }
                    else {
                        here->num_noise_srcs = decl_count;
                        alloc_noise_state(here);
                        setup_declarative_sources(here, &parm_idx, 0, MIF_FALSE);
                    }

                    here->noise_initialized = MIF_TRUE;
                }

                /* Register output variable names (both N_DENS and INT_NOIZ passes) */
                if (here->num_noise_srcs > 0 && job->NStpsSm != 0) {

                    switch (mode) {
                    case N_DENS:
                        for (i = 0; i < here->num_noise_srcs; i++) {
                            NOISE_ADD_OUTVAR(ckt, data, "onoise_%s%s",
                                here->MIFname,
                                here->noise_src_names[i] ? here->noise_src_names[i] : "");
                        }
                        break;

                    case INT_NOIZ:
                        for (i = 0; i < here->num_noise_srcs; i++) {
                            NOISE_ADD_OUTVAR(ckt, data, "onoise_total_%s%s",
                                here->MIFname,
                                here->noise_src_names[i] ? here->noise_src_names[i] : "");
                            NOISE_ADD_OUTVAR(ckt, data, "inoise_total_%s%s",
                                here->MIFname,
                                here->noise_src_names[i] ? here->noise_src_names[i] : "");
                        }
                        break;
                    }
                }
                break;
            }

            case N_CALC:
            {
                int nsrcs = here->num_noise_srcs;
                int prog_offset = here->noise_prog_offset;

                if (nsrcs <= 0)
                    break;

                switch (mode) {
                case N_DENS:
                {
                    /* Evaluate declarative groups using cached base indices */
                    if (here->noise_decl_nv_base >= 0) {
                        eval_declarative_group(here, ckt, data, job, OnDens,
                            parm_idx.nv_idx, parm_idx.corner_idx, parm_idx.exp_idx,
                            here->noise_decl_nv_base, nsrcs);
                    }

                    if (here->noise_decl_nc_base >= 0) {
                        eval_declarative_group(here, ckt, data, job, OnDens,
                            parm_idx.nc_idx, parm_idx.corner_idx, parm_idx.exp_idx,
                            here->noise_decl_nc_base, nsrcs);
                    }

                    /* Evaluate programmatic noise sources */
                    if (nsrcs > prog_offset && here->noise_prog_density) {
                        int num_prog = nsrcs - prog_offset;
                        Mif_Noise_Data_t noise_data;
                        double noizDens_p, lnNdens_p;
                        double tempOutNoise, tempInNoise;

                        memset(&noise_data, 0, sizeof(noise_data));
                        noise_data.registering = MIF_FALSE;
                        noise_data.freq = data->freq;
                        noise_data.density = here->noise_prog_density;
                        memset(noise_data.density, 0, (size_t)num_prog * sizeof(double));

                        setup_noise_cm_context(here, ckt, &noise_data);
                        DEVices[mod_type]->DEVpublic.cm_func(&g_mif_noise_cm_data);
                        restore_after_cm_func();

                        for (i = 0; i < num_prog; i++) {
                            int si = prog_offset + i;

                            NevalSrc(&noizDens_p, &lnNdens_p, ckt,
                                     N_GAIN,
                                     here->noise_node1[si],
                                     here->noise_node2[si], 0.0);

                            noizDens_p *= noise_data.density[i];
                            lnNdens_p = log(MAX(noizDens_p, N_MINLOG));

                            *OnDens += noizDens_p;

                            if (data->delFreq == 0.0) {
                                here->MIFnVar[LNLSTDENS * nsrcs + si] = lnNdens_p;

                                if (data->freq == job->NstartFreq) {
                                    here->MIFnVar[OUTNOIZ * nsrcs + si] = 0.0;
                                    here->MIFnVar[INNOIZ * nsrcs + si] = 0.0;
                                }
                            }
                            else {
                                tempOutNoise = Nintegrate(noizDens_p, lnNdens_p,
                                    here->MIFnVar[LNLSTDENS * nsrcs + si], data);
                                tempInNoise = Nintegrate(
                                    noizDens_p * data->GainSqInv,
                                    lnNdens_p + data->lnGainInv,
                                    here->MIFnVar[LNLSTDENS * nsrcs + si] + data->lnGainInv,
                                    data);

                                here->MIFnVar[LNLSTDENS * nsrcs + si] = lnNdens_p;
                                data->outNoiz += tempOutNoise;
                                data->inNoise += tempInNoise;

                                if (job->NStpsSm != 0) {
                                    here->MIFnVar[OUTNOIZ * nsrcs + si] += tempOutNoise;
                                    here->MIFnVar[INNOIZ * nsrcs + si] += tempInNoise;
                                }
                            }

                            if (data->prtSummary)
                                data->outpVector[data->outNumber++] = noizDens_p;
                        }
                    }
                    break;
                }

                case INT_NOIZ:
                    if (job->NStpsSm != 0) {

                        for (i = 0; i < nsrcs; i++) {
                            data->outpVector[data->outNumber++] =
                                here->MIFnVar[OUTNOIZ * nsrcs + i];
                            data->outpVector[data->outNumber++] =
                                here->MIFnVar[INNOIZ * nsrcs + i];
                        }
                    }
                    break;
                }
                break;
            }

            case N_CLOSE:
                return (OK);

            } /* switch operation */
        } /* for instances */
    } /* for models */

    return (OK);
}


void
MIF_free_noise_state(MIFinstance *here)
{
    int i;

    if (here->noise_src_names) {

        for (i = 0; i < here->num_noise_srcs; i++)
            tfree(here->noise_src_names[i]);

        tfree(here->noise_src_names);
    }

    tfree(here->MIFnVar);
    tfree(here->noise_node1);
    tfree(here->noise_node2);
    tfree(here->noise_prog_density);
    here->num_noise_srcs = 0;
    here->noise_prog_offset = 0;
    here->noise_decl_nv_base = -1;
    here->noise_decl_nc_base = -1;
    here->noise_initialized = MIF_FALSE;
}
