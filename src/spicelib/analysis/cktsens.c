/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Modified: 2000 AlanFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/spmatrix.h"
#include "ngspice/gendefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/sensdefs.h"
#include "ngspice/sensgen.h"

/* #define ASDEBUG */
#ifdef ASDEBUG
#define DEBUG(X)	if ((X) < Sens_Debug)
int Sens_Debug = 0;
char SF1[] = "res";
char SF2[] = "dc";
char SF3[] = "bf";
#endif

char *Sfilter = NULL;
double Sens_Delta = 0.000001;
double Sens_Abs_Delta = 0.000001;

static int sens_setp(sgen *sg, CKTcircuit *ckt, IFvalue *val);
static int sens_load(sgen *sg, CKTcircuit *ckt, int is_dc);
static int sens_temp(sgen *sg, CKTcircuit *ckt);
static int count_steps(int type, double low, double high, int steps, double *stepsize);
static double inc_freq(double freq, int type, double step_size);

#define save_context(thing, place) {	    \
    place = thing;			    \
}

#define release_context(thing, place)	    \
    if(place) {				    \
	thing = place;			    \
	place = NULL;			    \
    }


/*
 *	Procedure:
 *
 *		Determine operating point (call CKTop)
 *
 *		For each frequency point:
 *			(for AC) call NIacIter to get base node voltages
 *			For each element/parameter in the test list:
 *				construct the perturbation matrix
 *				Solve for the sensitivities:
 *					delta_E = Y^-1 (delta_Y E - delta_I)
 *				save results
 */

static int	error;
int sens_sens(CKTcircuit *ckt, int restart)
{
	SENS_AN	*job = (SENS_AN *) ckt->CKTcurJob;

	static int	size;
	static double	*delta_I, *delta_iI,
			*delta_I_delta_Y, *delta_iI_delta_Y;
	sgen		*sg;
	static double	freq;
	static int	nfreqs;
	static int	i;
	static SMPmatrix	*delta_Y = NULL, *Y;
	static double	step_size;
	double		*E, *iE;
	IFvalue		value, nvalue;
	double		*output_values;
	IFcomplex	*output_cvalues;
	double		delta_var;
	int             (*fn) (SMPmatrix *, GENmodel *, CKTcircuit *, int *);
	static int	is_dc;
	int		k, j, n;
	int		num_vars, branch_eq=0;
	runDesc		*sen_data = NULL;
	char		namebuf[513];
	IFuid		*output_names, freq_name;
	int		bypass;
	int		type;
	double		*saved_rhs = NULL,
			*saved_irhs = NULL;
	SMPmatrix	*saved_matrix = NULL;

#ifndef notdef
#ifdef notdef
	for (sg = sgen_init(ckt, 0); sg; sgen_next(&sg)) {
		if (sg->is_instparam)
			printf("%s:%s:%s -> param %s\n",
				DEVices[sg->dev]->DEVpublic.name,
				sg->model->GENmodName,
				sg->instance->GENname,
				sg->ptable[sg->param].keyword);
		else
			printf("%s:%s:%s -> mparam %s\n",
				DEVices[sg->dev]->DEVpublic.name,
				sg->model->GENmodName,
				sg->instance->GENname,
				sg->ptable[sg->param].keyword);
	}
#endif
#ifdef ASDEBUG
	DEBUG(1)
		printf(">>> restart : %d\n", restart);
#endif

	/* get to work */

	restart = 1;
	if (restart) {

		freq = 0.0;
		is_dc = (job->step_type == SENS_DC);
		nfreqs = count_steps(job->step_type, job->start_freq,
			job->stop_freq, job->n_freq_steps,
			&step_size);

		if (!is_dc)
			freq = job->start_freq;

		error = CKTop(ckt,
			(ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
			(ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
			ckt->CKTdcMaxIter);

#ifdef notdef
		ckt->CKTmode = (ckt->CKTmode & MODEUIC)
			| MODEDCOP | MODEINITSMSIG;
#endif
		if (error)
			return error;

		size = SMPmatSize(ckt->CKTmatrix);

		/* Create the perturbation matrix */
		error = SMPnewMatrix(&delta_Y, size);
		if (error)
			return error;

		size += 1;

		/* Create an extra rhs */
		delta_I = TMALLOC(double, size);
		delta_iI = TMALLOC(double, size);

		delta_I_delta_Y = TMALLOC(double, size);
		delta_iI_delta_Y = TMALLOC(double, size);


		num_vars = 0;
		for (sg = sgen_init(ckt, is_dc); sg; sgen_next(&sg)) {
			num_vars += 1;
		}

		if (!num_vars)
			return OK;	/* XXXX Should be E_ something */

		k = 0;
		output_names = TMALLOC(IFuid, num_vars);
		for (sg = sgen_init(ckt, is_dc); sg; sgen_next(&sg)) {
			if (!sg->is_instparam) {
				sprintf(namebuf, "%s:%s",
					sg->instance->GENname,
					sg->ptable[sg->param].keyword);
			} else if ((sg->ptable[sg->param].dataType
				& IF_PRINCIPAL) && sg->is_principle == 1)
			{
				sprintf(namebuf, "%s", sg->instance->GENname);
			} else {
				sprintf(namebuf, "%s_%s",
					sg->instance->GENname,
					sg->ptable[sg->param].keyword);
			}

			SPfrontEnd->IFnewUid (ckt, output_names + k, NULL, namebuf, UID_OTHER, NULL);
			k += 1;
		}

		if (is_dc) {
			type = IF_REAL;
			freq_name = NULL;
		} else {
			type = IF_COMPLEX;
			SPfrontEnd->IFnewUid (ckt, &freq_name, NULL, "frequency", UID_OTHER, NULL);
		}

                error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                                   ckt->CKTcurJob->JOBname,
                                                   freq_name, IF_REAL,
                                                   num_vars, output_names, type,
                                                   &sen_data);
		if (error)
			return error;

		FREE(output_names);
		if (is_dc) {
			output_values = TMALLOC(double, num_vars);
			output_cvalues = NULL;
		} else {
			output_values = NULL;
			output_cvalues = TMALLOC(IFcomplex, num_vars);
			if (job->step_type != SENS_LINEAR)
			    SPfrontEnd->OUTattributes (sen_data, NULL, OUT_SCALE_LOG, NULL);

		}

	} else {
		/*XXX Restore saved state */
		output_values = NULL;
		output_cvalues = NULL;
		fprintf(stderr, "ERROR: restore is not implemented for cktsens\n");
		controlled_exit(1);
	}

#ifdef ASDEBUG
	DEBUG(1)
		printf("start: %f, num: %d, dc: %d\n", freq, nfreqs, is_dc);
#endif

	if (!job->output_volt)
		branch_eq = CKTfndBranch(ckt, job->output_src);
	bypass = ckt->CKTbypass;
	ckt->CKTbypass = 0;

	/* CKTop solves into CKTrhs and CKTmatrix,
	 *	 CKTirhs is hopefully zero (fresh allocated ?) */

	E = ckt->CKTrhs;
	iE = ckt->CKTirhs;
	Y = ckt->CKTmatrix;

#ifdef ASDEBUG
	DEBUG(1) {
		printf("Operating point:\n");
		for (i = 0; i < size; i++)
			printf("          E [%d] = %20.15g\n", i, E[i]);
	}
#endif

#ifdef notdef
	for (j = 0; j <= ckt->CKTmaxOrder + 1; j++) {
		save_states[j] = ckt->CKTstates[j];
		ckt->CKTstates[j] = NULL;
	}
#endif

	for (i = 0; i < nfreqs; i++) {
		/* XXX handle restart */

		n = 0;

		if (SPfrontEnd->IFpauseTest()) {
			/* XXX Save State */
			return E_PAUSE;
		}

		for (j = 0; j < size; j++) {
			delta_I[j] = 0.0;
			delta_iI[j] = 0.0;
		}

		if (freq != 0.0) {

			/* This generates Y in LU form */
			ckt->CKTomega = 2.0 * M_PI * freq;

			/* Yes, all this has to be re-done */
			/* XXX Free old states */
			error = CKTunsetup(ckt);
			if (error)
				return error;

			/* XXX ckt->CKTmatrix = Y; */

			error = CKTsetup(ckt);
			if (error)
				return error;

#ifdef notdef
			for (j = 0; j <= ckt->CKTmaxOrder + 1; j++) {
				/* XXX Free new states */
				ckt->CKTstates[j] = save_states[j];
			}
#endif
			error = CKTtemp(ckt);
			if (error)
				return error;
			error = CKTload(ckt); /* INITSMSIGS */
			if (error)
				return error;
			error = NIacIter(ckt);
			if (error)
				return error;

#ifdef notdef
			/* XXX Why? */
			for (j = 0; j <= ckt->CKTmaxOrder + 1; j++) {
				ckt->CKTstates[j] = NULL;
			}
#endif

			/* NIacIter solves into CKTrhsOld, CKTirhsOld and CKTmatrix */
			E = ckt->CKTrhsOld;
			iE = ckt->CKTirhsOld;
			Y = ckt->CKTmatrix;
		}

		/* Use a different vector & matrix */

		save_context(ckt->CKTrhs, saved_rhs);
		save_context(ckt->CKTirhs, saved_irhs);
		save_context(ckt->CKTmatrix, saved_matrix);

		ckt->CKTrhs = delta_I;
		ckt->CKTirhs = delta_iI;
		ckt->CKTmatrix = delta_Y;

		/* calc. effect of each param */
		for (sg = sgen_init(ckt, is_dc /* job->plist */);
			sg; sgen_next(&sg))
		{

#ifdef ASDEBUG
			DEBUG(2) {
				printf("E/iE: %x/%x; delta_I/iI: %x/%x\n",
					E, iE, delta_I, delta_iI);
				printf("cktrhs/irhs: %x/%x\n",
					ckt->CKTrhs, ckt->CKTirhs);

				if (sg->is_instparam)
					printf("%s:%s:%s -> param %s\n",
					DEVices[sg->dev]->DEVpublic.name,
						sg->model->GENmodName,
						sg->instance->GENname,
						sg->ptable[sg->param].keyword);
				else
					printf("%s:%s:%s -> mparam %s\n",
					DEVices[sg->dev]->DEVpublic.name,
						sg->model->GENmodName,
						sg->instance->GENname,
						sg->ptable[sg->param].keyword);
			}
#endif

			SMPcClear(delta_Y);

			for (j = 0; j < size; j++) {
				delta_I[j] = 0.0;
				delta_iI[j] = 0.0;
			}

			/* ? should this just call CKTsetup
			 * ? but then CKThead would have to get fiddled with */

			ckt->CKTnumStates = sg->istate;

			fn = DEVices[sg->dev]->DEVsetup;
			if (fn) {
                CKTnode *node = ckt->CKTlastNode;
                /* XXXX insert old state base here ?? */
                fn (delta_Y, sg->model, ckt, &ckt->CKTnumStates);
                if (node != ckt->CKTlastNode) {
                    fprintf(stderr, "Internal Error: node allocation in DEVsetup() during sensitivity analysis, this will cause serious troubles !, please report this issue !\n");
                    controlled_exit(EXIT_FAILURE);
                }
            }

			/* ? CKTsetup would call NIreinit instead */
			ckt->CKTniState = NISHOULDREORDER | NIACSHOULDREORDER;

			/* XXX instead of calling temp here, just swap
			 * back to the original states */
			(void) sens_temp(sg, ckt);

			/* XXX Leave original E until here!! so that temp reads
			 * the right node voltages */

			if (sens_load(sg, ckt, is_dc)) {
				if (error && error != E_BADPARM)
					return error;	/* XXX */
				continue;
			}

			/* Alter the parameter */

#ifdef ASDEBUG
			DEBUG(1) printf("Original value: %g\n", sg->value);
#endif

#ifdef ASDEBUG
			DEBUG(2) {
				printf("Effect of device:\n");
				SMPprint(delta_Y, NULL);
				printf("LHS:\n");
				for (j = 0; j < size; j++)
					printf("%d: %g, %g\n", j,
						delta_I[j], delta_iI[j]);
			}
#endif

			if (sg->value != 0.0)
				delta_var = sg->value * Sens_Delta;
			else
				delta_var = Sens_Abs_Delta;

			nvalue.rValue = sg->value + delta_var;

#ifdef ASDEBUG
			DEBUG(1)
				printf("New value: %g\n", nvalue.rValue);
#endif

			sens_setp(sg, ckt, &nvalue);
			if (error && error != E_BADPARM)
				return error;

			SMPconstMult(delta_Y, -1.0);
			for (j = 0; j < size; j++) {
				delta_I[j] *= -1.0;
				delta_iI[j] *= -1.0;
			}

#ifdef ASDEBUG
			DEBUG(2) {
				printf("Effect of negating matrix:\n");
				SMPprint(delta_Y, NULL);
				for (j = 0; j < size; j++)
					printf("%d: %g, %g\n", j,
						delta_I[j], delta_iI[j]);
			}
#endif

			/* XXX swap back to temp states ??   Naw ... */
			(void) sens_temp(sg, ckt);

#ifdef ASDEBUG
			DEBUG(1) {
				if (sens_getp(sg, ckt, &value)) {
					continue;
				}

				printf("New value in device: %g\n",
					value.rValue);
			}
#endif

			sens_load(sg, ckt, is_dc);

#ifdef ASDEBUG
			DEBUG(2) {
				printf("Effect of changing the parameter:\n");
				SMPprint(delta_Y, NULL);
				for (j = 0; j < size; j++)
					printf("%d: %g, %g\n", j,
						delta_I[j], delta_iI[j]);
			}
#endif
			/* Set the perturbed variable back to it's
			 * original value
			 */

			value.rValue = sg->value;
			sens_setp(sg, ckt, &value);
			(void) sens_temp(sg, ckt); /* XXX is this necessary? */

			/* Back to business . . . */

#ifdef ASDEBUG
			DEBUG(2)
				for (j = 0; j < size; j++)
					printf("          E [%d] = %20.15g\n",
						j, E[j]);
#endif

			/* delta_Y E */
			SMPmultiply(delta_Y, delta_I_delta_Y, E,
				    delta_iI_delta_Y, iE);

#ifdef ASDEBUG
			DEBUG(2)
				for (j = 0; j < size; j++)
					printf("delta_Y * E [%d] = %20.15g\n",
						j, delta_I_delta_Y[j]);
#endif

			/* delta_I - delta_Y E */
			for (j = 0; j < size; j++) {
				delta_I[j] -= delta_I_delta_Y[j];
				delta_iI[j] -= delta_iI_delta_Y[j];
			}

#ifdef ASDEBUG
			DEBUG(2) {
				printf(">>> Y:\n");
				SMPprint(Y, NULL);
				for (j = 0; j < size; j++)
					printf("%d: %g, %g\n", j,
						delta_I[j], delta_iI[j]);
			}
#endif
			/* Solve; Y already factored */
			SMPcSolve(Y, delta_I, delta_iI, NULL, NULL);

                        /* the special `0' node
                        *    the matrix indizes are [1..n]
                        *    yet the vector indizes are [0..n]
                        *    with [0] being implicit === 0
                        */
                        delta_I[0]  = 0.0;
                        delta_iI[0] = 0.0;

#ifdef ASDEBUG
			DEBUG(2) {
				for (j = 1; j < size; j++) {

					if (sg->is_instparam)
						printf("%d/%s.%s = %g, %g\n",
							j,
							sg->instance->GENname,
						sg->ptable[sg->param].keyword,
						delta_I[j], delta_iI[j]);
					else
						printf("%d/%s:%s = %g, %g\n",
							j,
							sg->instance->GENname,
						sg->ptable[sg->param].keyword,
						delta_I[j], delta_iI[j]);

				}
			}
#endif

			/* delta_I is now equal to delta_E */

			if (is_dc) {
				if (job->output_volt)
					output_values[n] =
					    delta_I [job->output_pos->number]
					    - delta_I [job->output_neg->number];
				else {
					output_values[n] = delta_I[branch_eq];
				}
				output_values[n] /= delta_var;
			} else {
				if (job->output_volt) {
					output_cvalues[n].real =
					    delta_I [job->output_pos->number]
					    - delta_I [job->output_neg->number];
					output_cvalues[n].imag =
					    delta_iI [job->output_pos->number]
					    - delta_iI [job->output_neg->number];
				} else {
					output_cvalues[n].real =
						delta_I[branch_eq];
					output_cvalues[n].imag =
						delta_iI[branch_eq];
				}
				output_cvalues[n].real /= delta_var;
				output_cvalues[n].imag /= delta_var;
			}

			n += 1;

		}

		release_context(ckt->CKTrhs, saved_rhs);
		release_context(ckt->CKTirhs, saved_irhs);
		release_context(ckt->CKTmatrix, saved_matrix);

		if (is_dc)
			nvalue.v.vec.rVec = output_values;
		else
			nvalue.v.vec.cVec = output_cvalues;

		value.rValue = freq;

		SPfrontEnd->OUTpData (sen_data, &value, &nvalue);

		freq = inc_freq(freq, job->step_type, step_size);

	}

	SPfrontEnd->OUTendPlot (sen_data);

	if (is_dc) {
		FREE(output_values);	/* XXX free various vectors */
	} else {
		FREE(output_cvalues);	/* XXX free various vectors */
	}

	release_context(ckt->CKTrhs, saved_rhs);
	release_context(ckt->CKTirhs, saved_irhs);
	release_context(ckt->CKTmatrix, saved_matrix);

	SMPdestroy(delta_Y);
	FREE(delta_I);
	FREE(delta_iI);

	FREE(delta_I_delta_Y);
	FREE(delta_iI_delta_Y);

	ckt->CKTbypass = bypass;

#ifdef notdef
	for (j = 0; j <= ckt->CKTmaxOrder + 1; j++) {
		if (ckt->CKTstates[j])
			FREE(ckt->CKTstates[j]);
		ckt->CKTstates[j] = save_states[j];
	}
#endif
#endif

	return OK;
}

double
inc_freq(double freq, int type, double step_size)
{
	if (type != SENS_LINEAR)
		freq *= step_size;
	else
		freq += step_size;

	return freq;
}
/*
static double
next_freq(int type, double freq, double stepsize)
{
	double	s=0;

	switch (type) {
	case SENS_DC:
		s = 0;
		break;

	case SENS_LINEAR:
		s = freq + stepsize;
		break;

	case SENS_DECADE:
	case SENS_OCTAVE:
		s = freq * stepsize;
		break;
	}
	return s;
}
*/
int
count_steps(int type, double low, double high, int steps, double *stepsize)
{
	double	s;
	int	n;

	if (steps < 1)
		steps = 1;

	switch (type) {
	default:
	case SENS_DC:
		n = 0;
		s = 0;
		break;

	case SENS_LINEAR:
		n = steps;
		s = (high - low) / steps;
		break;

	case SENS_DECADE:
		if (low <= 0.0)
			low = 1e-3;
		if (high <= low)
			high = 10.0 * low;
		n = (int)(steps * log10(high/low) + 1.01);
		s = pow(10.0, 1.0 / steps);
		break;

	case SENS_OCTAVE:
		if (low <= 0.0)
			low = 1e-3;
		if (high <= low)
			high = 2.0 * low;
		n = (int)(steps * log(high/low) / M_LOG2E + 1.01);
		s = pow(2.0, 1.0 / steps);
		break;
	}

	if (n <= 0)
		n = 1;

	*stepsize = s;
	return n;
}

static int
sens_load(sgen *sg, CKTcircuit *ckt, int is_dc)
{
 	int	(*fn) (GENmodel *, CKTcircuit *);

	error = 0;

	if (!is_dc)
		fn = DEVices[sg->dev]->DEVacLoad;
	else
		fn = DEVices[sg->dev]->DEVload;

	if (fn)
		error = fn (sg->model, ckt);
	else
		return 1;

	return error;
}


static int
sens_temp(sgen *sg, CKTcircuit *ckt)
{
	int	(*fn) (GENmodel *, CKTcircuit *);

	error = 0;

	fn = DEVices[sg->dev]->DEVtemperature;

	if (fn)
		error = fn (sg->model, ckt);
	else
		return 1;

	return error;
}

/* Get parameter value */
int
sens_getp(sgen *sg, CKTcircuit *ckt, IFvalue *val)
{
	int	pid;

	NG_IGNORE(ckt);

	error = 0;

	if (sg->is_instparam) {
		int (*fn) (CKTcircuit*, GENinstance*, int, IFvalue*, IFvalue*);
		fn = DEVices[sg->dev]->DEVask;
		pid = DEVices[sg->dev]->DEVpublic.instanceParms[sg->param].id;
		if (fn)
			error = fn (ckt, sg->instance, pid, val, NULL);
		else
			return 1;
	} else {
		int (*fn) (CKTcircuit*, GENmodel*, int, IFvalue*);
		fn = DEVices[sg->dev]->DEVmodAsk;
		pid = DEVices[sg->dev]->DEVpublic.modelParms[sg->param].id;
		if (fn)
			error = fn (ckt, sg->model, pid, val);
		else
			return 1;
	}

	if (error) {
		if (sg->is_instparam)
			printf("GET ERROR: %s:%s:%s -> param %s (%d)\n",
				DEVices[sg->dev]->DEVpublic.name,
				sg->model->GENmodName,
				sg->instance->GENname,
				sg->ptable[sg->param].keyword, pid);
		else
			printf("GET ERROR: %s:%s:%s -> mparam %s (%d)\n",
				DEVices[sg->dev]->DEVpublic.name,
				sg->model->GENmodName,
				sg->instance->GENname,
				sg->ptable[sg->param].keyword, pid);
	}

	return error;
}

/* Get parameter value */
int
sens_setp(sgen *sg, CKTcircuit *ckt, IFvalue *val)
{
	int	pid;

	NG_IGNORE(ckt);

	error = 0;

	if (sg->is_instparam) {
		int (*fn) (int, IFvalue *, GENinstance *, IFvalue *);
		fn = DEVices[sg->dev]->DEVparam;
		pid = DEVices[sg->dev]->DEVpublic.instanceParms[sg->param].id;
		if (fn)
			error = fn (pid, val, sg->instance, NULL);
		else
			return 1;
	} else {
		int (*fn) (int, IFvalue *, GENmodel *);
		fn = DEVices[sg->dev]->DEVmodParam;
		pid = DEVices[sg->dev]->DEVpublic.modelParms[sg->param].id;
		if (fn)
			error = fn (pid, val, sg->model);
		else
			return 1;
	}

	if (error) {
		if (sg->is_instparam)
			printf("SET ERROR: %s:%s:%s -> param %s (%d)\n",
				DEVices[sg->dev]->DEVpublic.name,
				sg->model->GENmodName,
				sg->instance->GENname,
				sg->ptable[sg->param].keyword, pid);
		else
			printf("SET ERROR: %s:%s:%s -> mparam %s (%d)\n",
				DEVices[sg->dev]->DEVpublic.name,
				sg->model->GENmodName,
				sg->instance->GENname,
				sg->ptable[sg->param].keyword, pid);
	}

	return error;
}
