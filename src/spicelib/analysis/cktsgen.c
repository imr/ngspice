/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/gendefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sensgen.h"

	/* XXX */
extern char	*Sfilter;

int set_model(sgen *);
int set_param(sgen *);
int set_inst(sgen *);
int set_dev(sgen *);

sgen *
sgen_init(CKTcircuit *ckt, int is_dc)
{
	sgen	*sg;

	sg = TMALLOC(sgen, 1);
	sg->param = 99999;
	sg->is_instparam = 0;
	sg->dev = -1;
	sg->istate = 0;
	sg->ckt = ckt;
	sg->devlist = ckt->CKThead;
	sg->instance = sg->first_instance = sg->next_instance = NULL;
	sg->model = sg->next_model = NULL;
	sg->ptable = NULL;
	sg->is_dc = is_dc;
	sg->is_principle = 0;
	sg->is_q = 0;
	sg->is_zerook = 0;
	sg->value = 0.0;

	sgen_next(&sg);	/* get the ball rolling XXX check return val? */

	return sg;
}

int
sgen_next(sgen **xsg)
{
	sgen	*sg = *xsg;
	int	good, done;
	int	i;

	done = 0;
	i = sg->dev;

	do {
		if (sg->instance) {
			if (sg->ptable) {
				do {
					sg->param += 1;
				} while (sg->param < sg->max_param
					&& !set_param(sg));
			} else {
				sg->max_param = -1;
			}

			if (sg->param < sg->max_param) {
				done = 1;
			} else if (!sg->is_instparam) {
				/* Try instance parameters now */
				sg->is_instparam = 1;
				sg->param = -1;
				sg->max_param =
					*DEVices[i]->DEVpublic.numInstanceParms;
				sg->ptable =
					DEVices[i]->DEVpublic.instanceParms;
			} else {
				sg->is_principle = 0;
				sg->instance->GENnextInstance =
					sg->next_instance;
				sg->instance->GENstate = sg->istate;
				sg->instance = NULL;
			}

		} else if (sg->model) {

			/* Find the first/next good instance for this model */
			for (good = 0; !good && sg->next_instance;
				good = set_inst(sg))
			{
				sg->instance = sg->next_instance;
				sg->next_instance =
					sg->instance->GENnextInstance;
			}


			if (good) {
				sg->is_principle = 0;
				sg->istate = sg->instance->GENstate;
				sg->instance->GENnextInstance = NULL;
				sg->model->GENinstances = sg->instance;
				if (DEVices[i]->DEVpublic.modelParms) {
					sg->max_param =
						*DEVices[i]->DEVpublic.
						numModelParms;
					sg->ptable =
						DEVices[i]->DEVpublic.
						modelParms;
				} else {
					sg->ptable = NULL;
				}
				sg->param = -1;
				sg->is_instparam = 0;
			} else {
				/* No good instances of this model */
				sg->model->GENinstances = sg->first_instance;
				sg->model->GENnextModel = sg->next_model;
				sg->model = NULL;
			}

		} else if (i >= 0) {

			/* Find the first/next good model for this device */
			for (good = 0; !good && sg->next_model;
				good = set_model(sg))
			{
				sg->model = sg->next_model;
				sg->next_model = sg->model->GENnextModel;
			}

			if (good) {
				sg->model->GENnextModel = NULL;
				sg->devlist[i] = sg->model;
				if (DEVices[i]->DEVpublic.modelParms) {
					sg->max_param =
						*DEVices[i]->DEVpublic.
						numModelParms;
					sg->ptable =
						DEVices[i]->DEVpublic.
						modelParms;
				} else {
					sg->ptable = NULL;
				}
				sg->next_instance = sg->first_instance
					= sg->model->GENinstances;
			} else {
				/* No more good models for this device */
				sg->devlist[i] = sg->first_model;
				i = -1; /* Try the next good device */
			}

		} else if (i < DEVmaxnum && sg->dev < DEVmaxnum) {

			/* Find the next good device in this circuit */

			do
				sg->dev++;
			while (sg->dev < DEVmaxnum && sg->devlist[sg->dev]
				&& !set_dev(sg));

			i = sg->dev;

			if (i >= DEVmaxnum) /* PN: Segafult if not = */
				done = 1;
			sg->first_model = sg->next_model = (i<DEVmaxnum) ? sg->devlist[i] : NULL;

		} else {
			done = 1;
		}

	} while (!done);

	if (sg->dev >= DEVmaxnum) {
		FREE(sg);
		*xsg = NULL;
	}
	return 1;
}

int set_inst(sgen *sg)
{
	NG_IGNORE(sg);
	return 1;
}

int set_model(sgen *sg)
{
	NG_IGNORE(sg);
	return 1;
}

int set_dev(sgen *sg)
{
	NG_IGNORE(sg);
	return 1;
}

int set_param(sgen *sg)
{
	IFvalue	ifval;

	if (!sg->ptable[sg->param].keyword)
		return 0;
	if (Sfilter && strncmp(sg->ptable[sg->param].keyword, Sfilter,
			strlen(Sfilter)))
		return 0;
	if ((sg->ptable[sg->param].dataType &
		(IF_SET|IF_ASK|IF_REAL|IF_VECTOR|IF_REDUNDANT|IF_NONSENSE))
			!= (IF_SET|IF_ASK|IF_REAL))
		return 0;
	if (sg->is_dc &&
		(sg->ptable[sg->param].dataType & (IF_AC | IF_AC_ONLY)))
		return 0;
	if ((sg->ptable[sg->param].dataType & IF_CHKQUERY) && !sg->is_q)
		return 0;

	if (sens_getp(sg, sg->ckt, &ifval))
		return 0;

	if (fabs(ifval.rValue) < 1e-30) {
		if (sg->ptable[sg->param].dataType & IF_SETQUERY)
			sg->is_q = 0;

		if (!sg->is_zerook
			&& !(sg->ptable[sg->param].dataType & IF_PRINCIPAL))
			return 0;

	} else if (sg->ptable[sg->param].dataType & (IF_SETQUERY|IF_ORQUERY))
			sg->is_q = 1;

	if (sg->ptable[sg->param].dataType & IF_PRINCIPAL)
		sg->is_principle += 1;

	sg->value = ifval.rValue;

	return 1;
}

#ifdef notdef
void
sgen_suspend(sgen *sg)
{
	sg->devlist[sg->dev] = sg->first_model;
	sg->model->GENnextModel = sg->next_model;
	sg->instance->GENnextInstance = sg->next_instance;
	sg->model->GENinstances = sg->first_instance;
}

void
sgen_restore(sgen *sg)
{
	sg->devlist[sg->dev] = sg->model;
	sg->model->GENnextModel = NULL;
	sg->instance->GENnextInstance = NULL;
	sg->model->GENinstances = sg->instance;
}
#endif
