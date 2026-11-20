#ifndef NG_KLU_PARALLEL_H
#define NG_KLU_PARALLEL_H
#include "ngspice/klu.h"
typedef struct ng_klu_plan NgKluPlan;
int ng_klu_refactor(NgKluPlan **, int *, int *, double *, klu_symbolic *, klu_numeric *, klu_common *, int);
void ng_klu_plan_free(NgKluPlan **);
#endif
