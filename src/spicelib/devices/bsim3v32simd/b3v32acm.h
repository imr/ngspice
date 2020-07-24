#ifndef BSIM3v32ACM_H
#define BSIM3v32ACM_H

int BSIM3v32_ACM_saturationCurrents
(
	BSIM3v32model *model,
	BSIM3v32instance *here,
        double *DrainSatCurrent,
        double *SourceSatCurrent
);

	    
int BSIM3v32_ACM_junctionCapacitances
(
	BSIM3v32model *model,
	BSIM3v32instance *here,
	double *areaDrainBulkCapacitance,
	double *periDrainBulkCapacitance,
	double *gateDrainBulkCapacitance,
	double *areaSourceBulkCapacitance,
	double *periSourceBulkCapacitance,
	double *gateSourceBulkCapacitance
);

#endif
