inline int BSIM3v32_ACM_saturationCurrents
(
	BSIM3v32model *model,
	BSIM3v32instance *here,
        double *DrainSatCurrent,
        double *SourceSatCurrent
)
{
            return ACM_saturationCurrents(
            model->BSIM3v32acmMod,
            model->BSIM3v32calcacm,
            here->BSIM3v32geo,
            model->BSIM3v32hdif,
            model->BSIM3v32wmlt,
            here->BSIM3v32w,
            model->BSIM3v32xw,
            model->BSIM3v32jctTempSatCurDensity,
            model->BSIM3v32jctSidewallTempSatCurDensity,
            here->BSIM3v32drainAreaGiven,
            here->BSIM3v32drainArea,
            here->BSIM3v32drainPerimeterGiven,
            here->BSIM3v32drainPerimeter,
            here->BSIM3v32sourceAreaGiven,
            here->BSIM3v32sourceArea,
            here->BSIM3v32sourcePerimeterGiven,
            here->BSIM3v32sourcePerimeter,
            DrainSatCurrent,
            SourceSatCurrent
            );
}

	    
inline int BSIM3v32_ACM_junctionCapacitances
(
	BSIM3v32model *model,
	BSIM3v32instance *here,
	double *areaDrainBulkCapacitance,
	double *periDrainBulkCapacitance,
	double *gateDrainBulkCapacitance,
	double *areaSourceBulkCapacitance,
	double *periSourceBulkCapacitance,
	double *gateSourceBulkCapacitance
)
{
	switch (model->BSIM3v32intVersion) {
                    case BSIM3v32V324:
                    case BSIM3v32V323:
		      return ACM_junctionCapacitances(
                      model->BSIM3v32acmMod,
                      model->BSIM3v32calcacm,
                      here->BSIM3v32geo,
                      model->BSIM3v32hdif,
                      model->BSIM3v32wmlt,
                      here->BSIM3v32w,
                      model->BSIM3v32xw,
                      here->BSIM3v32drainAreaGiven,
                      here->BSIM3v32drainArea,
                      here->BSIM3v32drainPerimeterGiven,
                      here->BSIM3v32drainPerimeter,
                      here->BSIM3v32sourceAreaGiven,
                      here->BSIM3v32sourceArea,
                      here->BSIM3v32sourcePerimeterGiven,
                      here->BSIM3v32sourcePerimeter,
                      model->BSIM3v32unitAreaTempJctCap,
                      model->BSIM3v32unitLengthSidewallTempJctCap,
                      model->BSIM3v32unitLengthGateSidewallTempJctCap,
                      areaDrainBulkCapacitance,
                      periDrainBulkCapacitance,
                      gateDrainBulkCapacitance,
                      areaSourceBulkCapacitance,
                      periSourceBulkCapacitance,
                      gateSourceBulkCapacitance
              	      );
		    case BSIM3v32V322:
                    case BSIM3v32V32:
                    default:
		      return ACM_junctionCapacitances(
                      model->BSIM3v32acmMod,
                      model->BSIM3v32calcacm,
                      here->BSIM3v32geo,
                      model->BSIM3v32hdif,
                      model->BSIM3v32wmlt,
                      here->BSIM3v32w,
                      model->BSIM3v32xw,
                      here->BSIM3v32drainAreaGiven,
                      here->BSIM3v32drainArea,
                      here->BSIM3v32drainPerimeterGiven,
                      here->BSIM3v32drainPerimeter,
                      here->BSIM3v32sourceAreaGiven,
                      here->BSIM3v32sourceArea,
                      here->BSIM3v32sourcePerimeterGiven,
                      here->BSIM3v32sourcePerimeter,
                      model->BSIM3v32unitAreaJctCap,
                      model->BSIM3v32unitLengthSidewallJctCap,
                      model->BSIM3v32unitLengthGateSidewallJctCap,
		      areaDrainBulkCapacitance,
                      periDrainBulkCapacitance,
                      gateDrainBulkCapacitance,
                      areaSourceBulkCapacitance,
                      periSourceBulkCapacitance,
                      gateSourceBulkCapacitance
              	      );
	}
}
