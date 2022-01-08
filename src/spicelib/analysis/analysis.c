#include "ngspice/ngspice.h"

#include "ngspice/jobdefs.h"
#include "ngspice/cktdefs.h"

#include "analysis.h"

extern SPICEanalysis OPTinfo;
extern SPICEanalysis ACinfo;
extern SPICEanalysis DCTinfo;
extern SPICEanalysis DCOinfo;
extern SPICEanalysis TRANinfo;
extern SPICEanalysis PZinfo;
extern SPICEanalysis TFinfo;
extern SPICEanalysis DISTOinfo;
extern SPICEanalysis NOISEinfo;
extern SPICEanalysis SENSinfo;

#ifdef RFSPICE
extern SPICEanalysis SPinfo;

#ifdef WITH_HB
extern SPICEanalysis HBinfo;
#endif

#endif

#ifdef WITH_PSS
extern SPICEanalysis PSSinfo;
#endif

#ifdef WANT_SENSE2
extern SPICEanalysis SEN2info;
#endif

SPICEanalysis *analInfo[] = {
    &OPTinfo,
    &ACinfo,
    &DCTinfo,
    &DCOinfo,
    &TRANinfo,
    &PZinfo,
    &TFinfo,
    &DISTOinfo,
    &NOISEinfo,
    &SENSinfo,
#ifdef WITH_PSS
    &PSSinfo,
#endif
#ifdef WANT_SENSE2
    &SEN2info,
#endif
#ifdef RFSPICE
    & SPinfo,
#ifdef WITH_HB
    & HBinfo,
#endif
#endif
};


char *spice_analysis_get_name(int index)
{
    return analInfo[index]->if_analysis.name;
}

char *spice_analysis_get_description(int index)
{
    return analInfo[index]->if_analysis.description;
}

int spice_num_analysis(void)
{
    return NUMELEMS(analInfo);
}

SPICEanalysis **spice_analysis_ptr(void)
{
    return analInfo;
}
