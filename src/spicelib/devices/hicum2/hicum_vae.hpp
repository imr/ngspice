#ifndef hicum_vae
#define hicum_vae
//this file contains a class definition for models to be integrated into ngspice
//later, it should be moved somewhere else.
//for the time beeing, HICUM is the first model to use this.
#ifdef __cplusplus
#include <unordered_map>
#include <vector>
#include <limits>
#include <model_class.hpp>
extern "C" {
#include "ngspice/cktdefs.h"
#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
}
class HICUML2 : public NGSpiceModel
{
    public:
    char * name          = "hl2";
    char * description   = "high-speed HBT model";        //the description of the model
    char * MODELnames[5] = {
        "collector",
        "base",
        "emitter",
        "substrate",
        "temp"
    };
    HICUML2();
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern int HICUMacLoad(GENmodel *,CKTcircuit*);
extern int HICUMask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int HICUMconvTest(GENmodel*,CKTcircuit*);
extern int HICUMdelete(GENinstance*);
extern int HICUMgetic(GENmodel*,CKTcircuit*);
extern int HICUMload(GENmodel*,CKTcircuit*);
extern int HICUMmAsk(CKTcircuit*,GENmodel*,int,IFvalue*);
extern int HICUMmParam(int,IFvalue*,GENmodel*);
extern int HICUMparam(int,IFvalue*,GENinstance*,IFvalue*);
extern int HICUMpzLoad(GENmodel*, CKTcircuit*, SPcomplex*);
extern int HICUMsetup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int HICUMunsetup(GENmodel*,CKTcircuit*);
extern int HICUMtemp(GENmodel*,CKTcircuit*);
extern int HICUMtrunc(GENmodel*,CKTcircuit*,double*);
extern int HICUMnoise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int HICUMsoaCheck(CKTcircuit *, GENmodel *);
extern SPICEdev * get_hicum_info(void);

#ifdef __cplusplus
}
#endif
#endif