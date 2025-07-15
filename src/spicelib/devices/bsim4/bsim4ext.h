/* ******************************************************************************
   *  BSIM4 4.8.3 released on 05/19/2025                                        *
   *  BSIM4 Model Equations                                                     *
   ******************************************************************************

   ******************************************************************************
   *  Copyright (c) 2025 University of California                               *
   *                                                                            *
   *  Project Directors: Prof. Sayeef Salahuddin and Prof. Chenming Hu          *
   *  Developers list: https://www.bsim.berkeley.edu/models/bsim4/auth_bsim4/   *
   ******************************************************************************/

/*
Licensed under Educational Community License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain a copy of the license at
http://opensource.org/licenses/ECL-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations
under the License.

BSIM4 model is supported by the members of Silicon Integration Initiative's Compact Model Coalition. A link to the most recent version of this
standard can be found at: http://www.si2.org/cmc
*/

extern int BSIM4acLoad(GENmodel *,CKTcircuit*);
extern int BSIM4ask(CKTcircuit *,GENinstance*,int,IFvalue*,IFvalue*);
extern int BSIM4convTest(GENmodel *,CKTcircuit*);
extern int BSIM4getic(GENmodel*,CKTcircuit*);
extern int BSIM4load(GENmodel*,CKTcircuit*);
extern int BSIM4mAsk(CKTcircuit*,GENmodel *,int, IFvalue*);
extern int BSIM4mDelete(GENmodel*);
extern int BSIM4mParam(int,IFvalue*,GENmodel*);
extern void BSIM4mosCap(CKTcircuit*, double, double, double, double,
        double, double, double, double, double, double, double,
        double, double, double, double, double, double, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double*, double*,
        double*);
extern int BSIM4param(int,IFvalue*,GENinstance*,IFvalue*);
extern int BSIM4pzLoad(GENmodel*,CKTcircuit*,SPcomplex*);
extern int BSIM4setup(SMPmatrix*,GENmodel*,CKTcircuit*,int*);
extern int BSIM4temp(GENmodel*,CKTcircuit*);
extern int BSIM4trunc(GENmodel*,CKTcircuit*,double*);
extern int BSIM4noise(int,int,GENmodel*,CKTcircuit*,Ndata*,double*);
extern int BSIM4unsetup(GENmodel*,CKTcircuit*);
extern int BSIM4soaCheck(CKTcircuit *, GENmodel *);

#ifdef KLU
extern int BSIM4bindCSC (GENmodel*, CKTcircuit*) ;
extern int BSIM4bindCSCComplex (GENmodel*, CKTcircuit*) ;
extern int BSIM4bindCSCComplexToReal (GENmodel*, CKTcircuit*) ;
#endif
