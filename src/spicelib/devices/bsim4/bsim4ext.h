/* ******************************************************************************
   *  BSIM4 4.8.2 released by Chetan Kumar Dabhi 01/01/2020                     *
   *  BSIM4 Model Equations                                                     *
   ******************************************************************************

   ******************************************************************************
   *  Copyright (c) 2020 University of California                               *
   *                                                                            *
   *  Project Director: Prof. Chenming Hu.                                      *
   *  Current developers: Chetan Kumar Dabhi   (Ph.D. student, IIT Kanpur)      *
   *                      Prof. Yogesh Chauhan (IIT Kanpur)                     *
   *                      Dr. Pragya Kushwaha  (Postdoc, UC Berkeley)           *
   *                      Dr. Avirup Dasgupta  (Postdoc, UC Berkeley)           *
   *                      Ming-Yen Kao         (Ph.D. student, UC Berkeley)     *
   *  Authors: Gary W. Ng, Weidong Liu, Xuemei Xi, Mohan Dunga, Wenwei Yang     *
   *           Ali Niknejad, Chetan Kumar Dabhi, Yogesh Singh Chauhan,          *
   *           Sayeef Salahuddin, Chenming Hu                                   * 
   ******************************************************************************/

/*
Licensed under Educational Community License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain a copy of the license at
http://opensource.org/licenses/ECL-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations
under the License.
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
