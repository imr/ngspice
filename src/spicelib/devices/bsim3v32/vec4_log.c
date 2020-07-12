/* This program implements a show-case vector (vectorizable) double
   precision logarithm with a 4 ulp error bound.

   Author: Christoph Lauter,

           Sorbonne Université - LIP6 - PEQUAN team.

   This program uses code generated using Sollya and Metalibm; see the
   licences and exception texts below.

   This program is

   Copyright 2014-2018 Christoph Lauter Sorbonne Université

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

   3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
   OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/*

    This code was generated using non-trivial code generation commands
    of the Metalibm software program.

    Before using, modifying and/or integrating this code into other
    software, review the copyright and license status of this
    generated code. In particular, see the exception below.

    This generated program is partly or entirely based on a program
    generated using non-trivial code generation commands of the Sollya
    software program. See the copyright notice and exception text
    referring to that Sollya-generated part of this program generated
    with Metalibm below.

    Metalibm is

    Copyright 2008-2013 by

    Laboratoire de l'Informatique du Parallélisme,
    UMR CNRS - ENS Lyon - UCB Lyon 1 - INRIA 5668

    and by

    Laboratoire d'Informatique de Paris 6, equipe PEQUAN,
    UPMC Universite Paris 06 - CNRS - UMR 7606 - LIP6, Paris, France.

    Contributors: Christoph Quirin Lauter
                  (UPMC LIP6 PEQUAN formerly LIP/ENS Lyon)
                  christoph.lauter@lip6.fr

		  and

		  Olga Kupriianova
		  (UPMC LIP6 PEQUAN)
		  olga.kupriianova@lip6.fr

    Metalibm was formerly developed by the Arenaire project at Ecole
    Normale Superieure de Lyon and is now developed by Equipe PEQUAN
    at Universite Pierre et Marie Curie Paris 6.

    The Metalibm software program is free software; you can
    redistribute it and/or modify it under the terms of the GNU Lesser
    General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option)
    any later version.

    Metalibm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with the Metalibm program; if not, write to the Free
    Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
    02111-1307, USA.

    This generated program is distributed WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE.

    As a special exception, you may create a larger work that contains
    part or all of this software generated using Metalibm and
    distribute that work under terms of your choice, so long as that
    work isn't itself a numerical code generator using the skeleton of
    this code or a modified version thereof as a code skeleton.
    Alternatively, if you modify or redistribute this generated code
    itself, or its skeleton, you may (at your option) remove this
    special exception, which will cause this generated code and its
    skeleton and the resulting Metalibm output files to be licensed
    under the General Public licence (version 2) without this special
    exception.

    This special exception was added by the Metalibm copyright holders
    on November 20th 2013.

*/



/*
    This code was generated using non-trivial code generation commands of
    the Sollya software program.

    Before using, modifying and/or integrating this code into other
    software, review the copyright and license status of this generated
    code. In particular, see the exception below.

    Sollya is

    Copyright 2006-2013 by

    Laboratoire de l'Informatique du Parallelisme, UMR CNRS - ENS Lyon -
    UCB Lyon 1 - INRIA 5668,

    Laboratoire d'Informatique de Paris 6, equipe PEQUAN, UPMC Universite
    Paris 06 - CNRS - UMR 7606 - LIP6, Paris, France

    and by

    Centre de recherche INRIA Sophia-Antipolis Mediterranee, equipe APICS,
    Sophia Antipolis, France.

    Contributors Ch. Lauter, S. Chevillard, M. Joldes

    christoph.lauter@ens-lyon.org
    sylvain.chevillard@ens-lyon.org
    joldes@lass.fr

    The Sollya software is a computer program whose purpose is to provide
    an environment for safe floating-point code development. It is
    particularily targeted to the automatized implementation of
    mathematical floating-point libraries (libm). Amongst other features,
    it offers a certified infinity norm, an automatic polynomial
    implementer and a fast Remez algorithm.

    The Sollya software is governed by the CeCILL-C license under French
    law and abiding by the rules of distribution of free software.  You
    can use, modify and/ or redistribute the software under the terms of
    the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
    following URL "http://www.cecill.info".

    As a counterpart to the access to the source code and rights to copy,
    modify and redistribute granted by the license, users are provided
    only with a limited warranty and the software's author, the holder of
    the economic rights, and the successive licensors have only limited
    liability.

    In this respect, the user's attention is drawn to the risks associated
    with loading, using, modifying and/or developing or reproducing the
    software by the user in light of its specific status of free software,
    that may mean that it is complicated to manipulate, and that also
    therefore means that it is reserved for developers and experienced
    professionals having in-depth computer knowledge. Users are therefore
    encouraged to load and test the software's suitability as regards
    their requirements in conditions enabling the security of their
    systems and/or data to be ensured and, more generally, to use and
    operate it in the same conditions as regards security.

    The fact that you are presently reading this means that you have had
    knowledge of the CeCILL-C license and that you accept its terms.

    The Sollya program is distributed WITHOUT ANY WARRANTY; without even
    the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    PURPOSE.

    This generated program is distributed WITHOUT ANY WARRANTY; without
    even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE.

    As a special exception, you may create a larger work that contains
    part or all of this software generated using Sollya and distribute
    that work under terms of your choice, so long as that work isn't
    itself a numerical code generator using the skeleton of this code or a
    modified version thereof as a code skeleton.  Alternatively, if you
    modify or redistribute this generated code itself, or its skeleton,
    you may (at your option) remove this special exception, which will
    cause this generated code and its skeleton and the resulting Sollya
    output files to be licensed under the CeCILL-C licence without this
    special exception.

    This special exception was added by the Sollya copyright holders in
    version 4.1 of Sollya.

*/

/* Modified by Florian Ballenegger, Anamosic Ballenegger Design, 2020.
Use of gcc vector extensions instead of pointers to double */

#include <stdint.h>

/* Two caster types */
typedef union _dblcast {
  double   d;
  uint64_t i;
} dblcast;

typedef union {
  int64_t l;
  double d;
} db_number;

/* Compiler tricks and hints */
#define INLINE inline
#define RESTRICT restrict
#define STATIC static
#define CONST const

/* Vector length */
#define VECTOR_LENGTH 4
typedef double Vec4d __attribute__ ((vector_size (sizeof(double)*VECTOR_LENGTH),
 aligned (sizeof(double)*VECTOR_LENGTH)));

/* Macro implementations of some double-double operations */
#define Add12(s, r, a, b)                       \
  {double _z, _a=a, _b=b;                       \
    s = _a + _b;                                \
    _z = s - _a;                                \
    r = _b - _z;   }

#define Mul22(zh,zl,xh,xl,yh,yl)                        \
  {                                                     \
    double mh, ml;                                      \
                                                        \
    const double c = 134217729.;                        \
    double up, u1, u2, vp, v1, v2;                      \
                                                        \
    up = (xh)*c;        vp = (yh)*c;                    \
    u1 = ((xh)-up)+up;  v1 = ((yh)-vp)+vp;              \
    u2 = (xh)-u1;       v2 = (yh)-v1;                   \
                                                        \
    mh = (xh)*(yh);                                     \
    ml = (((u1*v1-mh)+(u1*v2))+(u2*v1))+(u2*v2);        \
                                                        \
    ml += (xh)*(yl) + (xl)*(yh);                        \
    *zh = mh+ml;                                        \
    *zl = mh - (*zh) + ml;                              \
  }

#define Mul122(resh,resl,a,bh,bl)               \
  {                                             \
    double _t1, _t2, _t3, _t4;                  \
                                                \
    Mul12(&_t1,&_t2,(a),(bh));                  \
    _t3 = (a) * (bl);                           \
    _t4 = _t2 + _t3;                            \
    Add12((*(resh)),(*(resl)),_t1,_t4);         \
  }

#define Add22(zh,zl,xh,xl,yh,yl)                \
  do {                                          \
    double _r,_s;                               \
    _r = (xh)+(yh);                             \
    _s = ((((xh)-_r) +(yh)) + (yl)) + (xl);     \
    *zh = _r+_s;                                \
    *zl = (_r - (*zh)) + _s;                    \
  } while(0)

#define Mul12(rh,rl,u,v)                                \
  {                                                     \
    const double c  = 134217729.; /* 2^27 +1 */         \
    double up, u1, u2, vp, v1, v2;                      \
    double _u =u, _v=v;                                 \
                                                        \
    up = _u*c;        vp = _v*c;                        \
    u1 = (_u-up)+up;  v1 = (_v-vp)+vp;                  \
    u2 = _u-u1;       v2 = _v-v1;                       \
                                                        \
    *rh = _u*_v;                                        \
    *rl = (((u1*v1-*rh)+(u1*v2))+(u2*v1))+(u2*v2);      \
  }


/* Need fabs */
double fabs(double);


/* Some constants */

#define LOG_TWO_HI 0.693147180559890330187045037746429443359375
#define LOG_TWO_LO 5.4979230187083711552420206887059365096458163346682e-14


/* A metalibm generated function for the callout */
#define f_approx_log_arg_red_coeff_1h 1.00000000000000000000000000000000000000000000000000000000000000000000000000000000e+00
#define f_approx_log_arg_red_coeff_2h -4.99999999999998390176614293523016385734081268310546875000000000000000000000000000e-01
#define f_approx_log_arg_red_coeff_3h 3.33333333333923731434111914495588280260562896728515625000000000000000000000000000e-01
#define f_approx_log_arg_red_coeff_4h -2.50000000052116866378071335930144414305686950683593750000000000000000000000000000e-01
#define f_approx_log_arg_red_coeff_5h 1.99999988486698782041983690760389436036348342895507812500000000000000000000000000e-01
#define f_approx_log_arg_red_coeff_6h -1.66666258081627438603078417145297862589359283447265625000000000000000000000000000e-01
#define f_approx_log_arg_red_coeff_7h 1.42921894210221167575980416586389765143394470214843750000000000000000000000000000e-01
#define f_approx_log_arg_red_coeff_8h -1.25915254741829296669664017827017232775688171386718750000000000000000000000000000e-01


STATIC INLINE void f_approx_log_arg_red(double * RESTRICT f_approx_log_arg_red_resh, double * RESTRICT f_approx_log_arg_red_resm, double xh, double xm) {




  double f_approx_log_arg_red_t_1_0h;
  double f_approx_log_arg_red_t_2_0h;
  double f_approx_log_arg_red_t_3_0h;
  double f_approx_log_arg_red_t_4_0h;
  double f_approx_log_arg_red_t_5_0h;
  double f_approx_log_arg_red_t_6_0h;
  double f_approx_log_arg_red_t_7_0h;
  double f_approx_log_arg_red_t_8_0h;
  double f_approx_log_arg_red_t_9_0h;
  double f_approx_log_arg_red_t_10_0h;
  double f_approx_log_arg_red_t_11_0h;
  double f_approx_log_arg_red_t_12_0h;
  double f_approx_log_arg_red_t_13_0h;
  double f_approx_log_arg_red_t_14_0h;
  double f_approx_log_arg_red_t_15_0h, f_approx_log_arg_red_t_15_0m;
  double f_approx_log_arg_red_t_16_0h, f_approx_log_arg_red_t_16_0m;
 


  f_approx_log_arg_red_t_1_0h = f_approx_log_arg_red_coeff_8h;
  f_approx_log_arg_red_t_2_0h = f_approx_log_arg_red_t_1_0h * xh;
  f_approx_log_arg_red_t_3_0h = f_approx_log_arg_red_coeff_7h + f_approx_log_arg_red_t_2_0h;
  f_approx_log_arg_red_t_4_0h = f_approx_log_arg_red_t_3_0h * xh;
  f_approx_log_arg_red_t_5_0h = f_approx_log_arg_red_coeff_6h + f_approx_log_arg_red_t_4_0h;
  f_approx_log_arg_red_t_6_0h = f_approx_log_arg_red_t_5_0h * xh;
  f_approx_log_arg_red_t_7_0h = f_approx_log_arg_red_coeff_5h + f_approx_log_arg_red_t_6_0h;
  f_approx_log_arg_red_t_8_0h = f_approx_log_arg_red_t_7_0h * xh;
  f_approx_log_arg_red_t_9_0h = f_approx_log_arg_red_coeff_4h + f_approx_log_arg_red_t_8_0h;
  f_approx_log_arg_red_t_10_0h = f_approx_log_arg_red_t_9_0h * xh;
  f_approx_log_arg_red_t_11_0h = f_approx_log_arg_red_coeff_3h + f_approx_log_arg_red_t_10_0h;
  f_approx_log_arg_red_t_12_0h = f_approx_log_arg_red_t_11_0h * xh;
  f_approx_log_arg_red_t_13_0h = f_approx_log_arg_red_coeff_2h + f_approx_log_arg_red_t_12_0h;
  f_approx_log_arg_red_t_14_0h = f_approx_log_arg_red_t_13_0h * xh;
  Add12(f_approx_log_arg_red_t_15_0h,f_approx_log_arg_red_t_15_0m,f_approx_log_arg_red_coeff_1h,f_approx_log_arg_red_t_14_0h);
  Mul22(&f_approx_log_arg_red_t_16_0h,&f_approx_log_arg_red_t_16_0m,f_approx_log_arg_red_t_15_0h,f_approx_log_arg_red_t_15_0m,xh,xm);
  *f_approx_log_arg_red_resh = f_approx_log_arg_red_t_16_0h; *f_approx_log_arg_red_resm = f_approx_log_arg_red_t_16_0m;


}


#define f_approx_tablewidth 5
#define f_approx_maxindex 14
#define f_approx_rcpr_log_two_of_base_hi 0.69314718055994528622676398299518041312694549560546875
#define f_approx_rcpr_log_two_of_base_mi 2.3190468138462995584177710797133615750739959242786823734316925538223586045205593109130859375e-17

static const double f_approx_log_rcpr_tbl_hi[33] = {
  0,
  3.17486983145802981187699742804397828876972198486328125e-2,
  6.453852113757117814341057737692608498036861419677734375e-2,
  8.985632912186104770402295116582536138594150543212890625e-2,
  0.1158318155251217007606356901305844075977802276611328125,
  0.142500062607283040083672176479012705385684967041015625,
  0.169899036795397473387225772967212833464145660400390625,
  0.1980699137620937910764240541539038531482219696044921875,
  0.2270574506353460753071971112149185501039028167724609375,
  0.2468600779315257842672082233548280782997608184814453125,
  0.2670627852490452536216025691828690469264984130859375,
  0.2981533723190763485177967595518566668033599853515625,
  0.319430770766361227241958431477542035281658172607421875,
  0.34117075740276714412857472780160605907440185546875,
  -0.329753286372467979692402195723843760788440704345703125,
  -0.30702503529491187439504074063734151422977447509765625,
  -0.28376817313064461867355703361681662499904632568359375,
  -0.27193371548364175804834985683555714786052703857421875,
  -0.2478361639045812692128123444490483961999416351318359375,
  -0.223143551314209764857565687634632922708988189697265625,
  -0.2105647691073496419189581274622469209134578704833984375,
  -0.1849223384940119896402421773018431849777698516845703125,
  -0.17185025692665922836255276706651784479618072509765625,
  -0.1451820098444978890395162807180895470082759857177734375,
  -0.1315763577887192614657152489598956890404224395751953125,
  -0.1177830356563834557359626842298894189298152923583984375,
  -0.10379679368164355934833764649738441221415996551513671875,
  -7.522342123758753162920953627690323628485202789306640625e-2,
  -6.062462181643483993820353816772694699466228485107421875e-2,
  -4.58095360312942012637194011404062621295452117919921875e-2,
  -3.077165866675368732785500469617545604705810546875e-2,
  -1.5504186535965254478686148331689764745533466339111328125e-2,
  0
};


static const double f_approx_log_rcpr_tbl_mi[33] = {
  0,
  3.03822630846808578765259986229142635550407126467467068542394059704747633077204227447509765625e-18,
  -6.4704866616929329974161813916713618427728286285169519154170103547585313208401203155517578125e-18,
  -6.2737601636895940223772151595043522169967894903434509935868934604741298244334757328033447265625e-19,
  4.33848436980809595557198228135728192959103146527353490891076859270469867624342441558837890625e-18,
  -9.9263882342257491397106905651454915981827472977916566876377402195430477149784564971923828125e-18,
  -4.8680087644390707941393631766999763543363602831990049994714819803220962057821452617645263671875e-19,
  3.74284348246143901356926696786621497402944711010920782190414257684096810407936573028564453125e-18,
  9.551415762738488431492098722158984238118586922020904206309666051311069168150424957275390625e-18,
  1.3617433717483680171009009478499574446783469284919833308666881066528731025755405426025390625e-17,
  -7.3289153273201694886198949831953541788954485227476805253576941368010011501610279083251953125e-18,
  -1.72069586744586603715170366469832022772114935873187524517646806998527608811855316162109375e-17,
  1.3542568572648110745997524461078410815028703905694095442624469427528310916386544704437255859375e-18,
  -1.936679006260286699473802044740827141118261398825169117277056329839979298412799835205078125e-17,
  -2.122020616196946023332814001844389995179410458238009572207172226399052306078374385833740234375e-18,
  1.231991620010196428468632499036271595368677926845939196720536301654647104442119598388671875e-17,
  2.0326655811266561230291019136542876238402571524729010865595313362064189277589321136474609375e-17,
  -7.8331963769744201243220009945333356568337002449775477268267831476578066940419375896453857421875e-19,
  1.24322095787025231818185093190325423423584424116919953939852661051190807484090328216552734375e-17,
  9.091270597324799048711045191818233254271755021066504787174977764152572490274906158447265625e-18,
  4.24940531472989532850360049655226441340213720053550945643383585093033616431057453155517578125e-18,
  -3.023661415357406426577090417003710240867302228907377570354952922571101225912570953369140625e-18,
  6.0224538210113704760318352588172818979944380808860641962620974254605243913829326629638671875e-18,
  -8.2424187830224753896228153425798328521705177161500548155270706729425000958144664764404296875e-18,
  -1.112300087972958802991298461231701795529693224825161512736571012283093295991420745849609375e-17,
  1.197168574759367729935408317875380291366461975031726568119427867031845380552113056182861328125e-18,
  -5.47772415726659012592706002045618002605660904524354816447218041730593540705740451812744140625e-18,
  5.93060419629324071708218111258442537327230935598090626192924190718258614651858806610107421875e-18,
  -2.642402593872693418157455274069099088532417945381102798718675472855466068722307682037353515625e-18,
  -1.90295986647425706325531188416869176372485943199669260195161513138373265974223613739013671875e-18,
  -1.0431732029005967805059792190367890366163673586242621564579291515428849379532039165496826171875e-18,
  3.27832102289242912962985506573138544887782756899054594813824881072150674299336969852447509765625e-19,
  0
};


static const double f_approx_rcpr_tbl[33] = {
  1.0,
  0.96875,
  0.9375,
  0.9140625,
  0.890625,
  0.8671875,
  0.84375,
  0.8203125,
  0.796875,
  0.78125,
  0.765625,
  0.7421875,
  0.7265625,
  0.7109375,
  0.6953125,
  0.6796875,
  0.6640625,
  0.65625,
  0.640625,
  0.625,
  0.6171875,
  0.6015625,
  0.59375,
  0.578125,
  0.5703125,
  0.5625,
  0.5546875,
  0.5390625,
  0.53125,
  0.5234375,
  0.515625,
  0.5078125,
  0.5
};

STATIC INLINE void scalar_log_callout_inner(double * RESTRICT res_resh, double * RESTRICT res_resm, double xh) {

  db_number argRedCaster;
  int E;
  int index;
  double ed;
  double m;
  double r;
  double zh;
  double zm;
  double mrh, mrl;
  double temp;
  double polyHi;
  double polyMi;
  double tableHi;
  double tableMi;
  double scaledExpoHi;
  double scaledExpoMi;
  double logMHi;
  double logMMi;


  argRedCaster.d = xh;
  E = 0;
  if (argRedCaster.l < 0x0010000000000000) {
    argRedCaster.d *= 9007199254740992.0;
    E -= 53;
  }
  E += (int) ((argRedCaster.l >> 52) - 1023ll);
  index = (int) ((argRedCaster.l & 0x000fffffffffffffull) >> (52 - f_approx_tablewidth - 1));
  index = (index + 1) >> 1;
  if (index >= f_approx_maxindex) E++;
  ed = (double) E;
  argRedCaster.l = (argRedCaster.l & 0x800fffffffffffffull) | 0x3ff0000000000000ull;
  m = argRedCaster.d;
  r = f_approx_rcpr_tbl[index];
  Mul12(&mrh,&mrl,m,r);
  temp = mrh - 1.0;
  Add12(zh,zm,temp,mrl);

  f_approx_log_arg_red(&polyHi, &polyMi, zh, zm);


  tableHi = f_approx_log_rcpr_tbl_hi[index];
  tableMi = f_approx_log_rcpr_tbl_mi[index];
  Mul122(&scaledExpoHi,&scaledExpoMi,ed,f_approx_rcpr_log_two_of_base_hi,f_approx_rcpr_log_two_of_base_mi);
  Add22(&logMHi,&logMMi,tableHi,tableMi,polyHi,polyMi);
  Add22(res_resh,res_resm,scaledExpoHi,scaledExpoMi,logMHi,logMMi);


}


/* A scalar logarithm for the callout */
STATIC INLINE double scalar_log_callout(double x) {
  dblcast xdb;
  double yh, yl;
  double temp;

  /* Check for special inputs: x less than the smallest positive
     subnormal, x Inf or NaN 
  */
  xdb.d = x;
  if ((xdb.i == 0x0ull) || (xdb.i >= 0x7ff0000000000000ull)) {
    /* Here, we have a special case to handle 

       The input is either +/-0, negative, +/-Inf or +/- NaN.

    */
    if ((xdb.i & 0x7fffffffffffffffull) >= 0x7ff0000000000000ull) {
      /* The input is either Inf or NaN */
      if ((xdb.i & 0x7fffffffffffffffull) > 0x7ff0000000000000ull) {
	/* The input is NaN. Return the quietized NaN */
	return 1.0 + x;
      }
      /* The input is +Inf or -Inf */
      if ((xdb.i & 0x8000000000000000ull) == 0x0ull) {
	/* The input is +Inf. Return log(+Inf) = + Inf. */
	return x;
      }
      /* The input is -Inf. Let the case fall through */
    }

    /* The input is +/- 0, -Inf or a negative real number */
    if (x == 0.0) {
      /* The input is +/-0. Return -Inf and raise the division-by-zero
	 exception.
      */
      temp = 1.0 - 1.0;   /* temp = +0.0 or -0.0 */
      temp = temp * temp; /* temp = +0.0 */
      return -1.0 / temp; /* Return -Inf and raise div-by-zero. */
    }

    /* The input is -Inf or a negative real number.

       Return NaN and raise the invalid exception.

    */
    temp = 0.0;
    return temp / temp; /* Return NaN and raise invalid. */
  }

  /* Here the input is a positive subnormal or normal 

     Just call a Metalibm generated function.
 
  */
  scalar_log_callout_inner(&yh, &yl, x);

  /* Return the result */
  return yh + yl;
}

/* A vector logarithm callout */
STATIC INLINE Vec4d vector_log_callout(Vec4d x) {
  int i;
  Vec4d y;
  for (i=0;i<VECTOR_LENGTH;i++) {
    y[i] = scalar_log_callout(x[i]);
  }
  return y;
}

/* Generated polynomial for vector logarithm */
#define vector_log_poly_coeff_1h 1.00000000000000000000000000000000000000000000000000000000000000000000000000000000e+00
#define vector_log_poly_coeff_2h -5.00000000000000999200722162640886381268501281738281250000000000000000000000000000e-01
#define vector_log_poly_coeff_3h 3.33333333333384995711412557284347712993621826171875000000000000000000000000000000e-01
#define vector_log_poly_coeff_4h -2.49999999999541949735615276040334720164537429809570312500000000000000000000000000e-01
#define vector_log_poly_coeff_5h 1.99999999982921977670358160139585379511117935180664062500000000000000000000000000e-01
#define vector_log_poly_coeff_6h -1.66666666708135652319455743963771965354681015014648437500000000000000000000000000e-01
#define vector_log_poly_coeff_7h 1.42857144801517760290821001945005264133214950561523437500000000000000000000000000e-01
#define vector_log_poly_coeff_8h -1.25000000676456918258239170427259523421525955200195312500000000000000000000000000e-01
#define vector_log_poly_coeff_9h 1.11111007470194977919675238808849826455116271972656250000000000000000000000000000e-01
#define vector_log_poly_coeff_10h -9.99997732686361273657382753299316391348838806152343750000000000000000000000000000e-02
#define vector_log_poly_coeff_11h 9.09118368248343633464259028187370859086513519287109375000000000000000000000000000e-02
#define vector_log_poly_coeff_12h -8.33440688797140172283661740948446094989776611328125000000000000000000000000000000e-02
#define vector_log_poly_coeff_13h 7.68928106123701327057062826497713103890419006347656250000000000000000000000000000e-02
#define vector_log_poly_coeff_14h -7.12109533797148086531336730331531725823879241943359375000000000000000000000000000e-02
#define vector_log_poly_coeff_15h 6.65850051807088672006784690893255174160003662109375000000000000000000000000000000e-02
#define vector_log_poly_coeff_16h -6.43233317758114681028658310424361843615770339965820312500000000000000000000000000e-02
#define vector_log_poly_coeff_17h 6.31209736682013661246415381356200668960809707641601562500000000000000000000000000e-02
#define vector_log_poly_coeff_18h -5.44324247927492413379191305011772783473134040832519531250000000000000000000000000e-02
#define vector_log_poly_coeff_19h 3.23620871610351343306000160282565047964453697204589843750000000000000000000000000e-02
#define vector_log_poly_coeff_20h -9.16877113215055876416226254832508857361972332000732421875000000000000000000000000e-03


STATIC void vector_log_poly(double * RESTRICT vector_log_poly_resh, double x) {




  double vector_log_poly_t_1_0h;
  double vector_log_poly_t_2_0h;
  double vector_log_poly_t_3_0h;
  double vector_log_poly_t_4_0h;
  double vector_log_poly_t_5_0h;
  double vector_log_poly_t_6_0h;
  double vector_log_poly_t_7_0h;
  double vector_log_poly_t_8_0h;
  double vector_log_poly_t_9_0h;
  double vector_log_poly_t_10_0h;
  double vector_log_poly_t_11_0h;
  double vector_log_poly_t_12_0h;
  double vector_log_poly_t_13_0h;
  double vector_log_poly_t_14_0h;
  double vector_log_poly_t_15_0h;
  double vector_log_poly_t_16_0h;
  double vector_log_poly_t_17_0h;
  double vector_log_poly_t_18_0h;
  double vector_log_poly_t_19_0h;
  double vector_log_poly_t_20_0h;
  double vector_log_poly_t_21_0h;
  double vector_log_poly_t_22_0h;
  double vector_log_poly_t_23_0h;
  double vector_log_poly_t_24_0h;
  double vector_log_poly_t_25_0h;
  double vector_log_poly_t_26_0h;
  double vector_log_poly_t_27_0h;
  double vector_log_poly_t_28_0h;
  double vector_log_poly_t_29_0h;
  double vector_log_poly_t_30_0h;
  double vector_log_poly_t_31_0h;
  double vector_log_poly_t_32_0h;
  double vector_log_poly_t_33_0h;
  double vector_log_poly_t_34_0h;
  double vector_log_poly_t_35_0h;
  double vector_log_poly_t_36_0h;
  double vector_log_poly_t_37_0h;
  double vector_log_poly_t_38_0h;
  double vector_log_poly_t_39_0h;
  double vector_log_poly_t_40_0h;
 


  vector_log_poly_t_1_0h = vector_log_poly_coeff_20h;
  vector_log_poly_t_2_0h = vector_log_poly_t_1_0h * x;
  vector_log_poly_t_3_0h = vector_log_poly_coeff_19h + vector_log_poly_t_2_0h;
  vector_log_poly_t_4_0h = vector_log_poly_t_3_0h * x;
  vector_log_poly_t_5_0h = vector_log_poly_coeff_18h + vector_log_poly_t_4_0h;
  vector_log_poly_t_6_0h = vector_log_poly_t_5_0h * x;
  vector_log_poly_t_7_0h = vector_log_poly_coeff_17h + vector_log_poly_t_6_0h;
  vector_log_poly_t_8_0h = vector_log_poly_t_7_0h * x;
  vector_log_poly_t_9_0h = vector_log_poly_coeff_16h + vector_log_poly_t_8_0h;
  vector_log_poly_t_10_0h = vector_log_poly_t_9_0h * x;
  vector_log_poly_t_11_0h = vector_log_poly_coeff_15h + vector_log_poly_t_10_0h;
  vector_log_poly_t_12_0h = vector_log_poly_t_11_0h * x;
  vector_log_poly_t_13_0h = vector_log_poly_coeff_14h + vector_log_poly_t_12_0h;
  vector_log_poly_t_14_0h = vector_log_poly_t_13_0h * x;
  vector_log_poly_t_15_0h = vector_log_poly_coeff_13h + vector_log_poly_t_14_0h;
  vector_log_poly_t_16_0h = vector_log_poly_t_15_0h * x;
  vector_log_poly_t_17_0h = vector_log_poly_coeff_12h + vector_log_poly_t_16_0h;
  vector_log_poly_t_18_0h = vector_log_poly_t_17_0h * x;
  vector_log_poly_t_19_0h = vector_log_poly_coeff_11h + vector_log_poly_t_18_0h;
  vector_log_poly_t_20_0h = vector_log_poly_t_19_0h * x;
  vector_log_poly_t_21_0h = vector_log_poly_coeff_10h + vector_log_poly_t_20_0h;
  vector_log_poly_t_22_0h = vector_log_poly_t_21_0h * x;
  vector_log_poly_t_23_0h = vector_log_poly_coeff_9h + vector_log_poly_t_22_0h;
  vector_log_poly_t_24_0h = vector_log_poly_t_23_0h * x;
  vector_log_poly_t_25_0h = vector_log_poly_coeff_8h + vector_log_poly_t_24_0h;
  vector_log_poly_t_26_0h = vector_log_poly_t_25_0h * x;
  vector_log_poly_t_27_0h = vector_log_poly_coeff_7h + vector_log_poly_t_26_0h;
  vector_log_poly_t_28_0h = vector_log_poly_t_27_0h * x;
  vector_log_poly_t_29_0h = vector_log_poly_coeff_6h + vector_log_poly_t_28_0h;
  vector_log_poly_t_30_0h = vector_log_poly_t_29_0h * x;
  vector_log_poly_t_31_0h = vector_log_poly_coeff_5h + vector_log_poly_t_30_0h;
  vector_log_poly_t_32_0h = vector_log_poly_t_31_0h * x;
  vector_log_poly_t_33_0h = vector_log_poly_coeff_4h + vector_log_poly_t_32_0h;
  vector_log_poly_t_34_0h = vector_log_poly_t_33_0h * x;
  vector_log_poly_t_35_0h = vector_log_poly_coeff_3h + vector_log_poly_t_34_0h;
  vector_log_poly_t_36_0h = vector_log_poly_t_35_0h * x;
  vector_log_poly_t_37_0h = vector_log_poly_coeff_2h + vector_log_poly_t_36_0h;
  vector_log_poly_t_38_0h = vector_log_poly_t_37_0h * x;
  vector_log_poly_t_39_0h = vector_log_poly_coeff_1h + vector_log_poly_t_38_0h;
  vector_log_poly_t_40_0h = vector_log_poly_t_39_0h * x;
  *vector_log_poly_resh = vector_log_poly_t_40_0h;


}

/* A vector logarithm */
Vec4d vec4_log_vectorlibm(Vec4d x) {
  int i;
  int okaySlots;
  Vec4d y;
  dblcast xdb;
  uint64_t tui1, tui2, tui3;
  int E;
  double eDouble, m, r, p, elog2h, elog2l;
  double t1h, t1l, t2, t3;

  /* Check if we can handle all inputs */
  okaySlots = 0;
  for (i=0;i<VECTOR_LENGTH;i++) {
    xdb.d = x[i];
    okaySlots += ((xdb.i >= 0x0020000000000000ull) && (xdb.i < 0x7ff0000000000000ull));
  }

  /* Perform a callout if we cannot handle the input in one slot */
  if (okaySlots != VECTOR_LENGTH) {
    return vector_log_callout(x);
  }

  /* Here, the input is real, and far enough from the subnormal
     range
  */
  for (i=0;i<VECTOR_LENGTH;i++) {
    xdb.d = x[i];
    tui1 = xdb.i;
    tui2 = tui1 + 0x0008000000000000ull;
    tui1 >>= 52;
    tui2 >>= 52;
    tui3 = tui2 - tui1;
    tui3 <<= 52;
    E = ((int) tui2) - 1023;
    eDouble = (double) E;
    xdb.i = ((xdb.i & 0x000fffffffffffffull) | 0x3ff0000000000000ull) - tui3;
    m = xdb.d;                     /* 2^E * m = x exactly, 0.75 <= m < 1.5 */
    r = m - 1.0;                   /* exact: Sterbenz */
    vector_log_poly(&p, r);
    elog2h = eDouble * LOG_TWO_HI; /* exact: trailing zeros */
    elog2l = eDouble * LOG_TWO_LO;
    t1h = elog2h + p;
    t2 = t1h - elog2h;
    t1l = p - t2;                  /* exact: t1h + t1l = elog2h + p */
    t3 = elog2l + t1l;
    y[i] = t1h + t3; 
  }
  return y;
}
