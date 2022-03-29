
#include <math.h>

 inline void setcplx(cplx* d, double r, double i);
 inline void cmultc(cplx* res, cplx a, cplx b);
 inline cplx cmultco(cplx a, cplx b);
 inline cplx cmultdo(cplx a, double d);
 inline void cmultd(cplx* res, cplx a, double d);
 inline void caddc(cplx* res, cplx a, cplx b);
 inline cplx caddco(cplx a, cplx b);
 inline void caddd(cplx* res, cplx a, double d);
 inline void csubc(cplx* res, cplx a, cplx b);
 inline cplx csubco(cplx a, cplx b);
 inline void csubd(cplx* res, cplx a, double d);
 inline double cmodinv(cplx a);
 inline double cmodsqr(cplx a);

 inline int ciszero(cplx a);
 inline cplx cinv(cplx a);
 inline cplx conju(cplx a);

 inline double cmodu(cplx a);
 inline cplx cdivco(cplx a, cplx b);

 inline void setcplx(cplx* d, double r, double i)
 {
	 d->re = r; d->im = i;
 }

 inline void cmultc(cplx* res, cplx a, cplx b)
 {
	 res->re = a.re * b.re - a.im * b.im;
	 res->im = a.im * b.re + a.re * b.im;
 }

 inline cplx cmultco(cplx a, cplx b)
 {
	 cplx res;
	 res.re = a.re * b.re - a.im * b.im;
	 res.im = a.im * b.re + a.re * b.im;
	 return res;
 }



 inline cplx cdivco(cplx a, cplx b)
 {
	 cplx res;
	 double dmod = cmodinv(b);

	 res.re = (a.re * b.re + a.im * b.im) * dmod;
	 res.im = (a.im * b.re - a.re * b.im) * dmod;
	 return res;
 }


 inline cplx cmultdo(cplx a, double d)
 {
	 cplx res;
	 res.re = a.re * d;
	 res.im = a.im * d;
	 return res;
 }
 inline void cmultd(cplx* res, cplx a, double d)
 {
	 res->re = a.re * d;
	 res->im = a.im * d;
 }

 inline void caddc(cplx* res, cplx a, cplx b)
 {
	 res->re = a.re + b.re;
	 res->im = a.im + b.im;
 }

 inline cplx caddco(cplx a, cplx b)
 {
	 cplx res;
	 res.re = a.re + b.re;
	 res.im = a.im + b.im;
	 return res;
 }


 inline void caddd(cplx* res, cplx a, double d)
 {
	 res->re = a.re + d;
 }

 inline void csubc(cplx* res, cplx a, cplx b)
 {
	 res->re = a.re - b.re;
	 res->im = a.im - b.im;
 }

 inline cplx csubco(cplx a, cplx b)
 {
	 cplx res;
	 res.re = a.re - b.re;
	 res.im = a.im - b.im;
	 return res;
 }

 inline void csubd(cplx* res, cplx a, double d)
 {
	 res->re = a.re - d;
 }

 inline double cmodinv(cplx a)
 {
	 return 1.0 / cmodsqr(a);
 }

 inline double cmodsqr(cplx a)
 {
	 return (a.re * a.re + a.im * a.im);
 }

 inline double cmodu(cplx a)
 {
	 return sqrt(cmodsqr(a));
 }
 inline int ciszero(cplx a)
 {
	 return (a.re == 0) && (a.im == 0);
 }
 inline cplx cinv(cplx a)
 {
	 cplx res;
	 double cpmod = cmodinv(a);
	 res.re = a.re * cpmod;
	 res.im = -a.im * cpmod;
	 return res;
 }

 inline cplx conju(cplx a)
 {
	 cplx res;
	 res.re = a.re;
	 res.im = -a.im;
	 return res;
 }


