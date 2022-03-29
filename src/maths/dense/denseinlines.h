
#include <math.h>

 static inline void setcplx(cplx* d, double r, double i);
 static inline void cmultc(cplx* res, cplx a, cplx b);
 static inline cplx cmultco(cplx a, cplx b);
 static inline cplx cmultdo(cplx a, double d);
 static inline void cmultd(cplx* res, cplx a, double d);
 static inline void caddc(cplx* res, cplx a, cplx b);
 static inline cplx caddco(cplx a, cplx b);
 static inline void caddd(cplx* res, cplx a, double d);
 static inline void csubc(cplx* res, cplx a, cplx b);
 static inline cplx csubco(cplx a, cplx b);
 static inline void csubd(cplx* res, cplx a, double d);
 static inline double cmodinv(cplx a);
 static inline double cmodsqr(cplx a);

 static inline int ciszero(cplx a);
 static inline cplx cinv(cplx a);
 static inline cplx conju(cplx a);

 static inline double cmodu(cplx a);
 static inline cplx cdivco(cplx a, cplx b);

 static inline void setcplx(cplx* d, double r, double i)
 {
	 d->re = r; d->im = i;
 }

 static inline void cmultc(cplx* res, cplx a, cplx b)
 {
	 res->re = a.re * b.re - a.im * b.im;
	 res->im = a.im * b.re + a.re * b.im;
 }

 static inline cplx cmultco(cplx a, cplx b)
 {
	 cplx res;
	 res.re = a.re * b.re - a.im * b.im;
	 res.im = a.im * b.re + a.re * b.im;
	 return res;
 }

 static inline cplx cmultdo(cplx a, double d)
 {
	 cplx res;
	 res.re = a.re * d;
	 res.im = a.im * d;
	 return res;
 }
 
 static inline void cmultd(cplx* res, cplx a, double d)
 {
	 res->re = a.re * d;
	 res->im = a.im * d;
 }

 static inline void caddc(cplx* res, cplx a, cplx b)
 {
	 res->re = a.re + b.re;
	 res->im = a.im + b.im;
 }

 static inline cplx caddco(cplx a, cplx b)
 {
	 cplx res;
	 res.re = a.re + b.re;
	 res.im = a.im + b.im;
	 return res;
 }

 static inline void caddd(cplx* res, cplx a, double d)
 {
	 res->re = a.re + d;
 }

 static inline void csubc(cplx* res, cplx a, cplx b)
 {
	 res->re = a.re - b.re;
	 res->im = a.im - b.im;
 }

 static inline cplx csubco(cplx a, cplx b)
 {
	 cplx res;
	 res.re = a.re - b.re;
	 res.im = a.im - b.im;
	 return res;
 }

 static inline void csubd(cplx* res, cplx a, double d)
 {
	 res->re = a.re - d;
 }

 static inline double cmodsqr(cplx a)
 {
	 return (a.re * a.re + a.im * a.im);
 }

 static inline double cmodinv(cplx a)
 {
	 return 1.0 / cmodsqr(a);
 }

 static inline double cmodu(cplx a)
 {
	 return sqrt(cmodsqr(a));
 }

 static inline int ciszero(cplx a)
 {
	 return (a.re == 0) && (a.im == 0);
 }
 
 static inline cplx cinv(cplx a)
 {
	 cplx res;
	 double cpmod = cmodinv(a);
	 res.re = a.re * cpmod;
	 res.im = -a.im * cpmod;
	 return res;
 }

 static inline cplx conju(cplx a)
 {
	 cplx res;
	 res.re = a.re;
	 res.im = -a.im;
	 return res;
 }

 static inline cplx cdivco(cplx a, cplx b)
 {
	 cplx res;
	 double dmod = cmodinv(b);

	 res.re = (a.re * b.re + a.im * b.im) * dmod;
	 res.im = (a.im * b.re - a.re * b.im) * dmod;
	 return res;
 }
