#ifndef ngspice_DENSE_MATRIX_H
#define ngspice_DENSE_MATRIX_H

#include <math.h>

typedef struct cplx
{
	double re;
	double im;
}cplx;

typedef struct Mat{
	double** d;
	int row;
	int col;
}Mat;

typedef struct CMat {
	cplx** d;
	int row;
	int col;
}CMat;


typedef struct MatList{
	Mat* mat;
	struct MatList* next;
}MatList;

 void showmat(Mat* A);
 void showcmat(CMat* A);
 CMat* newcmat(int r, int c, double dr, double di);
 CMat* newcmatnoinit(int r, int c);
 Mat* newmat(int r, int c, double d);
 Mat* newmatnoinit(int r, int c);
 void freecmat(CMat* A);
 void freemat(Mat* A);
 CMat* ceye(int n);
 Mat* eye(int n);
 Mat* zeros(int r, int c);
 CMat* czeros(int r, int c);
 Mat* ones(int r, int c);
 CMat* cones(int r, int c);
 Mat* randm(int r, int c, double l, double u);
 CMat* randcm(int r, int c, double l, double u);
 double get(Mat* M, int r, int c);	
 cplx getcplx(CMat* M, int r, int c);
 void set(Mat* M, int r, int c, double d);
 void setc(CMat* M, int r, int c, cplx d);
 Mat* scalarmultiply(Mat* M, double c);
 CMat* cscalarmultiply(CMat* M, double c);
 CMat* complexmultiply(CMat* M, cplx c);
 Mat* sum(Mat* A, Mat* B);
 CMat* csum(CMat* A, CMat* B);
 Mat* minus(Mat* A, Mat* B);
 CMat* cminus(CMat* A, CMat* B);
 Mat* submat(Mat* A, int r1, int r2, int c1, int c2);
 void submat2(Mat* A, Mat* B, int r1, int r2, int c1, int c2);
 CMat* subcmat(CMat* A, int r1, int r2, int c1, int c2);
 void subcmat2(CMat* A, CMat* B, int r1, int r2, int c1, int c2);
 Mat* multiply(Mat* A, Mat* B);
 CMat* cmultiply(CMat* A, CMat* B);
 Mat* removerow(Mat* A, int r);
 Mat* removecol(Mat* A, int c);
 void removerow2(Mat* A, Mat* B, int r);
 void removecol2(Mat* A, Mat* B, int c);
 CMat* cremoverow(CMat* A, int r);
 CMat* cremovecol(CMat* A, int c);
 void cremoverow2(CMat* A, CMat* B, int r);
 void cremovecol2(CMat* A, CMat* B, int c);
 Mat* transpose(Mat* A);
 CMat* ctranspose(CMat* A);
 double trace(Mat* A);
 cplx ctrace(CMat* A);
 Mat* adjoint(Mat* A);
 CMat* cadjoint(CMat* A);
 CMat* ctransposeconj(CMat* source);
 Mat* inverse(Mat* A);
 CMat* cinverse(CMat* A);
 Mat* copyvalue(Mat* A);
 CMat* copycvalue(CMat* A);
 Mat* triinverse(Mat* A);
 CMat* ctriinverse(CMat* A);
 Mat* rowechelon(Mat* A);
 CMat* crowechelon(CMat* A);
 Mat* hconcat(Mat* A, Mat* B);
 CMat* chconcat(CMat* A, CMat* B);
 Mat* vconcat(Mat* A, Mat* B);
 CMat* cvconcat(CMat* A, CMat* B);
 double norm(Mat* A);
 double cnorm(CMat* A);
 Mat* nullmat(Mat* A);
 MatList* lu(Mat* A);
 double innermultiply(Mat* a, Mat* b);
 MatList* qr(Mat* A);
 int complexmultiplydest(CMat* M, cplx c, CMat* dest);
 void cinversedest(CMat* A, CMat* dest);
 int copycvaluedest(CMat* A, CMat* dest);
 int cmultiplydest(CMat* A, CMat* B, CMat* dest);
 void init(Mat* A, double d);
 void cinit(CMat* A, double dr, double di);
 void resizemat(Mat* A, int r, int c);
 /*
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

 extern inline double cmodu(cplx a);
 extern inline cplx cdivco(cplx a, cplx b);

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
 */


#endif
