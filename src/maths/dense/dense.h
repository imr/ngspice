#ifndef ngspice_DENSE_MATRIX_H
#define ngspice_DENSE_MATRIX_H


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

extern void showmat(Mat* A);
extern void showcmat(CMat* A);
extern CMat* newcmat(int r, int c, double dr, double di);
extern CMat* newcmatnoinit(int r, int c);
extern Mat* newmat(int r, int c, double d);
extern Mat* newmatnoinit(int r, int c);
extern void freecmat(CMat* A);
extern void freemat(Mat* A);
extern CMat* ceye(int n);
extern Mat* eye(int n);
extern Mat* zeros(int r, int c);
extern CMat* czeros(int r, int c);
extern Mat* ones(int r, int c);
extern CMat* cones(int r, int c);
extern Mat* randm(int r, int c, double l, double u);
extern CMat* randcm(int r, int c, double l, double u);
extern double get(Mat* M, int r, int c);	
extern cplx getcplx(CMat* M, int r, int c);
extern void set(Mat* M, int r, int c, double d);
extern void setc(CMat* M, int r, int c, cplx d);
extern Mat* scalarmultiply(Mat* M, double c);
extern CMat* cscalarmultiply(CMat* M, double c);
extern CMat* complexmultiply(CMat* M, cplx c);
extern Mat* sum(Mat* A, Mat* B);
extern CMat* csum(CMat* A, CMat* B);
extern Mat* minus(Mat* A, Mat* B);
extern CMat* cminus(CMat* A, CMat* B);
extern Mat* submat(Mat* A, int r1, int r2, int c1, int c2);
extern void submat2(Mat* A, Mat* B, int r1, int r2, int c1, int c2);
extern CMat* subcmat(CMat* A, int r1, int r2, int c1, int c2);
extern void subcmat2(CMat* A, CMat* B, int r1, int r2, int c1, int c2);
extern Mat* multiply(Mat* A, Mat* B);
extern CMat* cmultiply(CMat* A, CMat* B);
extern Mat* removerow(Mat* A, int r);
extern Mat* removecol(Mat* A, int c);
extern void removerow2(Mat* A, Mat* B, int r);
extern void removecol2(Mat* A, Mat* B, int c);
extern CMat* cremoverow(CMat* A, int r);
extern CMat* cremovecol(CMat* A, int c);
extern void cremoverow2(CMat* A, CMat* B, int r);
extern void cremovecol2(CMat* A, CMat* B, int c);
extern Mat* transpose(Mat* A);
extern CMat* ctranspose(CMat* A);
extern double trace(Mat* A);
extern cplx ctrace(CMat* A);
extern Mat* adjoint(Mat* A);
extern CMat* cadjoint(CMat* A);
extern CMat* ctransposeconj(CMat* source);
extern Mat* inverse(Mat* A);
extern CMat* cinverse(CMat* A);
extern Mat* copyvalue(Mat* A);
extern CMat* copycvalue(CMat* A);
extern Mat* triinverse(Mat* A);
extern CMat* ctriinverse(CMat* A);
extern Mat* rowechelon(Mat* A);
extern CMat* crowechelon(CMat* A);
extern Mat* hconcat(Mat* A, Mat* B);
extern CMat* chconcat(CMat* A, CMat* B);
extern Mat* vconcat(Mat* A, Mat* B);
extern CMat* cvconcat(CMat* A, CMat* B);
extern double norm(Mat* A);
extern double cnorm(CMat* A);
extern Mat* nullmat(Mat* A);
extern MatList* lu(Mat* A);
extern double innermultiply(Mat* a, Mat* b);
extern MatList* qr(Mat* A);
extern int complexmultiplydest(CMat* M, cplx c, CMat* dest);
extern void cinversedest(CMat* A, CMat* dest);
extern int copycvaluedest(CMat* A, CMat* dest);
extern int cmultiplydest(CMat* A, CMat* B, CMat* dest);
extern void init(Mat* A, double d);
extern void cinit(CMat* A, double dr, double di);
extern void resizemat(Mat* A, int r, int c);

#endif
