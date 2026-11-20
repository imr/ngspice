/* Direct real-matrix regression and paired refactor timing. */
#include "ngspice/config.h"
#include "ng_klu_parallel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(2); } } while(0)
#ifdef NG_KLU_ALLOC_TEST
void ng_test_alloc_fail(int);
#endif

static double now(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec;
}

static double transpose_error(int n, const int *ap, const int *ai,
                              const double *ax, const double *b, const double *x)
{
    int j,k;
    double worst=0;
    for(j=0;j<n;++j) {
        double residual=b[j],denominator=fabs(b[j]);
        for(k=ap[j];k<ap[j+1];++k) {
            double value=ax[k]*x[ai[k]];
            residual-=value;denominator+=fabs(value);
        }
        worst=fmax(worst,fabs(residual)/fmax(denominator,1e-300));
    }
    return worst;
}

int main(int argc,char **argv)
{
    int n,m,nz,i,j,k,row,col,threads,repeats,scale, pass, varied, transpose;
    int *ap,*ai; double *ax,*original,*b,*x,*y;
    double times[2]={0,0}, worst=0, backwards=0, trans_backwards=0, trans_diff=0;
    char header[256]; FILE *file;
    klu_common cs,cf; klu_symbolic *s; klu_numeric *ns,*nf;
    NgKluPlan *plan=NULL;
    CHECK(argc>=3);threads=atoi(argv[2]);repeats=argc>3?atoi(argv[3]):10;scale=argc>4?atoi(argv[4]):2;
    varied=argc>5 && atoi(argv[5]);
    transpose=argc<=6 || atoi(argv[6]);
    file=fopen(argv[1],"r");CHECK(file);CHECK(fgets(header,sizeof(header),file));
    CHECK(fscanf(file,"%d %d %d",&n,&m,&nz)==3 && n==m && n>0 && nz>0);
    ap=calloc(n+1,sizeof(int));ai=malloc((size_t)nz*sizeof(int));ax=malloc((size_t)nz*sizeof(double));original=malloc((size_t)nz*sizeof(double));
    b=calloc(n,sizeof(double));x=malloc((size_t)n*sizeof(double));y=malloc((size_t)n*sizeof(double));
    CHECK(ap&&ai&&ax&&original&&b&&x&&y);
    col=0;
    for(k=0;k<nz;++k){int c;CHECK(fscanf(file,"%d %d %lf",&row,&c,&ax[k])==3);--row;--c;CHECK(row>=0&&row<n&&c>=col&&c<n);ai[k]=row;++ap[c+1];col=c;}
    fclose(file);for(j=0;j<n;++j)ap[j+1]+=ap[j];memcpy(original,ax,(size_t)nz*sizeof(double));
    CHECK(klu_defaults(&cs)&&klu_defaults(&cf));cs.scale=cf.scale=scale;cs.tol=cf.tol=0.001;
    s=klu_analyze(n,ap,ai,&cs);CHECK(s);
    ns=klu_factor(ap,ai,ax,s,&cs);nf=klu_factor(ap,ai,ax,s,&cf);CHECK(ns&&nf);
#ifdef NG_KLU_ALLOC_TEST
    for(i=1;i<=11;++i) {
        ng_klu_plan_free(&plan);ng_test_alloc_fail(i);
        CHECK(ng_klu_refactor(&plan,ap,ai,ax,s,nf,&cf,threads));
        ng_test_alloc_fail(0);
        for(j=0;j<n;++j)x[j]=y[j]=.2+sin(j*.19);
        CHECK(klu_solve(s,ns,n,1,x,&cs)&&klu_solve(s,nf,n,1,y,&cf));
        CHECK(memcmp(x,y,(size_t)n*sizeof(double))==0);
    }
    ng_klu_plan_free(&plan);
#endif
    CHECK(ng_klu_refactor(&plan,ap,ai,ax,s,nf,&cf,threads));
    for(pass=0;pass<repeats;++pass){
        double t, factor=1+0.002*sin(pass*0.7);
        for(k=0;k<nz;++k)ax[k]=original[k]*(varied ? 1+0.002*sin(pass*.7+k*.11) : factor);
        /* Alternate order to reduce warm-cache and load-order bias. */
        for(i=0;i<2;++i){int fast=(i+pass)%2;t=now();
            CHECK(fast?ng_klu_refactor(&plan,ap,ai,ax,s,nf,&cf,threads):klu_refactor(ap,ai,ax,s,ns,&cs));
            times[fast]+=now()-t;
        }
        CHECK(cs.status==cf.status);
        memset(b,0,(size_t)n*sizeof(double));
        for(j=0;j<n;++j)for(k=ap[j];k<ap[j+1];++k)b[ai[k]]+=ax[k]*(0.3+sin(j*0.31+pass*.1));
        memcpy(x,b,(size_t)n*sizeof(double));memcpy(y,b,(size_t)n*sizeof(double));
        CHECK(klu_solve(s,ns,n,1,x,&cs)&&klu_solve(s,nf,n,1,y,&cf));
        for(i=0;i<n;++i){CHECK(isfinite(x[i])&&isfinite(y[i]));worst=fmax(worst,fabs(x[i]-y[i]));}
        CHECK(memcmp(x,y,(size_t)n*sizeof(double))==0);
        /* Componentwise backward error, independent of the factor routine. */
        for(i=0;i<n;++i)x[i]=fabs(b[i]);
        for(j=0;j<n;++j)for(k=ap[j];k<ap[j+1];++k){double v=ax[k]*y[j];b[ai[k]]-=v;x[ai[k]]+=fabs(v);}
        for(i=0;i<n;++i)backwards=fmax(backwards,fabs(b[i])/fmax(x[i],1e-300));
        if(backwards>1e-10) fprintf(stderr,"backward_error=%.17g (candidate and KLU solution are bit-identical) scale=%d pass=%d\n",backwards,scale,pass);
        CHECK(backwards<=1e-10);
        if(transpose) {
        /* Sorting L entries can reorder a transpose solve's reduction. Check
         * both transpose residuals, without promising identical roundoff. */
        memset(b,0,(size_t)n*sizeof(double));
        for(j=0;j<n;++j)for(k=ap[j];k<ap[j+1];++k)b[j]+=ax[k]*(.3+sin(ai[k]*.31+pass*.1));
        memcpy(x,b,(size_t)n*sizeof(double));memcpy(y,b,(size_t)n*sizeof(double));
        CHECK(klu_tsolve(s,ns,n,1,x,&cs)&&klu_tsolve(s,nf,n,1,y,&cf));
        for(i=0;i<n;++i) {
            CHECK(isfinite(x[i])&&isfinite(y[i]));
            trans_diff=fmax(trans_diff,fabs(x[i]-y[i])/fmax(1.,fmax(fabs(x[i]),fabs(y[i]))));
        }
        {
            double baseline_error=transpose_error(n,ap,ai,ax,b,x);
            double candidate_error=transpose_error(n,ap,ai,ax,b,y);
            if(baseline_error>1e-10 || candidate_error>1e-10)
                fprintf(stderr,"transpose baseline_error=%.17g candidate_error=%.17g scaled_diff=%.17g scale=%d pass=%d\n",
                        baseline_error,candidate_error,trans_diff,scale,pass);
            trans_backwards=fmax(trans_backwards,fmax(baseline_error,candidate_error));
        }
        CHECK(trans_backwards<=1e-10);
        }
    }
    if(varied) {
        int a,z;
        memset(ax,0,(size_t)nz*sizeof(double));
        a=klu_refactor(ap,ai,ax,s,ns,&cs);
        z=ng_klu_refactor(&plan,ap,ai,ax,s,nf,&cf,threads);
        CHECK(a==z && cs.status==cf.status && cs.numerical_rank==cf.numerical_rank && cs.singular_col==cf.singular_col);
        memcpy(ax,original,(size_t)nz*sizeof(double));
        CHECK(klu_refactor(ap,ai,ax,s,ns,&cs) && ng_klu_refactor(&plan,ap,ai,ax,s,nf,&cf,threads));
        for(i=0;i<n;++i)x[i]=y[i]=0.1+sin(i*.31);
        CHECK(klu_solve(s,ns,n,1,x,&cs)&&klu_solve(s,nf,n,1,y,&cf));
        CHECK(memcmp(x,y,(size_t)n*sizeof(double))==0);
    }
    printf("PASS n=%d nz=%d threads=%d scale=%d repeats=%d serial_s=%.9g optimized_s=%.9g speedup=%.6g max_diff=%.17g backward_error=%.17g transpose_checked=%d transpose_backward_error=%.17g transpose_scaled_diff=%.17g\n",n,nz,threads,scale,repeats,times[0],times[1],times[0]/times[1],worst,backwards,transpose,trans_backwards,trans_diff);
    ng_klu_plan_free(&plan);klu_free_numeric(&ns,&cs);klu_free_numeric(&nf,&cf);klu_free_symbolic(&s,&cs);
    free(ap);free(ai);free(ax);free(original);free(b);free(x);free(y);
    return 0;
}
