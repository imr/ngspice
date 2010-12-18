


void f_alpha(int n_pts, int n_exp, float X[], float Q_d,
float alpha);

void rvfft(float X[], unsigned long int n);


#define TRNOISE_STATE_MEM_LEN 4
struct trnoise_state
{
    double points[TRNOISE_STATE_MEM_LEN];
    size_t top;

    double NA, TS, NAMP, NALPHA, RTSAM, RTSCAPT, RTSEMT;

    float *oneof;
    size_t oneof_length;

    double RTScapTime, RTSemTime;
    bool RTS;
};


struct trnoise_state *trnoise_state_init(double NA, double TS, double NALPHA, double NAMP, double RTSAM, double RTSCAPT, double RTSEMT);

void trnoise_state_gen(struct trnoise_state *this, CKTcircuit *ckt);
void trnoise_state_free(struct trnoise_state *this);


static inline void
trnoise_state_push(struct trnoise_state *this, double val)
{
    this->points[this->top++ % TRNOISE_STATE_MEM_LEN] = val;
}


static inline double
trnoise_state_get(struct trnoise_state *this, CKTcircuit *ckt, size_t index)
{
    while(index >= this->top)
        trnoise_state_gen(this, ckt);

    if(index + TRNOISE_STATE_MEM_LEN < this->top) {
        fprintf(stderr, "ouch, trying to fetch from the past %d %d\n",
                index, this->top);
        exit(1);
    }

    return this->points[index % TRNOISE_STATE_MEM_LEN];
}
