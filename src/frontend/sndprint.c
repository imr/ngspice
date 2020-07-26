#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sndfile.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <config.h>
#include "sndprint.h"


int o_samplerate = 48000;
int o_sndfmt = (SF_FORMAT_WAV | SF_FORMAT_PCM_24);
float o_mult = 1.0;
float o_off = 0.0;

//////////////////////////////////   aliki  //////////////////////////////////

#define HDRSIZE 256

void * my_open_aliki(char *fn, int nchannel) {
  char p[HDRSIZE];
  FILE *aldfile;
  if ((aldfile = fopen (fn, "w")) == 0) {
    printf ("Error: Not able to open output file '%s'\n", fn);
    exit (1);
  }

  strcpy (p, "aliki");
  p [6] = p [7] = 0;
  *(uint32_t *)(p +  8) = 2; //_vers
  *(uint32_t *)(p + 12) = nchannel; // _type;
  *(uint32_t *)(p + 16) = o_samplerate; //_rate_n;
  *(uint32_t *)(p + 20) = 1; //_rate_d;
  *(uint32_t *)(p + 24) = 486239; //_n_fram;
  *(uint32_t *)(p + 28) = 1; // _n_sect;
  *(uint32_t *)(p + 32) = 0; // _tref_i;
  *(uint32_t *)(p + 36) = 0; // _tref_n;
  *(uint32_t *)(p + 40) = 1; // _tref_d;
  *(uint32_t *)(p + 44) = 0; // _bits;

  memset (p + 48, 0, HDRSIZE - 48);
  if (fwrite (p, 1, HDRSIZE, aldfile) != HDRSIZE) {
    printf ("Error: Not able to write aliki header to '%s'\n", fn);
    fclose (aldfile);
    exit(1);
  }
  return ((void*) aldfile);
}

int my_write_aliki(void *d, float val) {
  return(fwrite(&val, sizeof (float), 1, (FILE*) d));
}

void my_close_aliki(void *d) {
  fclose((FILE*) d);
}


//////////////////////////////////  sndfile //////////////////////////////////

typedef struct {
  SNDFILE *outfile ;
  int sf_channels;
  int sf_bptr;
  float *sf_buf;
} SSFILE;

void * my_open_sf(char *fn, int nchannel) {

  SSFILE *d = calloc(1,sizeof(SSFILE));
  SF_INFO sfinfo ;

  sfinfo.samplerate = o_samplerate;
  sfinfo.channels = nchannel;
  sfinfo.frames = 0;
  sfinfo.format = o_sndfmt;

  d->sf_channels = nchannel;
  d->sf_bptr = 0;
  d->sf_buf=calloc(nchannel, sizeof(float));

  if ((d->outfile = sf_open (fn, SFM_WRITE, &sfinfo)) == NULL) {
    printf ("Error: Not able to open output file '%s'\n", fn);
    exit (1) ;
  }

#if 1
  sf_command (d->outfile, SFC_SET_UPDATE_HEADER_AUTO, NULL, SF_TRUE) ;
  sf_command (d->outfile, SFC_SET_CLIPPING, NULL, SF_TRUE) ;
#endif

  return ((void*)d);
}

int my_write_sf(void *d, float val) {
  SSFILE *p= (SSFILE*) d;
  p->sf_buf[p->sf_bptr++] =val;
  if (p->sf_bptr >= p->sf_channels) {
    sf_writef_float (p->outfile, p->sf_buf, 1);
    p->sf_bptr=0;
  }
  return (1);
}

void my_close_sf(void *d) {
  sf_close(((SSFILE*)d)->outfile);
  free(((SSFILE*)d)->sf_buf);
  free((SSFILE*)d);
}





//////////////////////////////////   spice  //////////////////////////////////

typedef struct SP_BUF {
  double tme;
  double *val;
} SP_BUF;

void (*p_close)(void*);
void *(*p_open)(char*, int);
int  (*p_write)(void*, float);
void * outfile;
uint32_t sample;
int sp_nchannel;
#define SP_MAX (2)
SP_BUF *sp_buf;
char *filename = NULL;

#define HAVE_SRC

#ifndef HAVE_SRC
#define OVERSAMPLING (1.0)
#else
#include <samplerate.h>
#define OBUFSIZE 256
int oversampling = 64;
#define OVERSAMPLING ((double) oversampling)
SRC_STATE *rabbit;
int rabbit_err;
float *interleaved; 
float *resampled; 
int iptr = 0;

int resample_wrapper (void *d, float val) {
  interleaved[iptr++] = val;
  size_t ibufsize = sp_nchannel * OBUFSIZE * OVERSAMPLING;
  size_t obufsize = sp_nchannel * OBUFSIZE ;
  if (iptr ==  ibufsize) {

    SRC_DATA src_data;
    src_data.data_in = interleaved;
    src_data.data_out = resampled;
    src_data.input_frames  = iptr/sp_nchannel;
    src_data.output_frames = OBUFSIZE;
    src_data.end_of_input  = 0;
    src_data.src_ratio     =  1.0/OVERSAMPLING;
    src_data.input_frames_used = 0;
    src_data.output_frames_gen = 0;
  
    //printf ("the rabbit says: %s\n", src_strerror(
    src_process(rabbit, &src_data);
    //));

    if (src_data.output_frames_gen *sp_nchannel != obufsize) {
      printf ("resample warning: out %li != %i\n", src_data.output_frames_gen*sp_nchannel,  obufsize);
    }

    if (src_data.input_frames_used *sp_nchannel != iptr) {
      printf ("resample warning: in: %li != %i\n", src_data.input_frames_used*sp_nchannel, iptr);
    }

    int i;
    for (i=0; i< src_data.output_frames_gen*sp_nchannel; i++) 
      p_write (d, resampled[i]);

    iptr=0;
    return (src_data.output_frames_gen*sp_nchannel);
  }
  return (0);
}

#endif

void snd_configure(char *fn, int srate, int fmt, double mult, double off, int os ){
  if (filename) free(filename);
  filename=strdup(fn);

  o_samplerate = srate;
  o_mult = mult;
  o_off = off;
  oversampling = os;
  if (fmt!=0) {
    p_close = &my_close_sf;
    p_open = &my_open_sf;
    p_write = &my_write_sf;
    o_sndfmt=(fmt>0)?fmt:(SF_FORMAT_WAV | SF_FORMAT_PCM_24);
    printf("info: opened snd file '%s'\n",filename);
  } else {
    p_close = &my_close_aliki;
    p_open = &my_open_aliki;
    p_write = &my_write_aliki;
    printf("info: opened aliki file '%s'\n",filename);
  }
}

int snd_format(char *fmt) {
  int f = atoi(fmt);
  if (!strcmp(fmt, "wav")) f= (SF_FORMAT_WAV | SF_FORMAT_PCM_24);
  if (!strcmp(fmt, "wav16")) f= (SF_FORMAT_WAV | SF_FORMAT_PCM_16);
  if (!strcmp(fmt, "wav24")) f= (SF_FORMAT_WAV | SF_FORMAT_PCM_24);
  if (!strcmp(fmt, "wav32")) f= (SF_FORMAT_WAV | SF_FORMAT_PCM_32);
  if (!strcmp(fmt, "aiff")) f= (SF_FORMAT_AIFF | SF_FORMAT_PCM_16);
  if (!strcmp(fmt, "aliki")) f= 0;
  return (f);
}

void snd_init(int nchannel) {
  int i;
  if (!filename) snd_configure("spice.wav", 48000.0, o_sndfmt, o_mult, o_off, oversampling);
  outfile = p_open (filename, nchannel);
  sp_nchannel = nchannel;
  sp_buf = calloc(nchannel, sizeof(SP_BUF));
  for (i=0; i< SP_MAX; i++){
    sp_buf[i].tme=0.0;
    sp_buf[i].val = calloc(nchannel, sizeof(float));
  }
  sample=0;
#ifdef HAVE_SRC
  interleaved=calloc(nchannel*OBUFSIZE*OVERSAMPLING, sizeof(float));
  resampled=calloc(nchannel*OBUFSIZE, sizeof(float));
  rabbit=src_new(SRC_SINC_BEST_QUALITY, nchannel, &rabbit_err);
  src_set_ratio(rabbit, 1.0/OVERSAMPLING);
  src_reset(rabbit);
#endif
}

int snd_send(double tme, int c, double out) {
  int i;
  int rv =0;
  if (c==0) for (i=SP_MAX-1; i>0; i--) {
    memcpy(&(sp_buf[i]), &(sp_buf[i-1]), sizeof(SP_BUF));
  }
  sp_buf[0].tme=tme * OVERSAMPLING;
  sp_buf[0].val[c]=out;
#ifdef SND_DEBUG
  printf ("INFO : c:%i tme:%f fsmp:%i val:%f\n", c, tme, sample,out); 
#endif

  if (sample == 0) {
    if (c==(sp_nchannel-1)) 
      sample = ceil(tme * OVERSAMPLING);
    return (0);
  }

  if ( (sample) < ceil(tme * OVERSAMPLING) ) {
    if (!(sp_buf[0].tme > sample)) printf ("error 1 %f !> %i\n",sp_buf[0].tme,sample);
    if ( (sp_buf[1].tme > sample)) printf ("error 2 %f !< %i\n",sp_buf[1].tme,sample);
#if 1 // DEBUG
    if ((sp_buf[0].tme - sample) > 1.0) printf ("error 3 large timestep: dv/dt=%e dt:%f dv:%e\n",
      (sp_buf[0].val[c] - sp_buf[1].val[c]) / (sp_buf[0].tme - sample) ,
      (sp_buf[0].tme - sample), ( sp_buf[0].val[c] - sp_buf[1].val[c]) );
#endif

    // linear
    double p = (sp_buf[0].tme - sample ) / (sp_buf[0].tme - sp_buf[1].tme);
    double val = sp_buf[0].val[c] - p * ( sp_buf[0].val[c] - sp_buf[1].val[c] );
#ifdef SND_DEBUG
    printf ("DEBUG: writing c:%i p:%f*[%f - %f] v:%f\n",c,p,sp_buf[0].val[c], sp_buf[1].val[c], val);
#endif

#ifdef HAVE_SRC
    rv = resample_wrapper (outfile, o_off + val * o_mult);
#else
    p_write(outfile, o_off + val * o_mult);
    if (c==(sp_nchannel-1)) rv =1;
#endif
    if (c==(sp_nchannel-1)) sample ++;

  } else {
#ifdef SND_DEBUG
    printf(" ^^^^^^^^^ SKIPPED ^^^^^^^^^\n");
#endif
  }
  return (rv);
}

void snd_close(void) {
#ifdef HAVE_SRC
  while (!resample_wrapper(outfile, 0.0)); // flush buffer.
#endif
  p_close(outfile);
  free(filename); filename=NULL;
#ifdef HAVE_SRC
  free (interleaved);
  free (resampled);
#endif
  /*
  int i;
  for (i=0; i< SP_MAX; i){ 
    free (sp_buf[i].val);
    sp_buf[i].val=NULL;
  }
  */
  free (sp_buf);
}

double snd_get_samplerate(void) { 
  return ((double) o_samplerate);
}

/* vi:set ts=8 sts=2 sw=2: */

