#include <stdio.h>
#include <assert.h>
#include <string.h>

/////// SNDFILE ///////
#include <stdlib.h>
#include <math.h>
#include <sndfile.h>
#include <inttypes.h>
#define VS_BUFSIZ 1024

#include "ngspice/ngspice.h"

#define MAX_D 6 
static char* (sources[MAX_D]);

static SNDFILE* m_sndfile[MAX_D];
static int m_channels[MAX_D]; //< number of channles in src-file
static uint32_t m_samplerate[MAX_D]; //< samplerate of source
static uint32_t m_frames[MAX_D]; //< duration of source in frames
static float* (interleaved[MAX_D]); //< internal soundfile buffer
static uint32_t ilb_start[MAX_D]; //< first sample in buffer 
static uint32_t ilb_end[MAX_D]; //< last sample in buffer

#define HAVE_SRC

#ifdef HAVE_SRC
#include <samplerate.h>
static double src_ratio = 64.0;
#define SRC_RATIO 64
static SRC_STATE* rabbit[MAX_D];
static int rabbit_err[MAX_D];
static float* (resampled[MAX_D]); //< internal soundfile buffer
#endif

void vsjack_initialize(void) {
	int d;
	for (d = 0; d < MAX_D; d++) {
		m_sndfile[d] = NULL;
		interleaved[d] = NULL;
#ifdef HAVE_SRC
		resampled[d] = NULL;
#endif
		sources[d] = NULL;
	}
	sources[0] = strdup("/tmp/test.wav");
	sources[1] = strdup("/tmp/test1.wav");
}

void realloc_sf(int d, uint32_t buffersize) {
	if (interleaved[d]) free(interleaved[d]);
	interleaved[d] = (float*)calloc(m_channels[d] * buffersize, sizeof(float));
}

#ifdef HAVE_SRC
void realloc_src(int d, uint32_t buffersize) {
	if (resampled[d]) free(resampled[d]);
	resampled[d] = (float*)calloc(m_channels[d] * buffersize, sizeof(float));
}
#endif

#if 0
void closefile_sf(int d) {
	if (!m_sndfile[d]) return;
	sf_close(m_sndfile[d]);
#ifdef HAVE_SRC
	src_delete(rabbit[d]);
#endif
	m_sndfile[d] = NULL;
}
#endif

int openfile_sf(int d, char* filename) {
	SF_INFO sfinfo;
	if (!m_sndfile[d]) sf_close(m_sndfile[d]);
	printf("opening file '%s' for id:%i\n", filename, d);
	m_sndfile[d] = sf_open(filename, SFM_READ, &sfinfo);
	ilb_end[d] = ilb_start[d] = 0;

	if (SF_ERR_NO_ERROR != sf_error(m_sndfile[d])) {
		fprintf(stderr, "This is not a sndfile supported audio file format\n");
		return (-1);
	}
	if (sfinfo.frames == 0) {
		fprintf(stderr, "This is an empty audio file\n");
		return (-1);
	}
	m_channels[d] = sfinfo.channels;
	m_samplerate[d] = sfinfo.samplerate;
	m_frames[d] = (uint32_t)sfinfo.frames;
	realloc_sf(d, VS_BUFSIZ);
#ifdef HAVE_SRC
	realloc_src(d, VS_BUFSIZ * SRC_RATIO);
	rabbit[d] = src_new(SRC_SINC_BEST_QUALITY, m_channels[d], &(rabbit_err[d]));
	src_set_ratio(rabbit[d], SRC_RATIO);
	src_reset(rabbit[d]);
#endif
	return (0);
}

void load_buffer(int d, uint32_t sample) {
	sf_seek(m_sndfile[d], sample, SEEK_SET);
	ilb_start[d] = sample;
	uint32_t nframes;
	if ((nframes = (uint32_t)sf_readf_float(m_sndfile[d], (interleaved[d]), VS_BUFSIZ)) > 0) {
		ilb_end[d] = ilb_start[d] + nframes;
	}
	else {
		ilb_end[d] = ilb_start[d];
		printf("Decoder error.\n");
	}
#ifdef HAVE_SRC
	SRC_DATA src_data;
	src_data.data_in = interleaved[d];
	src_data.data_out = resampled[d];
	src_data.input_frames = VS_BUFSIZ;
	src_data.output_frames = VS_BUFSIZ * SRC_RATIO;
	src_data.end_of_input = ((ilb_end[d] - ilb_start[d]) < VS_BUFSIZ);
	src_data.src_ratio = SRC_RATIO;
	src_data.input_frames_used = 0;
	src_data.output_frames_gen = 0;

	src_process(rabbit[d], &src_data);
#endif
}

double get_value(int d, double time, int channel) {
	uint32_t sample = (uint32_t)floor(time * ((double)m_samplerate[d]));

	// TODO: print EOF warning (once). FIXME move to load_buffer
	if (sample > m_frames[d]) return (0.0);

	if (sample < ilb_start[d] || sample >= ilb_end[d])
		load_buffer(d, sample);

	if (sample < ilb_start[d] || sample >= ilb_end[d]) {
		printf("no such value buffered for file:%i.\n", d);
		return (0.0); // nan ?
	}

#ifdef HAVE_SRC
	int offset = (int)floor((sample - ilb_start[d]) * SRC_RATIO);
	if (offset > VS_BUFSIZ * SRC_RATIO || offset < 0) {
		printf("value not in buffer:%i.\n", d);
		return (0.0); // nan ?
	}
	float val = ((float*)(resampled[d]))[m_channels[d] * offset + channel];
#   if 0 // DEBUG 
#   define SQUARE(A) ((A)*(A))
	static double stride = 0;
	static double last = 0;
	static double deviation = 0;
	static int dev_cnt = 0;
	if (channel == 0) {
		stride += (SRC_RATIO * time * ((double)m_samplerate[d])) - last;
		last = (SRC_RATIO * time * ((double)m_samplerate[d]));
		deviation += SQUARE((SRC_RATIO * time * ((double)m_samplerate[d])) - floor(SRC_RATIO * time * ((double)m_samplerate[d])));
		dev_cnt++;
		if ((dev_cnt % (12000)) == 0)
			printf("read time dev= %f - stride= %f\n", sqrt(deviation / (double)dev_cnt), stride / (double)dev_cnt);
	}
#   endif 
# if 0 // zero order hold.
	return((double)val);
# else
	// linear interpolation
	float val1 = ((float*)(resampled[d]))[(m_channels[d] * (offset + 1)) + channel];
	double diff = (SRC_RATIO * time * ((double)m_samplerate[d])) -
		floor(SRC_RATIO * time * ((double)m_samplerate[d]));
	double rv = ((double)val) * (1.0 - diff) + ((double)val1) * diff;
	return(rv);
# endif

#else // no upsampling.

	int offset = sample - ilb_start[d];
	if (offset > VS_BUFSIZ || offset < 0) {
		printf("value not in buffer:%i.\n", d);
		return (0.0); // nan ?
	}
	return((double)(((float*)(interleaved[d]))[m_channels[d] * offset + channel]));
#endif
}


/*
 * "public" functions
 */

double vsjack_get_value(int d, double time, double time_offset, int channel, double oversampling) {
	assert(d >= 0 && d < MAX_D);
	if (m_sndfile[d] == NULL) return (0.0); // FIXME
	if (oversampling > 0) src_ratio = oversampling;

	double value = get_value(d, time + time_offset, channel);
	return (value);
}

void vsjack_set_file(int d, char* fn) {
	assert(d >= 0 && d < MAX_D);
	if (sources[d] != NULL) free(sources[d]);
	sources[d] = strdup(fn);
}

int vsjack_open(int d) {
	static int initialized = 0;
	if (!initialized) {
		initialized = 1;
		vsjack_initialize();
	}
	if (d == -1) return -1;// initialize only
	assert(d >= 0 && d < MAX_D);
	assert(sources[d] != NULL);
	if (openfile_sf(d, sources[d])) {
		fprintf(stderr, "could not open '%s'\n", sources[d]);
		controlled_exit(1);
	}
	return (d);
}

/* vi:set ts=8 sts=4 sw=4: */
