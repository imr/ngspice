#include <stdio.h>
#include <assert.h>
#include <string.h>

/////// SNDFILE ///////
#include <stdlib.h>
#include <math.h>
#include <sndfile.h>
#include <inttypes.h>

// Resampling can be rather slow. Don't resample
// the whole audio file, do it in smaller chunks
#define VS_RESAMPLING_CHUNK 1024

#include "ngspice/ngspice.h"
#include "vsjack.h"

extern char* inp_pathresolve(const char* name);

#define MAX_D 6 

static SNDFILE* m_sndfile[MAX_D];
static int m_channel[MAX_D]; //< channel to be used in src-file
static int m_channels[MAX_D]; //< number of channles in src-file
static uint32_t m_samplerate[MAX_D]; //< samplerate of source
static uint32_t m_frames[MAX_D]; //< duration of source in frames
static float* (interleaved[MAX_D]); //< internal soundfile buffer

#define HAVE_SRC

#ifdef HAVE_SRC
#include <samplerate.h>
static double src_ratio[MAX_D];
static SRC_STATE* rabbit[MAX_D];
static int rabbit_err[MAX_D];
static float* (resampled[MAX_D]); //< internal soundfile buffer
static uint32_t input_frames_used[MAX_D];
static uint32_t output_frames_generated[MAX_D];
#endif

static void vsjack_initialize(void) {
    int d;
    for (d = 0; d < MAX_D; d++) {
        m_sndfile[d] = NULL;
        interleaved[d] = NULL;
#ifdef HAVE_SRC
        resampled[d] = NULL;
#endif
    }
}

static void realloc_sf(int d, uint32_t buffersize) {
    if (interleaved[d]) free(interleaved[d]);
    interleaved[d] = (float*)calloc(m_channels[d] * buffersize, sizeof(float));
}

#ifdef HAVE_SRC
static void realloc_src(int d, uint32_t buffersize) {
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

static int openfile_sf(int d, char* filename, uint32_t channel, double oversampling) {
    int nframes;
    SF_INFO sfinfo;
    if (!m_sndfile[d])
        sf_close(m_sndfile[d]);
    printf("Opening file '%s' for id:%i\n", filename, d);

    /* search intensively for the input file */
    char* const path = inp_pathresolve(filename);

    if (!path) {
        fprintf(stderr, "Error: Could not find file %s.\n", filename);
        return (-1);
    }

    m_sndfile[d] = sf_open(path, SFM_READ, &sfinfo);
    txfree(path);

    if (SF_ERR_NO_ERROR != sf_error(m_sndfile[d])) {
        fprintf(stderr, "Error: This is not a sndfile supported audio file format\n");
        return (-1);
    }
    if (sfinfo.frames == 0) {
        fprintf(stderr, "Error: This is an empty audio file\n");
        return (-1);
    }
    nframes = sfinfo.frames;
    if (channel >= sfinfo.channels) {
        fprintf(stderr, "Error: Audio file does not have channel %d (0-%d)\n", channel, sfinfo.channels-1);
        return (-1);
    }

    m_channel[d] = channel;
    m_channels[d] = sfinfo.channels;
    m_samplerate[d] = sfinfo.samplerate;
    m_frames[d] = nframes;
    realloc_sf(d, nframes);
#ifdef HAVE_SRC

    src_ratio[d] = oversampling;
    realloc_src(d, nframes * oversampling);
    rabbit[d] = src_new(SRC_SINC_BEST_QUALITY, m_channels[d], &(rabbit_err[d]));
    src_set_ratio(rabbit[d], oversampling);
    src_reset(rabbit[d]);
    output_frames_generated[d] = 0;
    input_frames_used[d] = 0;

#endif
    nframes = sf_readf_float(m_sndfile[d], (interleaved[d]), nframes);
    if (nframes < 0) {
        fprintf(stderr, "Error: Failed to read audio frames\n");
        return (-1);
    }
    m_frames[d] = nframes;
    return (0);
}

static double get_value(int d, double time) {
    uint32_t channel = m_channel[d];
    uint32_t nframes = m_frames[d];
    double sample_fp = time * ((double)m_samplerate[d]);
    uint32_t sample = (uint32_t)floor(sample_fp);

    if (sample >= nframes) return (0.0);

#ifdef HAVE_SRC
    double SRC_RATIO = src_ratio[d];
    sample_fp *= SRC_RATIO;
    sample = (uint32_t)floor(sample_fp);

    // Do we need to generate more output frames?
    while (sample >= output_frames_generated[d]) {
        SRC_DATA src_data;
        uint32_t output_generated = output_frames_generated[d];
        uint32_t input_used = input_frames_used[d];
        uint32_t input_frames_left = nframes - input_used;

        // Not enough output frames, and nothing more to input?
        // Give up.
        if (!input_frames_left)
            return (0.0);

        // Do the resampling in smaller chunks
        src_data.end_of_input = 1;
        if (input_frames_left > VS_RESAMPLING_CHUNK) {
            input_frames_left = VS_RESAMPLING_CHUNK;
            src_data.end_of_input = 0;
        }

        src_data.data_in = interleaved[d] + m_channels[d] * input_used;
        src_data.data_out = resampled[d] + m_channels[d] * output_generated;
        src_data.input_frames = input_frames_left;
        src_data.output_frames = nframes * SRC_RATIO - output_generated;
        src_data.src_ratio = SRC_RATIO;
        src_data.output_frames_gen = 0;
        src_data.input_frames_used = 0;

        if (src_process(rabbit[d], &src_data)) {
            fprintf(stderr, "src_process() failed on sound file");
            return -1;
        }

        output_frames_generated[d] += src_data.output_frames_gen;
        input_frames_used[d] += src_data.input_frames_used;
        if (src_data.end_of_input)
            break;
    }

    // Are we past all the generated samples?
    if (sample >= output_frames_generated[d])
        return (0.0);
    float val = ((float*)(resampled[d]))[m_channels[d] * sample + channel];
    // Are we the last sample?
    if (sample + 1 == output_frames_generated[d])
        return val;

    // linear interpolation between samples
    double diff = sample_fp - sample;
    float val1 = ((float*)(resampled[d]))[(m_channels[d] * (sample + 1)) + channel];
    double rv = ((double)val) * (1.0 - diff) + ((double)val1) * diff;
    return(rv);

#else // no upsampling.

    return((double)(((float*)(interleaved[d]))[m_channels[d] * sample + channel]));
#endif
    }


/*
 * "public" functions
 */

double vsjack_get_value(int d, double time, double time_offset) {
    assert(d >= 0 && d < MAX_D);
    if (m_sndfile[d] == NULL) return (0.0); // FIXME

    double value = get_value(d, time + time_offset);
    return (value);
}

int vsjack_open(int d, char *file, int channel, double oversampling) {
    static int initialized = 0;
    if (!initialized) {
        initialized = 1;
        vsjack_initialize();
    }
    assert(d >= 0 && d < MAX_D);

    if (openfile_sf(d, file, channel, oversampling)) {
        fprintf(stderr, "Error: Could not open or read '%s'\n", file);
        controlled_exit(1);
    }
    return (d);
}

/* vi:set ts=8 sts=4 sw=4: */
