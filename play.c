#include <alsa/asoundlib.h>
#include <asm-generic/errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define SAMPLE_RATE 44100
#define CHUNK_SIZE 1024
#define FFT_SIZE 1024
#define FFT_BINS (FFT_SIZE / 2 + 1)

typedef struct {
    double re;
    double im;
} Complex;

typedef struct {
    float *magnitudes;
} FFTWindow;

char chid[4];
uint16_t audioformat = 0, channels = 0, bitspersample = 0, blockalign = 0;
uint32_t samplerate = 0, byterate = 0, chsize = 0;
uint8_t *data = NULL;

static double wpeak = 0.1;

FFTWindow *fft_windows;
size_t fft_wcnt;

void fft(Complex *x, int n) {
    for (int i = 0; i < n; i++) {
        int j = 0;
        int og = i;
        for (int bit = n; bit > 1; bit >>= 1) {
            j = (j << 1) | (og & 1);
            og >>= 1;
        }
        if (i > j) {
            Complex tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
    for (int bs = 2; bs <= n; bs <<= 1) {
        double angle = -2.0*M_PI/bs;
        Complex wlen = { .re = cos(angle), .im = sin(angle) };
        for (int start = 0; start < n; start += bs) {
            Complex w = { .re = 1.0, .im = 0.0 };
            for (int j = 0; j < bs / 2; j++) {
                Complex u = x[start + j];
                Complex v = x[start + j + bs / 2];
                Complex t = {w.re * v.re - w.im * v.im, w.re * v.im + w.im * v.re};

                x[start + j] = (Complex){u.re + t.re, u.im + t.im};
                x[start + j + bs / 2] = (Complex){u.re - t.re, u.im - t.im};

                double wr = w.re * wlen.re - w.im * wlen.im;
                double wi = w.re * wlen.im + w.im * wlen.re; 

                w.re = wr;
                w.im = wi;
            }
        }
    }
}

void calculate() {
    size_t tframes = chsize / blockalign;
    fft_wcnt = tframes / FFT_SIZE;
    fft_windows = malloc(fft_wcnt*sizeof(FFTWindow));

    int16_t *audio = (int16_t *)data;
    Complex samples[FFT_SIZE];

    for (int w = 0; w < fft_wcnt; w++) {
        for (int i = 0; i < FFT_SIZE; i++) {
            double sample = audio[(w*FFT_SIZE + i)*channels]/32768.0;
            double hann = 0.5 * (1.0 - cos(2.0 * M_PI * i / (FFT_SIZE - 1)));
            samples[i] = (Complex){sample * hann, 0.0};
        }
        fft(samples, FFT_SIZE);
        fft_windows[w].magnitudes = malloc(FFT_BINS*sizeof(float));

        for (int k = 0; k < FFT_BINS; k++) {
            double re = samples[k].re;
            double im = samples[k].im;

            fft_windows[w].magnitudes[k] = (float)(sqrt(re*re + im*im) / FFT_SIZE);
        }
    }
}

void show_wave(long cur) {
    enum { W = 80, H = 20, WIN = 1024 };
    enum { DW = W * 2, DH = H * 4 };
    char dots[DH][DW + 1] = {0};

    long tframs = chsize / blockalign;

    int base = cur - WIN;
    if (base < 0) base = 0;

    int16_t *audio = (int16_t *)data;

    long zs = base;
    for (long i = base + 1; i - base < WIN / 2; i++) {
        if (i >= tframs) break;
        int16_t p = audio[(size_t)(i - 1)*channels];
        int16_t c = audio[(size_t)i*channels];

        if (p < 0 && c >= 0) {
            zs = i;
            break;
        }
    }

    double peak = 0;
    for (int i = zs; i - zs < WIN; i++) {
        if (i >= tframs) break;
        double amp = fabs(audio[(size_t)i*channels]/32768.0);
        peak = fmax(amp, peak);
    }

    wpeak = peak > wpeak ? peak : 0.98 * wpeak + 0.02 * peak;
    double scale = 1 / fmax(wpeak, 0.05);

    for (int x = 0; x < DW; x++) {
        long s0 = zs + (long)x * WIN / DW;
        long s1 = zs + (long)(x + 1) * WIN / DW;

        double mn = 0.0, mx = 0.0;
        int has = 0;

        for (long i = s0; i < s1 && i < tframs; i++) {
            double v = audio[(size_t)i*channels]/32768.0;
            if (!has) {
                mn = mx = v;
                has = 1;
            } else {
                mn = fmin(mn, v);
                mx = fmax(mx, v);
            }
        }

        if (!has) continue;

        int y0 = (int)lround((1.0 - mx*scale)*(DH - 1)/2.0);
        int y1 = (int)lround((1.0 - mn*scale)*(DH - 1)/2.0);

        y0 = fmax(y0, 0);
        y1 = fmin(y1, DH - 1);

        for (int y = y0; y <= y1; y++) dots[y][x] = 1;
    }

    static const uint8_t bit[4][2] = {
        {0x01, 0x08},
        {0x02, 0x10},
        {0x04, 0x20},
        {0x40, 0x80}
    };

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            unsigned b = 0;
            for (int dy = 0; dy < 4; dy++)
                for (int dx = 0; dx < 2; dx++)
                    if (dots[r*4+dy][c*2+dx]) b|= bit[dy][dx];

            if (b == 0) {
                putchar(' ');
                continue;
            }
            char u[3] = {
                (char)0xE2,
                (char)(0xA0 | (b >> 6)),
                (char)(0x80 | (b & 0x3F))
            };

            fwrite(u, 1, 3, stdout);
        }
        putchar('\n');
    }
    fflush(stdout);
}

void show_fft(long cur) {
    enum { W = 80, H = 20 };
    enum { DW = W * 2, DH = H * 4 };

    static double levels[DW] = {0};

    const double F_MIN = 30.0;
    const double F_MAX = 16000.0;
    const double DB_FLOOR = -70.0;

    double bin_hz = (double)samplerate/FFT_SIZE;
    double max_freq = fmin(F_MAX, samplerate / 2.0 - bin_hz);

    if (fft_wcnt == 0 || max_freq <= F_MIN) return;

    int w = cur / FFT_SIZE;
    if (w >= fft_wcnt) w = fft_wcnt - 1;

    float *mag = fft_windows[w].magnitudes;

    uint8_t dots[DH][DW] = {0};

    double ratio = pow(max_freq / F_MIN, 1.0 / DW);

    for (int x = 0; x < DW; x++) {
        int f0 = F_MIN * pow(ratio, x);
        int f1 = F_MIN * pow(ratio, x + 1); 

        int k0 = (int)(f0 / bin_hz);
        int k1 = (int)(f1 / bin_hz);

        if (k0 < 1) k0 = 1;
        if (k0 >= FFT_BINS) k0 = FFT_BINS - 1;
        if (k1 <= k0) k1 = k0 + 1;
        if (k1 >= FFT_BINS) k1 = FFT_BINS;

        float mx = 0.0f;
        for (int k = k0; k < k1; k++) {
            if (mag[k] > mx) mx = mag[k];
        }

        double db = 20.0 * log10(mx/0.25 + 1e-9);
        double cfreq = sqrt(f0*f1);
        db += 3.0 * log2(cfreq/1000.0);

        double level = (db - DB_FLOOR) / -DB_FLOOR;
        if (level < 0.0) level = 0.0;
        if (level > 1.0) level = 1.0;

        levels[x] += (level - levels[x])*(level > levels[x] ? 0.6 : 0.12);

        int height = (int)lround(levels[x]*(DH - 1));
        int y = DH - 1 - height;

        for (int row = y; row < DH; row++) dots[row][x] = 1;
    }

    static const uint8_t bit[4][2] = {
        {0x01, 0x08},
        {0x02, 0x10},
        {0x04, 0x20},
        {0x40, 0x80}
    };

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            unsigned b = 0;

            for (int dy = 0; dy < 4; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    if (dots[r * 4 + dy][c * 2 + dx])
                        b |= bit[dy][dx];
                }
            }

            if (b == 0) {
                putchar(' ');
                continue;
            }

            char u[3] = {
                (char)0xE2,
                (char)(0xA0 | (b >> 6)),
                (char)(0x80 | (b & 0x3F))
            };

            fwrite(u, 1, 3, stdout);
        }

        putchar('\n');
    }

    fflush(stdout);
}

void play() {
    snd_pcm_t *pcm_channel;
    snd_pcm_open(&pcm_channel, "default", SND_PCM_STREAM_PLAYBACK, 0);
    snd_pcm_set_params(pcm_channel, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED, channels, samplerate, 1, 100000);

    snd_pcm_uframes_t fleft = chsize / blockalign;
    uint8_t *ptr = data;
    long long pos = 0;
    printf("\033[2J\033[H");
    while (fleft > 0) {
        snd_pcm_uframes_t frames_write = (fleft > CHUNK_SIZE) ? CHUNK_SIZE : fleft;
        snd_pcm_sframes_t frames = snd_pcm_writei(pcm_channel, ptr, frames_write);
        if (frames < 0) {
            snd_pcm_recover(pcm_channel, frames, 0);
            continue;
        } 
        if (frames == 0) {
            continue;
        }
        fleft -= (snd_pcm_uframes_t)frames;
        ptr += (size_t)frames * blockalign;
        pos += frames;

        snd_pcm_sframes_t delay = 0;
        if (snd_pcm_delay(pcm_channel, &delay) < 0 || delay < 0) delay = 0;
        long cur = pos - delay;
        if (cur < 0) cur = 0;

        long tframes = chsize / blockalign;
        if (cur > tframes) cur = tframes;
        fputs("\033[H", stdout);
        show_wave(cur);
        show_fft(cur);
    }
    snd_pcm_drain(pcm_channel);
    snd_pcm_close(pcm_channel);
}

int loadwav(char *path) {
    FILE *file = fopen(path, "rb");
    fseek(file, 12, SEEK_SET);

    size_t fsize;

    while (1) {
        if(fread(chid, 1, 4, file) != 4) break;
        if(fread(&chsize, 4, 1, file) != 1) break;

        if (memcmp(chid, "fmt ", 4) == 0) {
            fread(&audioformat, 2, 1, file);
            fread(&channels, 2, 1, file);
            fread(&samplerate, 4, 1, file);
            fread(&byterate, 4, 1, file);
            fread(&blockalign, 2, 1, file);
            fread(&bitspersample, 2, 1, file);
            if (chsize > 16) fseek(file, chsize - 16 + (chsize & 1), SEEK_CUR);
        }
        else if (memcmp(chid, "data", 4) == 0) {
            // the pointer is located just after the data size as of now
            break;
        } else {
            fseek(file, chsize + (chsize & 1), SEEK_CUR);
        };
    }

    if (chsize == 0) {
        fclose(file);
        return 0;
    }

    data = malloc(chsize);
    fread(data, 1, chsize, file);
    fclose(file);
    return 1;
}

int main(int argc, char **argv) {
    loadwav(argv[1]);
    calculate();
    setvbuf(stdout, NULL, _IOFBF, 1 << 16);
    play();

    for (int i = 0; i < fft_wcnt; i++) free(fft_windows[i].magnitudes);
    free(fft_windows);
    free(data);

    return 1;
}
