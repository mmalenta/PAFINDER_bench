#include <chrono>
#include <iostream>

#include "thrust/device_vector.h"
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include "cuComplex.h"
#include "cufft.h"
#include "cuda.h"

#define NACC 256
#define NFPGAS 48
#define NCHAN_COARSE 336
#define NCHAN_FINE_IN 32
#define NCHAN_FINE_OUT 27
#define NACCUMULATE 128 
#define NPOL 2
#define NSAMPS 4
#define NCHAN_SUM 16
#define NSAMP_PER_PACKET 128
#define NCHAN_PER_PACKET 7
#define HEADLEN 64
#define NSAMPS_SUMMED 2


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using std::cerr;
using std::cout;
using std::endl;

#define XSIZE 7
#define YSIZE 128
#define ZSIZE 48

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*void UnpackCpu(void) {
    #pragma unroll
    for (int chan = 0; chan < 7; chan++) {
        for (int sample = 0; sample < 128; sample++) {
            idx = (sample * 7 + chan) * BYTES_PER_WORD;    // get the  start of the word in the received data array
            idx2 = chan * 128 + sample + startidx;        // get the position in the buffer
            h_pol[idx2].x = static_cast<float>(static_cast<short>(data[HEADLEN + idx + 7] | (data[HEADLEN + idx + 6] << 8)));
            h_pol[idx2].y = static_cast<float>(static_cast<short>(data[HEADLEN + idx + 5] | (data[HEADLEN + idx + 4] << 8)));
            h_pol[idx2 + d_in_size / 2].x = static_cast<float>(static_cast<short>(data[HEADLEN + idx + 3] | (data[HEADLEN + idx + 2] << 8)));
            h_pol[idx2 + d_in_size / 2].y = static_cast<float>(static_cast<short>(data[HEADLEN + idx + 1] | (data[HEADLEN + idx + 0] << 8)));
        }
    }
} */

__global__ void unpack_original_tex(cudaTextureObject_t texObj, cufftComplex * __restrict__ out, unsigned int acc)
{

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * 128;
    int chanidx = threadIdx.x + blockIdx.y * 7;
    int skip;
    int2 word;

    for (int ac = 0; ac < acc; ac++) {
        skip = 336 * 128 * 2 * ac;
        for (int sample = 0; sample < YSIZE; sample++) {
            word = tex2D<int2>(texObj, xidx, yidx + ac * 48 * 128 + sample);
            out[skip + chanidx * YSIZE * 2 + sample].x = static_cast<float>(static_cast<short>(((word.y & 0xff000000) >> 24) | ((word.y & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + sample].y = static_cast<float>(static_cast<short>(((word.y & 0xff00) >> 8) | ((word.y & 0xff) << 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].x = static_cast<float>(static_cast<short>(((word.x & 0xff000000) >> 24) | ((word.x & 0xff0000) >> 8)));
            out[skip + chanidx * YSIZE * 2 + YSIZE + sample].y = static_cast<float>(static_cast<short>(((word.x & 0xff00) >> 8) | ((word.x & 0xff) << 8)));
        }
    }
}

__global__ void unpack_new(const unsigned int *__restrict__ in, cufftComplex * __restrict__ out)
{

    int skip = 0;

    __shared__ unsigned int accblock[1792];
    
    int chan = 0;
    int time = 0;
    int line = 0;

    cufftComplex cpol;
    int polint;
    int2 tmp;

    int outskip = 0;

    for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
        // NOTE: This is skipping whole words as in will be cast to int2
        skip = iacc * NCHAN_COARSE * NSAMP_PER_PACKET + blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET;
       
        for (int ichunk = 0; ichunk < 7; ++ichunk) {
            line = ichunk * blockDim.x + threadIdx.x;
            chan = line % 7;
            time = line / 7;
            tmp = ((int2*)in)[skip + line];
            accblock[chan * NSAMP_PER_PACKET + time] = tmp.y;
            accblock[NSAMP_PER_PACKET * NCHAN_PER_PACKET + chan * NSAMP_PER_PACKET + time] = tmp.x;
        }

        __syncthreads();
        
        skip = NCHAN_COARSE * NSAMP_PER_PACKET * NACCUMULATE;
        
        outskip = blockIdx.x * 7 * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET;

        for (chan = 0; chan < NCHAN_PER_PACKET; ++chan) {
            /*polaint = accblock[ichan * NSAMP_PER_PACKET + threadIdx.x];
            polbint = accblock[NSAMP_PER_PACKET * NCHAN_PER_PACKET + ichan * NSAMP_PER_PACKET + threadIdx.x];
            pola.x = static_cast<float>(static_cast<short>( ((polaint & 0xff000000) >> 24) | ((polaint & 0xff0000) >> 8) ));
            pola.y = static_cast<float>(static_cast<short>( ((polaint & 0xff00) >> 8) | ((polaint & 0xff) << 8) )); 
	    
            polb.x = static_cast<float>(static_cast<short>( ((polbint & 0xff000000) >> 24) | ((polbint & 0xff0000) >> 8) ));
            polb.y = static_cast<float>(static_cast<short>( ((polbint & 0xff00) >> 8) | ((polbint & 0xff) << 8) )); 
            */

            polint = accblock[chan * NSAMP_PER_PACKET + threadIdx.x];
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));
            out[outskip + threadIdx.x] = cpol;

            polint = accblock[NSAMP_PER_PACKET * NCHAN_PER_PACKET + chan * NSAMP_PER_PACKET + threadIdx.x];
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));

            out[skip + outskip + threadIdx.x] = cpol;

            outskip += NSAMP_PER_PACKET * NACCUMULATE;

        }

    } 

}

__global__ void unpack_new_int2(int2 *__restrict__ in, cufftComplex *__restrict__ out) {

    int skip = 0;

    __shared__ int2 accblock[896];

    int chan = 0;
    int time = 0;
    int line = 0;

    cufftComplex cpol;
    int polint;

    int outskip = 0;

    for (int iacc = 0; iacc < NACCUMULATE; ++iacc) {
        // NOTE: This is skipping whole words as in will be cast to int2
        // skip = iacc * NCHAN_COARSE * NSAMP_PER_PACKET + blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET;

        skip = blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE + iacc * NCHAN_PER_PACKET * NSAMP_PER_PACKET;

        for (int ichunk = 0; ichunk < 7; ++ichunk) {
            line = ichunk * blockDim.x + threadIdx.x;
            chan = line % 7;
            time = line / 7;
            accblock[chan * NSAMP_PER_PACKET + time] = in[skip + line];
        }

        __syncthreads();

        skip = NCHAN_COARSE * NSAMP_PER_PACKET * NACCUMULATE;

        outskip = blockIdx.x * 7 * NSAMP_PER_PACKET * NACCUMULATE + iacc * NSAMP_PER_PACKET;

        for (chan = 0; chan < NCHAN_PER_PACKET; ++chan) {
            polint = accblock[chan * NSAMP_PER_PACKET + threadIdx.x].y;
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));
            out[outskip + threadIdx.x] = cpol;

            polint = accblock[chan * NSAMP_PER_PACKET + threadIdx.x].x;
            cpol.x = static_cast<float>(static_cast<short>( ((polint & 0xff000000) >> 24) | ((polint & 0xff0000) >> 8) ));
            cpol.y = static_cast<float>(static_cast<short>( ((polint & 0xff00) >> 8) | ((polint & 0xff) << 8) ));

            out[skip + outskip + threadIdx.x] = cpol;

            outskip += NSAMP_PER_PACKET * NACCUMULATE;
        }
    }
}

__global__ void unpack_alt(const unsigned int *__restrict__ in, cufftComplex * __restrict__ out) {

    if (threadIdx.x == 1022 || threadIdx.x == 1023)
        return; 

    __shared__ unsigned int accblock[2044];

    int inskip = blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NPOL * NACCUMULATE;
    int outskip = blockIdx.x * NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE;

    int time = 0;
    int chan = 0;
    int line = 0;

    cufftComplex pola, polb;

    int polaint;
    int polbint;


    // NOTE: That will leave last 224 lines unprocessed
    // This can fit in 7 full warps of 32
    for (int iacc = 0; iacc < 113; ++iacc) {

        line = iacc * blockDim.y + threadIdx.y;

        if (line < NCHAN_PER_PACKET * NSAMP_PER_PACKET * NACCUMULATE) {

        chan = threadIdx.y % 7;
        time = threadIdx.y / 7;        
 
        accblock[chan * 146 + time] = in[inskip + threadIdx.y * NPOL];
        accblock[NCHAN_PER_PACKET * 146 + chan * 146 + time] = in[inskip + threadIdx.y * NPOL + 1];
        inskip += 2044;
        
        __syncthreads();

        polbint = accblock[threadIdx.y];
        polaint = accblock[NCHAN_PER_PACKET * 146 + threadIdx.y];

        pola.x = static_cast<float>(static_cast<short>( ((polaint & 0xff000000) >> 24) | ((polaint & 0xff0000) >> 8) ));
        pola.y = static_cast<float>(static_cast<short>( ((polaint & 0xff00) >> 8) | ((polaint & 0xff) << 8) ));

        polb.x = static_cast<float>(static_cast<short>( ((polbint & 0xff000000) >> 24) | ((polbint & 0xff0000) >> 8) ));
        polb.y = static_cast<float>(static_cast<short>( ((polbint & 0xff00) >> 8) | ((polbint & 0xff) << 8) ));

        chan = threadIdx.y / 146;
        time = threadIdx.y % 146; 

        out[outskip + chan * NSAMP_PER_PACKET * NACCUMULATE + time] = pola;
        out[NCHAN_COARSE * NSAMP_PER_PACKET * NACCUMULATE + outskip + chan * NSAMP_PER_PACKET * NACCUMULATE + time] = polb;
        outskip += 146;

        }
    }
}

__global__ void powertimefreq_new_hardcoded(
  cuComplex* __restrict__ in,
  float* __restrict__ out)
{

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE];

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;

  if (lane_idx >= NCHAN_FINE_OUT)
    return;

  int offset = blockIdx.x * NCHAN_COARSE * NPOL * NSAMPS * NCHAN_FINE_IN;
  int out_offset = blockIdx.x * NCHAN_COARSE * NCHAN_FINE_OUT / NCHAN_SUM;

  for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += warpSize)
    {
      float real = 0.0f;
      float imag = 0.0f;
      int coarse_chan_offset = offset + coarse_chan_idx * NPOL * NSAMPS * NCHAN_FINE_IN;

      for (int pol=0; pol<NPOL; ++pol)
      {
        int pol_offset = coarse_chan_offset + pol * NSAMPS * NCHAN_FINE_IN;
        for (int samp=0; samp<NSAMPS; ++samp)
        {
          int samp_offset = pol_offset + samp * NCHAN_FINE_IN;
          cuComplex val = in[samp_offset + lane_idx];
          real += val.x * val.x;
          imag += val.y * val.y;
        }
      }
      int output_idx = coarse_chan_idx * NCHAN_FINE_OUT + lane_idx;

      freq_sum_buffer[output_idx] = real+imag; //scaling goes here
      __syncthreads();

      for (int start_chan=threadIdx.x; start_chan<NCHAN_FINE_OUT*NCHAN_COARSE; start_chan*=blockDim.x)
      {
        if ((start_chan+NCHAN_SUM) > NCHAN_FINE_OUT*NCHAN_COARSE)
          return;
        float sum = freq_sum_buffer[start_chan];
        for (int ii=0; ii<4; ++ii)
        {
          sum += freq_sum_buffer[start_chan + (1<<ii)];
          __syncthreads();
        }
        out[out_offset+start_chan/NCHAN_SUM];
      }
    }
  return;
}

__global__ void DetectScrunchKernel(cuComplex* __restrict__ in, float* __restrict__ out, short nchans)
{
  /**
   * This block is going to do 2 timesamples for all coarse channels.
   * The fine channels are dealt with by the lanes, but on the fine
   * channel read we perform an fft shift and exclude the band edges.
   */
  // gridDim.x should be Nacc * 128 / (32 * nsamps_to_add) == 256

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE]; // 9072 elements

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;
  int pol_offset = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE;
  int coarse_chan_offet = NACCUMULATE * NCHAN_FINE_IN * NSAMPS;
  int block_offset = NCHAN_FINE_IN * NSAMPS_SUMMED * blockIdx.x;
  int nwarps_per_block = blockDim.x/warpSize;


  //Here we calculate indexes for FFT shift.
  int offset_lane_idx = (lane_idx + 19)%32;

  //Here only first 27 lanes are active as we drop
  //5 channels due to the 32/27 oversampling ratio
  if (lane_idx < 27)
    {
      // This warp
      // first sample in inner dimension = (32 * 2 * blockIdx.x)
      // This warp will loop over coarse channels in steps of NWARPS per block coarse_chan_idx (0,335)
      for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += nwarps_per_block)
        {
          float real = 0.0f;
          float imag = 0.0f;
          int base_offset = coarse_chan_offet * coarse_chan_idx + block_offset + offset_lane_idx;

          for (int pol_idx=0; pol_idx<NPOL; ++pol_idx)
            {
              int offset = base_offset + pol_offset * pol_idx;
              for (int sample_idx=0; sample_idx<NSAMPS_SUMMED; ++sample_idx)
                {
                  //Get first channel
                  // IDX = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE * pol_idx
                  // + NACCUMULATE * NCHAN_FINE_IN * NSAMPS * coarse_chan_idx
                  // + blockIdx.x * NCHAN_FINE_IN * NSAMPS_SUMMED
                  // + NCHAN_FINE_IN * sample_idx
                  // + lane_idx;
                  cuComplex val = in[offset + (NCHAN_FINE_IN * sample_idx)]; // load frequencies in right order
                  real += val.x * val.x;
                  imag += val.y * val.y;
                }
              // 3 is the leading dead lane count
              // sketchy
              freq_sum_buffer[coarse_chan_idx*NCHAN_FINE_OUT + lane_idx] = real + imag;
            }
        }
    }

    __syncthreads();

    int saveoff = blockIdx.x * nchans;

    if (threadIdx.x <  (NCHAN_FINE_OUT * NCHAN_COARSE / NCHAN_SUM)) {
        float sum = 0.0;
        for (int chan_idx = threadIdx.x * NCHAN_SUM; chan_idx < (threadIdx.x+1) * NCHAN_SUM; ++chan_idx) {
            sum += freq_sum_buffer[chan_idx];
        }
        out[saveoff + threadIdx.x] = sum;
    }

    return;
}

__global__ void DetectScrunchScaleKernel(cuComplex* __restrict__ in, float* __restrict__ out, short nchans, float *means, float *scales)
{
  /**
   * This block is going to do 2 timesamples for all coarse channels.
   * The fine channels are dealt with by the lanes, but on the fine
   * channel read we perform an fft shift and exclude the band edges.
   */
  // gridDim.x should be Nacc * 128 / (32 * nsamps_to_add) == 256

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE]; // 9072 elements

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;
  int pol_offset = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE;
  int coarse_chan_offet = NACCUMULATE * NCHAN_FINE_IN * NSAMPS;
  int block_offset = NCHAN_FINE_IN * NSAMPS_SUMMED * blockIdx.x;
  int nwarps_per_block = blockDim.x/warpSize;


  //Here we calculate indexes for FFT shift.
  int offset_lane_idx = (lane_idx + 19)%32;

  //Here only first 27 lanes are active as we drop
  //5 channels due to the 32/27 oversampling ratio
  if (lane_idx < 27)
    {
      // This warp
      // first sample in inner dimension = (32 * 2 * blockIdx.x)
      // This warp will loop over coarse channels in steps of NWARPS per block coarse_chan_idx (0,335)
      for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += nwarps_per_block)
        {
          float real = 0.0f;
          float imag = 0.0f;
          int base_offset = coarse_chan_offet * coarse_chan_idx + block_offset + offset_lane_idx;

          for (int pol_idx=0; pol_idx<NPOL; ++pol_idx)
            {
              int offset = base_offset + pol_offset * pol_idx;
              for (int sample_idx=0; sample_idx<NSAMPS_SUMMED; ++sample_idx)
                {
                  //Get first channel
                  // IDX = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE * pol_idx
                  // + NACCUMULATE * NCHAN_FINE_IN * NSAMPS * coarse_chan_idx
                  // + blockIdx.x * NCHAN_FINE_IN * NSAMPS_SUMMED
                  // + NCHAN_FINE_IN * sample_idx
                  // + lane_idx;
                  cuComplex val = in[offset + (NCHAN_FINE_IN * sample_idx)]; // load frequencies in right order
                  real += val.x * val.x;
                  imag += val.y * val.y;
                }
              // 3 is the leading dead lane count
              // sketchy
              freq_sum_buffer[coarse_chan_idx*NCHAN_FINE_OUT + lane_idx] = real + imag;
            }
        }
    }

    __syncthreads();

    int saveoff = blockIdx.x * nchans;

    if (threadIdx.x <  (NCHAN_FINE_OUT * NCHAN_COARSE / NCHAN_SUM)) {
        float sum = 0.0;
        int scaled = 0;
        for (int chan_idx = threadIdx.x * NCHAN_SUM; chan_idx < (threadIdx.x+1) * NCHAN_SUM; ++chan_idx) {
            sum += freq_sum_buffer[chan_idx];
        }
        
        scaled = __float2int_ru((sum - means[threadIdx.x]) * scales[threadIdx.x] + 64.5f);
        if (scaled > 255) {
            scaled = 255;
        } else if (scaled < 0) {
            scaled = 0;
        } 
        //out[saveoff + threadIdx.x] = (unsigned char)scaled;
        // NOTE: That puts the highest frequency first (OUTCHANS - 1 - threadIdx.x)
        out[saveoff + threadIdx.x] = (unsigned char)scaled;
    }

    return;
}

__global__ void DetectScrunchScaleRevKernel(cuComplex* __restrict__ in, float* __restrict__ out, short nchans, float *means, float *scales)
{
  /**
   * This block is going to do 2 timesamples for all coarse channels.
   * The fine channels are dealt with by the lanes, but on the fine
   * channel read we perform an fft shift and exclude the band edges.
   */
  // gridDim.x should be Nacc * 128 / (32 * nsamps_to_add) == 256

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE]; // 9072 elements

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;
  int pol_offset = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE;
  int coarse_chan_offet = NACCUMULATE * NCHAN_FINE_IN * NSAMPS;
  int block_offset = NCHAN_FINE_IN * NSAMPS_SUMMED * blockIdx.x;
  int nwarps_per_block = blockDim.x/warpSize;


  //Here we calculate indexes for FFT shift.
  int offset_lane_idx = (lane_idx + 19)%32;
  
  //Here only first 27 lanes are active as we drop
  //5 channels due to the 32/27 oversampling ratio
  if (lane_idx < 27)
    {
      // This warp
      // first sample in inner dimension = (32 * 2 * blockIdx.x)
      // This warp will loop over coarse channels in steps of NWARPS per block coarse_chan_idx (0,335)
      for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += nwarps_per_block)
        {
          float real = 0.0f;
          float imag = 0.0f;
          int base_offset = coarse_chan_offet * coarse_chan_idx + block_offset + offset_lane_idx;

          for (int pol_idx=0; pol_idx<NPOL; ++pol_idx)
            {
              int offset = base_offset + pol_offset * pol_idx; 
              for (int sample_idx=0; sample_idx<NSAMPS_SUMMED; ++sample_idx)
                {
                  //Get first channel
                  // IDX = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE * pol_idx
                  // + NACCUMULATE * NCHAN_FINE_IN * NSAMPS * coarse_chan_idx
                  // + blockIdx.x * NCHAN_FINE_IN * NSAMPS_SUMMED
                  // + NCHAN_FINE_IN * sample_idx
                  // + lane_idx;
                  cuComplex val = in[offset + (NCHAN_FINE_IN * sample_idx)]; // load frequencies in right order
                  real += val.x * val.x;
                  imag += val.y * val.y;
                }
              // 3 is the leading dead lane count
              // sketchy
              freq_sum_buffer[coarse_chan_idx*NCHAN_FINE_OUT + lane_idx] = real + imag;
            }
        }
    }

    __syncthreads();

    int saveoff = blockIdx.x * nchans;

    if (threadIdx.x <  (NCHAN_FINE_OUT * NCHAN_COARSE / NCHAN_SUM)) {
        float sum = 0.0;
        int scaled = 0;
        for (int chan_idx = threadIdx.x * NCHAN_SUM; chan_idx < (threadIdx.x+1) * NCHAN_SUM; ++chan_idx) {
            sum += freq_sum_buffer[chan_idx];
        }

        scaled = __float2int_ru((sum - means[566 - threadIdx.x]) * scales[566 - threadIdx.x] + 64.5f);
        if (scaled > 255) {
            scaled = 255; 
        } else if (scaled < 0) {
            scaled = 0;
        } 
        //out[saveoff + threadIdx.x] = (unsigned char)scaled;
        // NOTE: That puts the highest frequency first (OUTCHANS - 1 - threadIdx.x)
        out[saveoff + 566 - threadIdx.x] = (unsigned char)scaled;
    }

    return;
}

__global__ void DetectScrunchScaleTruncKernel(cuComplex* __restrict__ in, float* __restrict__ out, short nchans, float *means, float *scales)
{
  /**
   * This block is going to do 2 timesamples for all coarse channels.
   * The fine channels are dealt with by the lanes, but on the fine
   * channel read we perform an fft shift and exclude the band edges.
   */
  // gridDim.x should be Nacc * 128 / (32 * nsamps_to_add) == 256

  __shared__ float freq_sum_buffer[NCHAN_FINE_OUT*NCHAN_COARSE]; // 9072 elements

  int warp_idx = threadIdx.x >> 0x5;
  int lane_idx = threadIdx.x & 0x1f;
  int pol_offset = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE;
  int coarse_chan_offet = NACCUMULATE * NCHAN_FINE_IN * NSAMPS;
  int block_offset = NCHAN_FINE_IN * NSAMPS_SUMMED * blockIdx.x;
  int nwarps_per_block = blockDim.x/warpSize;


  //Here we calculate indexes for FFT shift.
  int offset_lane_idx = (lane_idx + 19)%32;

  //Here only first 27 lanes are active as we drop
  //5 channels due to the 32/27 oversampling ratio

  if (lane_idx < 27)
    {
      // This warp
      // first sample in inner dimension = (32 * 2 * blockIdx.x)
      // This warp will loop over coarse channels in steps of NWARPS per block coarse_chan_idx (0,335)
      for (int coarse_chan_idx = warp_idx; coarse_chan_idx < NCHAN_COARSE; coarse_chan_idx += nwarps_per_block)
        {
          float real = 0.0f;
          float imag = 0.0f;
          int base_offset = coarse_chan_offet * coarse_chan_idx + block_offset + offset_lane_idx;

          for (int pol_idx=0; pol_idx<NPOL; ++pol_idx)
            {
              int offset = base_offset + pol_offset * pol_idx;
              for (int sample_idx=0; sample_idx<NSAMPS_SUMMED; ++sample_idx)
                {
                  //Get first channel
                  // IDX = NCHAN_COARSE * NSAMPS * NCHAN_FINE_IN * NACCUMULATE * pol_idx
                  // + NACCUMULATE * NCHAN_FINE_IN * NSAMPS * coarse_chan_idx
                  // + blockIdx.x * NCHAN_FINE_IN * NSAMPS_SUMMED
                  // + NCHAN_FINE_IN * sample_idx
                  // + lane_idx;
                  cuComplex val = in[offset + (NCHAN_FINE_IN * sample_idx)]; // load frequencies in right order
                  real += val.x * val.x;
                  imag += val.y * val.y;
                }
              // 3 is the leading dead lane count
              // sketchy
              freq_sum_buffer[coarse_chan_idx*NCHAN_FINE_OUT + lane_idx] = real + imag;
            }
        }
    }

    __syncthreads();

    int saveoff = blockIdx.x * nchans;

    int skipbottom = 28 * NCHAN_SUM;
    if (threadIdx.x < 512) {
        float sum = 0.0;
        int scaled = 0;
        for (int chan_idx = threadIdx.x * NCHAN_SUM; chan_idx < (threadIdx.x+1) * NCHAN_SUM; ++chan_idx) {
            sum += freq_sum_buffer[skipbottom + chan_idx];
        }

        scaled = __float2int_ru((sum - means[511 - threadIdx.x]) * scales[511 - threadIdx.x] + 64.5f);
        if (scaled > 255) {
            scaled = 255;
        } else if (scaled < 0) {
            scaled = 0;
        }
        //out[saveoff + threadIdx.x] = (unsigned char)scaled;
        // NOTE: That puts the highest frequency first (OUTCHANS - 1 - threadIdx.x)
        out[saveoff + 511 - threadIdx.x] = (unsigned char)scaled;
    }

    return;
}


__global__ void GetPowerAddTimeKernel(cufftComplex* __restrict__ in, float* __restrict__ out, unsigned int jump, unsigned int factort, unsigned int acc) {

    int idx1, idx2;
    int outidx;
    int skip1, skip2;
    float power1, power2;

    for (int iac = 0; iac < acc; iac++) {
        skip1 = iac * 336 * 128 * 2;
        skip2 = iac * 336 * 27;
            for (int ichan = 0; ichan < 7; ichan++) {
            outidx = skip2 + 7 * 27 * blockIdx.x + ichan * 27 + threadIdx.x;
            out[outidx] = (float)0.0;
            out[outidx + jump] = (float)0.0;
            out[outidx + 2 * jump] = (float)0.0;
            out[outidx + 3 * jump] = (float)0.0;

            idx1 = skip1 + 256 * (blockIdx.x * 7 + ichan);

            for (int itime = 0; itime < factort; itime++) {
                idx2 = threadIdx.x + itime * 32;
                power1 = (in[idx1 + idx2].x * in[idx1 + idx2].x + in[idx1 + idx2].y * in[idx1 + idx2].y);
                power2 = (in[idx1 + 128 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + 128 + idx2].y * in[idx1 + 128 + idx2].y);
                out[outidx] += (power1 + power2);
                out[outidx + jump] += (power1 - power2);
                out[outidx + 2 * jump] += (2 * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].x + in[idx1 + idx2].y * in[idx1 + 128 + idx2].y));
                out[outidx + 3 * jump] += (2 * (in[idx1 + idx2].x * in[idx1 + 128 + idx2].y - in[idx1 + idx2].y * in[idx1 + 128 + idx2].x));
            }
        }
    }
}

__global__ void GetScaleFactorsKernel(float *indata, float *base, float *stdev, float *factors, int nchans, int processed) {
/*
    // NOTE: Filterbank file format coming in
    //float mean = indata[threadIdx.x];
    float mean = 0.0f;
    // NOTE: Depending whether I save STD or VAR at the end of every run
    // float estd = stdev[threadIdx.x];
    float estd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float oldmean = base[threadIdx.x];
    //float estd = 0.0f;
    //float oldmean = 0.0;
    float val = 0.0f;
    float diff = 0.0;
    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - oldmean;
        mean = oldmean + diff * factors[processed + isamp + 1];
        estd += diff * (val - mean);
        oldmean = mean;
    }
    base[threadIdx.x] = mean;
    stdev[threadIdx.x] = sqrtf(estd / (float)(processed + 2 * NACCUMULATE - 1.0f));
    // stdev[threadIdx.x] = estd;
*/
    float chmean = 0.0f;
    float chestd = 0.0f;
    float val = 0.0;
    float diff = 0.0;

    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - chmean;
        chmean += diff * factors[isamp + 1];
        chestd += diff * (val - chmean);
    }

    float oldmean = base[threadIdx.x];
    float oldestd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float newestd = 0.0f;

    diff = chmean - oldmean;
    base[threadIdx.x] = oldmean + diff * (float)(2.0f * NACCUMULATE) / (float)(processed + 2.0 * NACCUMULATE);
    newestd = oldestd + chestd + diff * diff * (float)(2.0f * NACCUMULATE) * (float)processed / (float)(processed + 2.0 * NACCUMULATE);
    stdev[threadIdx.x] = sqrt(newestd / (float)(processed + 2 * NACCUMULATE - 1.0f));

}

__global__ void GetScaleFactorsDivKernel(float *indata, float *base, float *stdev, int nchans, int processed) {
/*
    // NOTE: Filterbank file format coming in
    //float mean = indata[threadIdx.x];
    float mean = 0.0f;
    // NOTE: Depending whether I save STD or VAR at the end of every run
    // float estd = stdev[threadIdx.x];
    float estd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float oldmean = base[threadIdx.x];
    //float estd = 0.0f;
    //float oldmean = 0.0;
    float val = 0.0f;
    float diff = 0.0;
    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - oldmean;
        mean = oldmean + diff * factors[processed + isamp + 1];
        estd += diff * (val - mean);
        oldmean = mean;
    }
    base[threadIdx.x] = mean;
    stdev[threadIdx.x] = sqrtf(estd / (float)(processed + 2 * NACCUMULATE - 1.0f));
    // stdev[threadIdx.x] = estd;
*/
    float chmean = 0.0f;
    float chestd = 0.0f;
    float val = 0.0;
    float diff = 0.0;

    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - chmean;
        chmean += diff / (isamp + 1);
        chestd += diff * (val - chmean);
    }

    float oldmean = base[threadIdx.x];
    float oldestd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float newestd = 0.0f;

    diff = chmean - oldmean;
    base[threadIdx.x] = oldmean + diff * (float)(2.0f * NACCUMULATE) / (float)(processed + 2.0 * NACCUMULATE);
    newestd = oldestd + chestd + diff * diff * (float)(2.0f * NACCUMULATE) * (float)processed / (float)(processed + 2.0 * NACCUMULATE);
    stdev[threadIdx.x] = sqrt(newestd / (float)(processed + 2 * NACCUMULATE - 1.0f));

}

__global__ void GetScaleFactorsDoubleKernel(float *indata, float *base, float *stdev, int nchans) {
/*
    // NOTE: Filterbank file format coming in
    //float mean = indata[threadIdx.x];
    float mean = 0.0f;
    // NOTE: Depending whether I save STD or VAR at the end of every run
    // float estd = stdev[threadIdx.x];
    float estd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float oldmean = base[threadIdx.x];
    //float estd = 0.0f;
    //float oldmean = 0.0;
    float val = 0.0f;
    float diff = 0.0;
    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - oldmean;
        mean = oldmean + diff * factors[processed + isamp + 1];
        estd += diff * (val - mean);
        oldmean = mean;
    }
    base[threadIdx.x] = mean;
    stdev[threadIdx.x] = sqrtf(estd / (float)(processed + 2 * NACCUMULATE - 1.0f));
    // stdev[threadIdx.x] = estd;
*/
/*
    float chmean = 0.0f;
    float chestd = 0.0f;
    float val = 0.0;
    float diff = 0.0;

    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        val = indata[isamp * nchans + threadIdx.x];
        diff = val - chmean;
        chmean += diff / (isamp + 1);
        chestd += diff * (val - chmean);
    }

    float oldmean = base[threadIdx.x];
    float oldestd = stdev[threadIdx.x] * stdev[threadIdx.x] * (processed - 1.0f);
    float newestd = 0.0f;

    diff = chmean - oldmean;
    base[threadIdx.x] = oldmean + diff * (float)(2.0f * NACCUMULATE) / (float)(processed + 2.0 * NACCUMULATE);
    newestd = oldestd + chestd + diff * diff * (float)(2.0f * NACCUMULATE) * (float)processed / (float)(processed + 2.0 * NACCUMULATE);
    stdev[threadIdx.x] = sqrt(newestd / (float)(processed + 2 * NACCUMULATE - 1.0f));
*/
    float sum = indata[threadIdx.x];

    for (int isamp = 1; isamp < 2 * NACCUMULATE; ++isamp) {
        sum += indata[isamp * nchans + threadIdx.x];
    }

    float mean = sum / (float)(2.0 * NACCUMULATE);

    base[threadIdx.x] = mean;

    float sumsq = 0.0f;
    float diff = 0.0;

    for (int isamp = 0; isamp < 2 * NACCUMULATE; ++isamp) {
        diff = indata[isamp * nchans + threadIdx.x] - mean;
        sumsq += diff * diff;   
    }

    stdev[threadIdx.x] = sqrt(sumsq / (float)(NACCUMULATE - 1.0));
}


struct FactorFunctor {
    __host__ __device__ float operator()(float val) {
        return val != 0.0f ? 1.0f/val : val;
    }
};

int main(int argc, char* argv[]) {

    unsigned char *rawbuffer = new unsigned char[7168 * NFPGAS * NACC];

    thrust::device_vector<unsigned char> rawdata(7168 * NFPGAS * NACCUMULATE);
    // NOTE: 336 coarse channels * 32 fine channels * 4 time samples * 2 polarisations
    thrust::device_vector<cuComplex> input(336*32*4*2*NACCUMULATE);
    // NOTE: 336 coarse channels * 27 fine channels
    thrust::device_vector<float> output(336*27*NACCUMULATE);

    thrust::device_vector<float> means(567);
    thrust::device_vector<float> scales(567);

    thrust::device_vector<float> factors(NACCUMULATE * 2);
    thrust::sequence(factors.begin(), factors.end());
    thrust::transform(factors.begin(), factors.end(), factors.begin(), FactorFunctor());

    // NOTE: Benchmarking the unpacker kernel

    cudaArray *rawarray;
    cudaChannelFormatDesc cdesc;
    cdesc = cudaCreateChannelDesc<int2>();
    cudaMallocArray(&rawarray, &cdesc, 7, 48 * 128 * NACCUMULATE);

    cudaResourceDesc rdesc;
    memset(&rdesc, 0, sizeof(cudaResourceDesc));
    rdesc.resType = cudaResourceTypeArray;
    rdesc.res.array.array = rawarray;

    cudaTextureDesc tdesc;
    memset(&tdesc, 0, sizeof(cudaTextureDesc));
    tdesc.addressMode[0] = cudaAddressModeClamp;
    tdesc.filterMode = cudaFilterModePoint;
    tdesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &rdesc, &tdesc, NULL);

    cudaMemcpyToArray(rawarray, 0, 0, rawbuffer, 7168 * 48 * NACCUMULATE * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 rearrange_b(1,48,1);
    dim3 rearrange_t(7,1,1);

    dim3 unpackt(2, 128, 1);
    dim3 unpacka(1, 1024, 1);
    dim3 unpackb(48, 2, 1);

    // ##################################
    // ### UNPACK KERNEL BENCHMARKING ###
    // ##################################
 
    std::chrono::time_point<std::chrono::system_clock> unpackstart, unpackend;
    std::chrono::duration<double> unpackelapsed;

    unpackstart = std::chrono::system_clock::now();
    // NOTE: Unpacking with texture memory
    for (int ii = 0; ii < 32; ++ii) {
        unpack_original_tex<<<rearrange_b, rearrange_t, 0>>>(texObj, thrust::raw_pointer_cast(input.data()), NACCUMULATE);
        gpuErrchk(cudaDeviceSynchronize());
    }
    unpackend = std::chrono::system_clock::now();
    unpackelapsed = unpackend - unpackstart;
    cout << "Unpacking with texture memory: " << unpackelapsed.count() / 32.0 << "s" << endl;

    // NOTE: Unpacking with shared memory
    unpackstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        unpack_new<<<48, 128, 0>>>(reinterpret_cast<unsigned int*>(thrust::raw_pointer_cast(rawdata.data())), thrust::raw_pointer_cast(input.data()));       
        gpuErrchk(cudaDeviceSynchronize());
    }
    unpackend = std::chrono::system_clock::now();
    unpackelapsed = unpackend - unpackstart;
    cout << "Unpacking with shared memory: " << unpackelapsed.count() / 32.0 << "s" << endl; 

    // NOTE: Unpacking with shared memory with cast to int2 in the function call
    unpackstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        unpack_new_int2<<<48, 128, 0>>>(reinterpret_cast<int2*>(thrust::raw_pointer_cast(rawdata.data())), thrust::raw_pointer_cast(input.data()));
        gpuErrchk(cudaDeviceSynchronize());
    }
    unpackend = std::chrono::system_clock::now();
    unpackelapsed = unpackend - unpackstart;
    cout << "Unpacking with shared memory with cast to int2 in the function call: " << unpackelapsed.count() / 32.0 << "s" << endl;

    // NOTE: Unpacking with shared memory with the alternative incoming buffer arrangement
    unpackstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        unpack_alt<<<48, unpacka, 0>>>(reinterpret_cast<unsigned int*>(thrust::raw_pointer_cast(rawdata.data())), thrust::raw_pointer_cast(input.data()));
        gpuErrchk(cudaDeviceSynchronize());
    }
    unpackend = std::chrono::system_clock::now();
    unpackelapsed = unpackend - unpackstart;
    cout << "Unpacking with shared memory with the alternative incoming buffer arrangement: " << unpackelapsed.count() / 32.0 << "s" << endl;

    std::chrono::time_point<std::chrono::system_clock> powerstart, powerend;
    std::chrono::duration<double> powerelapsed;

    // #################################
    // ### POWER KERNEL BENCHMARKING ###
    // #################################

    // NOTE: Unoptimised power kernel with time averaging only
    powerstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        GetPowerAddTimeKernel<<<48, 27, 0>>>(thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), 0, NSAMPS, NACCUMULATE);
        gpuErrchk(cudaDeviceSynchronize());
    }
    powerend = std::chrono::system_clock::now();
    powerelapsed = powerend - powerstart;

    cout << "Unoptimised power kernel: " << powerelapsed.count() / 32.0 << "s" << endl;
    
    // NOTE: Optimised power kernel with shared memory with time and frequency averaging
    powerstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        DetectScrunchKernel<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()), 567);
        gpuErrchk(cudaDeviceSynchronize());
    }
    powerend = std::chrono::system_clock::now();
    powerelapsed = powerend - powerstart;

    cout << "Optimised power kernel: " << powerelapsed.count() / 32.0 << "s" << endl;

    // NOTE: Optimised power kernel with scaling
    powerstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        DetectScrunchScaleKernel<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()), 567,
                                                            thrust::raw_pointer_cast(means.data()), thrust::raw_pointer_cast(scales.data()));
        gpuErrchk(cudaDeviceSynchronize());
    }
    powerend = std::chrono::system_clock::now();
    powerelapsed = powerend - powerstart;

    cout << "Optimised power kernel with scaling: " << powerelapsed.count() / 32.0 << "s" << endl;

    // NOTE: Optimised power kernel with scaling and reversed frequency
    powerstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        DetectScrunchScaleRevKernel<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()), 567,
                                                            thrust::raw_pointer_cast(means.data()), thrust::raw_pointer_cast(scales.data()));
        gpuErrchk(cudaDeviceSynchronize());
    }
    powerend = std::chrono::system_clock::now();
    powerelapsed = powerend - powerstart;

    cout << "Optimised power kernel with scaling and reversed frequency: " << powerelapsed.count() / 32.0 << "s" << endl;

    // NOTE: Optimised power kernel with scaling truncated and reversed frequency
    powerstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        DetectScrunchScaleTruncKernel<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()), 512,
                                                            thrust::raw_pointer_cast(means.data()), thrust::raw_pointer_cast(scales.data()));
        gpuErrchk(cudaDeviceSynchronize());
    }
    powerend = std::chrono::system_clock::now();
    powerelapsed = powerend - powerstart;

    cout << "Optimised power kernel with scaling, truncated and reversed frequency: " << powerelapsed.count() / 32.0 << "s" << endl;


    // #################################
    // ### SCALE KERNEL BENCHMARKING ###
    // #################################

    std::chrono::time_point<std::chrono::system_clock> scalestart, scaleend;
    std::chrono::duration<double> scaleelapsed;

    
    // NOTE: Double-pass scaling factors kernel
    scalestart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        GetScaleFactorsDoubleKernel<<<1, 512, 0>>>(thrust::raw_pointer_cast(output.data()), thrust::raw_pointer_cast(means.data()), thrust::raw_pointer_cast(scales.data()), 512);
        gpuErrchk(cudaDeviceSynchronize());
    }
    scaleend = std::chrono::system_clock::now();
    scaleelapsed = scaleend - scalestart;

    cout << "Double pass scaling factors kernel with division: " << scaleelapsed.count() / 32.0 << "s" << endl;


    // NOTE: Unoptimised scaling factors kernel with division
    scalestart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        GetScaleFactorsDivKernel<<<1, 512, 0>>>(thrust::raw_pointer_cast(output.data()), thrust::raw_pointer_cast(means.data()), thrust::raw_pointer_cast(scales.data()),
                                                                    512, 0);
        gpuErrchk(cudaDeviceSynchronize());
    }
    scaleend = std::chrono::system_clock::now();
    scaleelapsed = scaleend - scalestart;

    cout << "Unoptimised scaling factors kernel with division: " << scaleelapsed.count() / 32.0 << "s" << endl;

    // NOTE: Optimised scaling factors kernel 
    scalestart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 32; ++ii) {
        GetScaleFactorsKernel<<<1, 512, 0>>>(thrust::raw_pointer_cast(output.data()), thrust::raw_pointer_cast(means.data()), thrust::raw_pointer_cast(scales.data()),
                                                                    thrust::raw_pointer_cast(factors.data()), 512, 0);
        gpuErrchk(cudaDeviceSynchronize());
    }
    scaleend = std::chrono::system_clock::now();
    scaleelapsed = scaleend - scalestart;

    cout << "Optimised scaling factors kernel: " << scaleelapsed.count() / 32.0 << "s" << endl;



}

