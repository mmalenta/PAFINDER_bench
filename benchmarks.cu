#include <chrono>
#include <iostream>

#include "thrust/device_vector.h"
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

int main(int argc, char* argv[]) {

    unsigned char *rawbuffer = new unsigned char[7168 * NFPGAS * NACC];

    thrust::device_vector<unsigned char> rawdata(7168 * NFPGAS * NACCUMULATE);
    // NOTE: 336 coarse channels * 32 fine channels * 4 time samples * 2 polarisations
    thrust::device_vector<cuComplex> input(336*32*4*2*NACCUMULATE);
    // NOTE: 336 coarse channels * 27 fine channels
    thrust::device_vector<float> output(336*27*NACCUMULATE);

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

/*    std::chrono::time_point<std::chrono::system_clock> powerstart, powerend;
    std::chrono::duration<double> powerelapsed;

    powerstart = std::chrono::system_clock::now();
    for (int ii = 0; ii < 1; ++ii) {
        powertimefreq_new_hardcoded<<<NACCUMULATE,1024,0>>>(thrust::raw_pointer_cast(input.data()),thrust::raw_pointer_cast(output.data()));
        gpuErrchk(cudaDeviceSynchronize());
    }
    powerend = std::chrono::system_clock::now();
    powerelapsed = powerend - powerstart;

    cout << "Optimised power kernel: " << powerelapsed.count() / 32.0 << "s" << endl;
*/
}

