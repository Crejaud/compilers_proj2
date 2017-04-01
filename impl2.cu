#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define TRUE 1
#define FALSE 0

/* Find number of edges to process */
__global__ void find_num_edges_to_process(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            unsigned int *num_edges_to_process) {
  unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int thread_num = blockDim.x * gridDim.x;

  unsigned int warp_id = thread_id / 32;
  unsigned int warp_num = thread_num % 32 == 0 ? thread_num / 32 : thread_num / 32 + 1;

  unsigned int load = edges_length % warp_num == 0 ? edges_length / warp_num : edges_length / warp_num + 1;
  unsigned int beg = load * warp_id;
  unsigned int end = min(edges_length, beg + load);
  unsigned int lane = thread_id % 32;
  beg += lane;

  for (unsigned int i = beg; i < end; i += 32) {
    if (distance_cur[src[i]] != distance_prev[src[i]]) {
      atomicAdd(&num_edges_to_process[warp_id], 1);
    }
  }
}

/* Work efficient edge process out of core with no shared memory */
__global__ void work_efficient_out_of_core(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            int *noChange,
                            unsigned int *num_edges_to_process,
                            unsigned int *warp_offsets,
                            unsigned int *T,
                            unsigned int *T_length) {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id / 32;

    // range is warp_offsets[warp_id] to warp_offsets[warp_id] + num_edges_to_process[warp_id] - 1
    unsigned int beg = warp_offsets[warp_id];
    unsigned int end = beg + num_edges_to_process[warp_id];
    unsigned int lane = thread_id % 32;
    beg += lane;

    for (unsigned int i = beg; i < end; i += 32) {
      //printf("warp_id %u | beg %u | end %u | lane %u\n", warp_id, beg, end);
      unsigned int u = src[T[i]];
      unsigned int v = dest[T[i]];
      unsigned int w = weight[T[i]];

      if (distance_prev[u] + w < distance_prev[v]) {
        // relax
        unsigned int old_distance = atomicMin(&distance_cur[v], distance_prev[u] + w);

        // test for change!
        if (old_distance != distance_cur[v]) {
          atomicMin(noChange, FALSE);
        }
      }
    }
}

/* This kernel function will perform block level parallel prefix sum to get
 * the offset of every warp's first to-process edge in the final to-process edge
 * list T */
__global__ void filtering(int edges_length,
                          unsigned int *num_edges_to_process,
                          unsigned int *warp_offsets,
                          unsigned int *distance_prev,
                          unsigned int *distance_cur,
                          unsigned int *T,
                          unsigned int *T_length,
                          unsigned int *src) {
  extern __shared__ unsigned int smem_warp_offsets[ ];
  // we can assume it will fit since there will be at most 64 warps

  __syncthreads();

  int offset = 1;

  if (threadIdx.x*2+1 < blockDim.x) {
    // do parallel prefix sum!

    smem_warp_offsets[2*threadIdx.x] = num_edges_to_process[2*threadIdx.x];
    smem_warp_offsets[2*threadIdx.x+1] = num_edges_to_process[2*threadIdx.x+1];

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();

      if (threadIdx.x < d) {
        int ai = offset*(2*threadIdx.x+1)-1;
        int bi = offset*(2*threadIdx.x+2)-1;

        smem_warp_offsets[bi] += smem_warp_offsets[ai];
      }
      offset *= 2;
    }

    if (threadIdx.x == 0) {
      smem_warp_offsets[blockDim.x - 1] = 0;
    }

    for (int d = 1; d < blockDim.x; d *= 2) {
      offset >>= 1;
      __syncthreads();

      if (threadIdx.x < d) {
        int ai = offset*(2*threadIdx.x+1)-1;
        int bi = offset*(2*threadIdx.x+2)-1;

        unsigned int t = smem_warp_offsets[ai];
        smem_warp_offsets[ai] = smem_warp_offsets[bi];
        smem_warp_offsets[bi] += t;
      }
    }

    __syncthreads();

    warp_offsets[2*threadIdx.x] = smem_warp_offsets[2*threadIdx.x];
    warp_offsets[2*threadIdx.x+1] = smem_warp_offsets[2*threadIdx.x + 1];
  }

  __syncthreads();

  // if (threadIdx.x == 0) {
  //   for(int j = 0; j < blockDim.x; j++) {
  //     printf("num_edges_to_process[%u] = %u | warp_offsets[%u] = %u\n", j, num_edges_to_process[j], j, warp_offsets[j]);
  //   }
  // }

  // the new length of T is the total number of edges to process!
  *T_length = warp_offsets[blockDim.x-1] + num_edges_to_process[blockDim.x-1];
  //printf("blockDim = %u | T becomes %u, since %u and %u | warp_id = %u\n", blockDim.x, *T_length, warp_offsets[blockDim.x-1], num_edges_to_process[blockDim.x-1], threadIdx.x);

  // parallel prefix sum is done and warp_offsets is complete

  // now we must create T

  unsigned int cur_offset = warp_offsets[threadIdx.x];

  unsigned int load = edges_length % blockDim.x == 0 ? edges_length / blockDim.x : edges_length / blockDim.x + 1;
  unsigned int beg = load * threadIdx.x;
  unsigned int end = min(edges_length, beg + load);

  for (unsigned int i = beg; i < end; i++) {
    // done with filling in T
    if (cur_offset >= warp_offsets[threadIdx.x] + num_edges_to_process[threadIdx.x])
      return;

    //if they're not the same
    if (distance_cur[src[i]] != distance_prev[src[i]]) {
      //printf("found src: %u, put into T[%u] |  range is [%u, %u]\n", src[i], cur_offset, warp_offsets[threadIdx.x], warp_offsets[threadIdx.x] + num_edges_to_process[threadIdx.x] - 1);
      T[cur_offset] = i;
      cur_offset++;
    }
  }
}

/* Work efficient edge process in core */
__global__ void work_efficient_in_core(unsigned int edges_length,
                            unsigned int vertices_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance,
                            int *noChange,
                            unsigned int *num_edges_to_process,
                            unsigned int *warp_offsets,
                            unsigned int *T,
                            unsigned int *T_length) {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id / 32;

    // range is warp_offsets[warp_id] to warp_offsets[warp_id] + num_edges_to_process[warp_id] - 1
    unsigned int beg = warp_offsets[warp_id];
    unsigned int end = beg + num_edges_to_process[warp_id];
    unsigned int lane = thread_id % 32;
    beg += lane;

    for (unsigned int i = beg; i < end; i += 32) {
      //printf("warp_id %u | beg %u | end %u | lane %u\n", warp_id, beg, end);
      unsigned int u = src[T[i]];
      unsigned int v = dest[T[i]];
      unsigned int w = weight[T[i]];

      unsigned int temp_dest = distance[u] + w;
      if (temp_dest < distance[v]) {
        // relax
        unsigned int old_distance = atomicMin(&distance[v], temp_dest);

        // test for change!
        if (old_distance != distance[v]) {
          atomicMin(noChange, FALSE);
        }
      }
    }
}

void neighborHandler(int blockSize, int blockNum,
  int sync, int smem, unsigned int *distance_cur,
  unsigned int *edges_src, unsigned int *edges_dest,
  unsigned int *edges_weight, unsigned int edges_length,
  unsigned int vertices_length){
  /* Will use these arrays instead of a vector
  * edges_src : array of all edges (indexed 0 to n) where the value is the vertex source index of the edge (since edges are directed)
  * edges_dest : same as above, except it tells the vertex destination index
  * edges_weight : same as above, except it tells the edge's weight
  * distance_prev : array of all vertices with their distance values
  * distance_cur : same as above
  */

  /* Allocate here... */
  unsigned int *distance_prev = (unsigned int *) malloc(vertices_length * sizeof(unsigned int));
  unsigned int *num_edges_to_process;
  int *noChange = (int *) malloc(sizeof(int));
  unsigned int *T, *T_length;
  unsigned int *warp_offsets;

  // timers
  double filteringTime = 0, computingTime = 0;

  // this is the total number of warps
  unsigned int warp_num = blockSize * blockNum % 32 == 0 ? blockSize * blockNum / 32 : blockSize * blockNum / 32 + 1;

  num_edges_to_process = (unsigned int *) malloc( warp_num * sizeof(unsigned int));
  warp_offsets = (unsigned int *) malloc( warp_num * sizeof(unsigned int));

  for(int i = 0; i < warp_num; i++) {
    num_edges_to_process[i] = 0;
    warp_offsets[i] = 0;
  }

  *noChange = TRUE;

  unsigned int *cuda_edges_src, *cuda_edges_dest, *cuda_edges_weight;
  unsigned int *cuda_distance_prev, *cuda_distance_cur;
  unsigned int *cuda_T, *cuda_T_length;
  unsigned int *cuda_num_edges_to_process;
  unsigned int *cuda_warp_offsets;
  int *cuda_noChange;

  // the distance to the first vertex is always 0
  distance_prev[0] = -1;
  distance_cur[0] = 0;

  // setting an unsigned int to -1 will set it to the maximum value!
  for (int i = 1; i < vertices_length; i++) {
    distance_prev[i] = -1;
    distance_cur[i] = -1;
  }

  // malloc edges arrays
  T = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
  T_length = (unsigned int *) malloc(sizeof(unsigned int));

  // get values for each array
  for (unsigned int edge_index = 0; edge_index < edges_length; edge_index++) {
    if (edges_src[edge_index] == 0) {
      unsigned int load = edges_length % warp_num == 0 ? edges_length / warp_num : edges_length / warp_num + 1;
      unsigned int warp_id = edge_index / load;
      num_edges_to_process[warp_id]++;;
    }
  }

  cudaMalloc((void **)&cuda_edges_src, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_edges_dest, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_edges_weight, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_distance_prev, vertices_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_distance_cur, vertices_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_noChange, sizeof(int));
  cudaMalloc((void **)&cuda_T, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_T_length, sizeof(unsigned int));
  cudaMalloc((void **)&cuda_num_edges_to_process, warp_num * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_warp_offsets, warp_num * sizeof(unsigned int));

  cudaMemcpy(cuda_edges_src, edges_src, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_edges_dest, edges_dest, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_edges_weight, edges_weight, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_distance_prev, distance_prev, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_T, T, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_T_length, T_length, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_num_edges_to_process, num_edges_to_process, warp_num * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_warp_offsets, warp_offsets, warp_num * sizeof(unsigned int), cudaMemcpyHostToDevice);


  /*
   * Do all the things here!
   **/
  // sync is out of core
  if (sync == 0) {
    for (unsigned int i = 1; i < vertices_length; i++) {
      //printf("pass %u, starting filtering\n", i);
      setTime();
      filtering<<<1, warp_num, warp_num * sizeof(unsigned int)>>>(edges_length,
                                cuda_num_edges_to_process,
                                cuda_warp_offsets,
                                cuda_distance_prev,
                                cuda_distance_cur,
                                cuda_T,
                                cuda_T_length,
                                cuda_edges_src);

      //printf("filtering done\n");
      filteringTime += getTime();

      // printf("T_length = %u\n", *T_length);
      // //print out T for testing
      // for (unsigned int j = 0; j < *T_length; j++) {
      //   printf("T[%u] = %u\n", j, T[j]);
      // }
      //printf("past forloop\n");

      //printf("past nochange reset\n");
      setTime();

      // get current distance and copy it to both cuda_distance_prev and cuda_distance_cur
      cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      // for (unsigned int j = 0; j < vertices_length; j++) {
      //   printf("distance_cur[%u] = %u\n", j, distance_cur[j]);
      // }
      cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

      //printf("past distance reset\n");

      //printf("starting outcore\n");

      //printf("pass %u\n", i);
      work_efficient_out_of_core<<<blockNum, blockSize>>>(edges_length, cuda_edges_src,
                                          cuda_edges_dest, cuda_edges_weight,
                                          cuda_distance_prev, cuda_distance_cur,
                                          cuda_noChange,
                                          cuda_num_edges_to_process,
                                          cuda_warp_offsets,
                                          cuda_T, cuda_T_length);

      computingTime += getTime();
      //printf("outcore done\n");

      cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
      if (*noChange == TRUE) break;
      *noChange = TRUE;
      cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

      cudaMemcpy(num_edges_to_process, cuda_num_edges_to_process, warp_num * sizeof(unsigned int), cudaMemcpyDeviceToHost);

      // reset these values back to 0
      for(unsigned int j = 0; j < warp_num; j++) {
        num_edges_to_process[j] = 0;
      }

      cudaMemcpy(cuda_num_edges_to_process, num_edges_to_process, warp_num * sizeof(unsigned int), cudaMemcpyHostToDevice);

      //printf("there was change\n");

      find_num_edges_to_process<<<blockNum, blockSize>>>(edges_length,
                                  cuda_edges_src,
                                  cuda_distance_prev,
                                  cuda_distance_cur,
                                  cuda_num_edges_to_process);
    }
  }
  // sync is in core
  else if (sync == 1) {
    for (unsigned int i = 1; i < vertices_length; i++) {
      setTime();
      filtering<<<1, warp_num, warp_num * sizeof(unsigned int)>>>(edges_length,
                                cuda_num_edges_to_process,
                                cuda_warp_offsets,
                                cuda_distance_prev,
                                cuda_distance_cur,
                                cuda_T,
                                cuda_T_length,
                                cuda_edges_src);

      //printf("filtering done\n");
      filteringTime += getTime();

      cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      // for (unsigned int j = 0; j < vertices_length; j++) {
      //   printf("distance_cur[%u] = %u\n", j, distance_cur[j]);
      // }
      cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

      setTime();
      work_efficient_in_core<<<blockNum, blockSize>>>(edges_length, vertices_length,
                                          cuda_edges_src, cuda_edges_dest,
                                          cuda_edges_weight, cuda_distance_cur,
                                          cuda_noChange,
                                          cuda_num_edges_to_process,
                                          cuda_warp_offsets,
                                          cuda_T,
                                          cuda_T_length);
      computingTime += getTime();

      cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
      if (*noChange == TRUE) break;
      *noChange = TRUE;
      cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

      cudaMemcpy(num_edges_to_process, cuda_num_edges_to_process, warp_num * sizeof(unsigned int), cudaMemcpyDeviceToHost);

      // reset these values back to 0
      for(unsigned int j = 0; j < warp_num; j++) {
        num_edges_to_process[j] = 0;
      }

      cudaMemcpy(cuda_num_edges_to_process, num_edges_to_process, warp_num * sizeof(unsigned int), cudaMemcpyHostToDevice);

      //printf("there was change\n");

      find_num_edges_to_process<<<blockNum, blockSize>>>(edges_length,
                                  cuda_edges_src,
                                  cuda_distance_prev,
                                  cuda_distance_cur,
                                  cuda_num_edges_to_process);
    }
  }

  else {
    // no syncing
    printf("No syncing tag\n");
    exit(1);
  }

  cudaDeviceSynchronize();
  std::cout << "The total computation kernel time on GPU is " << computingTime << " milli-seconds\n";
  std::cout << "The total filtering kernel time on GPU is " << filteringTime << " milli-seconds\n";

  cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int),
           cudaMemcpyDeviceToHost);

  /* Deallocate. */
  cudaFree(cuda_edges_src);
  cudaFree(cuda_edges_dest);
  cudaFree(cuda_edges_weight);
  cudaFree(cuda_distance_prev);
  cudaFree(cuda_distance_cur);
  cudaFree(cuda_noChange);
  cudaFree(cuda_num_edges_to_process);
  cudaFree(cuda_T);
  cudaFree(cuda_T_length);

  free(distance_prev);
  free(noChange);
  free(num_edges_to_process);
  free(T);
  free(T_length);
}
