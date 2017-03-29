#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

/* Work efficient edge process out of core with no shared memory */
__global__ void work_efficient_out_of_core(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            int *noChange,
                            int *is_distance_infinity_prev,
                            int *is_distance_infinity_cur,
                            int *should_update_edge) {
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
      unsigned int u = src[dataid];
      unsigned int v = dest[dataid];
      unsigned int w = weight[dataid];
      
      if (is_distance_infinity_prev[u] == TRUE) {
        continue;
      }
      //printf("%u isn't infinite distance\n", u);
      if (should_update_edge[dataid] == TRUE && distance_prev[u] + w < distance_prev[v]) {
        // relax
        //printf("%u %u\n", distance_cur[v], distance_prev[u] + w);
        unsigned int old_distance = atomicMin(&distance_cur[v], distance_prev[u] + w);
        atomicMin(&is_distance_infinity_cur[v], FALSE);
        //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w, is_distance_infinity[v]);
        // test for a change!
        if (old_distance != distance_cur[v]) {
          //printf("there is change\n");
          atomicMin(noChange, FALSE);
        }
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
                            int *is_distance_infinity,
                            int *should_update_edge) {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int iter = edges_length % thread_num == 0 ? edges_length / thread_num : edges_length / thread_num + 1;

    for (unsigned int j = 1; j < vertices_length; j++) {
      __syncthreads();
      for (unsigned int i = 0; i < iter; i++) {
        __syncthreads();
        unsigned int dataid = thread_id + i * thread_num;
        if (dataid >= edges_length)
          continue;
        unsigned int u = src[dataid];
        unsigned int v = dest[dataid];
        unsigned int w = weight[dataid];
        //printf("src %u | dest %u | weight %u | dataid %u\n", u, v, w, dataid);
        if (is_distance_infinity[u] == TRUE) {
          continue;
        }
        unsigned int temp_dist = distance[u] + w;
        if (distance[u] == -1) {
          continue;
        }
        if (temp_dist < distance[v]) {
          // relax
          //printf("%u %u\n", distance[v], temp_dist);
          int old_distance = atomicMin(&distance[v], temp_dist);
          atomicMin(&is_distance_infinity[v], FALSE);
          //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w, is_distance_infinity[v]);
          // test for a change!
          if (old_distance != distance[v]) {
            //printf("there is change\n");
            atomicMin(noChange, FALSE);
          }
        }
      }
    }
}

void neighborHandler(std::vector<initial_vertex> * peeps, int blockSize, int blockNum, int sync, int smem, unsigned int *distance_cur){
  /* Will use these arrays instead of a vector
  * edges_src : array of all edges (indexed 0 to n) where the value is the vertex source index of the edge (since edges are directed)
  * edges_dest : same as above, except it tells the vertex destination index
  * edges_weight : same as above, except it tells the edge's weight
  * distance_prev : array of all vertices with their distance values
  * distance_cur : same as above
  */

  /* Allocate here... */
  unsigned int *edges_src, *edges_dest, *edges_weight;
  unsigned int edges_length = 0;
  unsigned int vertices_length = peeps->size();
  unsigned int *distance_prev = (unsigned int *) malloc(vertices_length * sizeof(unsigned int));
  int *noChange = (int *) malloc(sizeof(int));
  int *is_distance_infinity = (int *) malloc(vertices_length * sizeof(int));
  int *should_update_edge;

  *noChange = TRUE;

  unsigned int *cuda_edges_src, *cuda_edges_dest, *cuda_edges_weight;
  unsigned int *cuda_distance_prev, *cuda_distance_cur;
  int *cuda_noChange, *cuda_is_distance_infinity_prev,
    *cuda_is_distance_infinity_cur, *cuda_should_update_edge;

  // the distance to the first vertex is always 0
  distance_prev[0] = 0;
  distance_cur[0] = 0;
  is_distance_infinity[0] = FALSE;

  // setting an unsigned int to -1 will set it to the maximum value!
  for (int i = 1; i < vertices_length; i++) {
    distance_prev[i] = -1;
    distance_cur[i] = -1;
    is_distance_infinity[i] = TRUE;
  }

  // get edges_length
  for(std::vector<int>::size_type i = 0; i != vertices_length; i++) {
    edges_length += peeps->at(i).nbrs.size();
  }

  // malloc edges arrays
  edges_src = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
  edges_dest = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
  edges_weight = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
  should_update_edge = (int *) malloc(edges_length * sizeof(int));


  int edge_index = 0;
  // get values for each array
  for(std::vector<int>::size_type i = 0; i != vertices_length; i++) {
    for(std::vector<int>::size_type j = 0; j != peeps->at(i).nbrs.size(); j++) {
      edges_src[edge_index] = peeps->at(i).nbrs[j].srcIndex;
      edges_dest[edge_index] = i;
      edges_weight[edge_index] = peeps->at(i).nbrs[j].edgeValue.weight;
      // initially set should_update_edges to true if the source is at 0, since everything
      // else will have infinite distance.
      if (edges_src[edge_index] == 0) {
        should_update_edge[edge_index] = TRUE;
      }
      else {
        should_update_edge[edge_index] = FALSE;
      }
      //printf("src: %u | dest: %u | weight: %u\n", edges_src[edge_index], edges_dest[edge_index], edges_weight[edge_index]);

      edge_index++;
    }
  }

  cudaMalloc((void **)&cuda_edges_src, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_edges_dest, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_edges_weight, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_distance_prev, vertices_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_distance_cur, vertices_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_noChange, sizeof(int));
  cudaMalloc((void **)&cuda_is_distance_infinity_prev, vertices_length * sizeof(int));
  cudaMalloc((void **)&cuda_is_distance_infinity_cur, vertices_length * sizeof(int));
  cudaMalloc((void **)&cuda_should_update_edge, edges_length * sizeof(int));

  cudaMemcpy(cuda_edges_src, edges_src, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_edges_dest, edges_dest, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_edges_weight, edges_weight, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_distance_prev, distance_prev, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_is_distance_infinity_prev, is_distance_infinity, vertices_length * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_is_distance_infinity_cur, is_distance_infinity, vertices_length * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_should_update_edge, should_update_edge, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

  setTime();

  /*
   * Do all the things here!
   **/
  // sync is out of core
  if (sync == 0) {
    for (unsigned int i = 1; i < vertices_length; i++) {
      //printf("pass %u\n", i);
      work_efficient_out_of_core<<<blockNum, blockSize>>>(edges_length, cuda_edges_src,
                                          cuda_edges_dest, cuda_edges_weight,
                                          cuda_distance_prev, cuda_distance_cur,
                                          cuda_noChange, cuda_is_distance_infinity_prev,
                                          cuda_is_distance_infinity_cur, cuda_should_update_edge);
      cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
      if (*noChange == TRUE) break;
      *noChange = TRUE;
      cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

      // get current distance and copy it to both cuda_distance_prev and cuda_distance_cur
      cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

      cudaMemcpy(is_distance_infinity, cuda_is_distance_infinity_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(cuda_is_distance_infinity_prev, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_is_distance_infinity_cur, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
  }
  // sync is in core
  else if (sync == 1) {
    work_efficient_in_core<<<blockNum, blockSize>>>(edges_length, vertices_length,
                                        cuda_edges_src, cuda_edges_dest,
                                        cuda_edges_weight, cuda_distance_cur,
                                        cuda_noChange, cuda_is_distance_infinity_prev,
                                        cuda_should_update_edge);
  }

  else {
    // no syncing
    printf("No syncing tag\n");
    exit(1);
  }

  cudaDeviceSynchronize();
  std::cout << "Took " << getTime() << "ms.\n";

  cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int),
           cudaMemcpyDeviceToHost);

  /* Deallocate. */
  cudaFree(cuda_edges_src);
  cudaFree(cuda_edges_dest);
  cudaFree(cuda_edges_weight);
  cudaFree(cuda_distance_prev);
  cudaFree(cuda_distance_cur);
  cudaFree(cuda_noChange);
  cudaFree(cuda_is_distance_infinity_prev);
  cudaFree(cuda_is_distance_infinity_cur);

  free(edges_src);
  free(edges_dest);
  free(edges_weight);
  free(distance_prev);
  free(noChange);
  free(is_distance_infinity);
}
