#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void pulling_kernel(std::vector<initial_vertex> * peeps, int offset, int * anyChange){

    //update me based on my neighbors. Toggle anyChange as needed.
    //offset will tell you who I am.
}

void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum){
    /* Allocate here... */
    unsigned int *cuda_edges_src, *cuda_edges_dest, *cuda_edges_weight;
    unsigned int edges_length;
    unsigned int vertices_length = peeps.size();
    unsigned int *cuda_distance_prev = (unsigned int *) malloc(peeps.size() * sizeof(unsigned int));
    unsigned int *cuda_distance_cur = (unsigned int *) malloc(peeps.size() * sizeof(unsigned int));

    int i = 0;
    for(auto const& vertex: peeps) {
      for(auto const& edge: vertex.nbrs) {
        std::cout << "vertexIndex: " << i << " | vertexDistance: " << vertex.get_vertex_ref().distance << " | src: " << edge.srcIndex << " | dest: " << " | weight: " << edge.edgeValue.weight << endl;
      }
      i++;
    }

    setTime();

    /*
     * Do all the things here!
     **/



    std::cout << "Took " << getTime() << "ms.\n";
}
