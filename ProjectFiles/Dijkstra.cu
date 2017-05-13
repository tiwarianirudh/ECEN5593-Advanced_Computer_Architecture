/*Includes*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

/*Macros*/
#define VERTICES (1024)          //number of vertices for graph
#define MIN_EDGES_VERTEX (25)    //minimum no. of edges for each vertex
#define MAX_DIST (1000)          //maximum possible distance
#define INF_DIST (10000000)      //Initial "infinite" distance value for each node
#define BLOCKSIZE (256)           //Threads per block 

/*Variable to calculate time taken*/
struct timeval initial, final;

/*Function Prototypes*/

/*This function initializes array to a int value*/
void Initialize_Array(int* Input_Array,int Value);

/*This function initializes array to a float value*/
void Initialize_Dist_Array(float* Input_Array,float Value);

/*This function initializes graph*/
void Initialize_Graph(float* Graph,float Value);

/*This function assigns random distance between nodes*/
void Set_Graph_Dist_Random(float* Graph, int* Edges_Per_Vertex);

/*This function finds the next closest node serially*/
int Shortest_Distance_Node(float* Node_Shortest_Dist, int* Completed_Node);

/*This function finds the shortest path from source node to all nodes serially*/
void Shortest_Path_Computation_Serial(float* Graph, float* Node_Shortest_Dist, int* Parent_Node, int* Completed_Node, int Source,int* Edges_Per_Vertex);

/*This function calculates the time difference*/
double timetaken();

/*This function calculates the next closest node parallely*/
//__global__ void Shortest_Distance_Node_CUDA(float* Node_Shortest_Dist, int* Completed_Node,int* closest_node);
//This function was working working for 1 block of 1 thread which had a lot of overhead, 
//couldn't make it work with multiple threads. For multiple threads, it gives an incorrect result.

/*This fuction calculates the shortest path from source node to all nodes parallely*/
__global__ void Shortest_Path_Computation_CUDA(float* Graph, float* Node_Shortest_Dist, int* Parent_Node, int* Completed_Node, int* closest_node); 

int main(){
    
    printf("Running Dijkstra Algorithm\n");
    srand(8421);

    /*Variables to initialize array and graphs*/
    int Integer_Array = VERTICES * sizeof(int);
    int Float_Array = VERTICES * sizeof(float);
    int64_t Size_Graph = VERTICES * VERTICES * sizeof(float);

    /*Host Memory Allocation*/
    float* Graph = (float*)malloc(Size_Graph);
    float* Node_Shortest_Dist_1 = (float*)malloc(Float_Array);
    float* Node_Shortest_Dist_2 = (float*)malloc(Float_Array);
    int* Parent_Node = (int*)malloc(Integer_Array);
    int* Edges_Per_Vertex = (int*)malloc(Integer_Array);
    int* Completed_Node = (int*)malloc(Integer_Array);
    int* closest_node= (int*)malloc(sizeof(int));
      
    /*Variables for device memory allocation*/
    float* cuda_Graph;
    float* cuda_Node_Shortest_Dist;
    int* cuda_Parent_Node;
    int* cuda_Completed_Node;
    int* cuda_closest_node;
    

    /*Device Memory Allocation*/
    cudaMalloc((void**)&cuda_Graph,Size_Graph);
    cudaMalloc((void**)&cuda_Node_Shortest_Dist,Float_Array);
    cudaMalloc((void**)&cuda_Parent_Node,Integer_Array);
    cudaMalloc((void**)&cuda_Completed_Node,Integer_Array);
    cudaMalloc((void**)&cuda_closest_node,sizeof(int));   
   
    printf("\nVertices: %d", VERTICES);
    printf("\nThreads Per Block: %d",BLOCKSIZE);

    /*Take a random source value*/ 
    int src=(rand()%VERTICES);

    /*Get the start time for cpu computation*/
    gettimeofday(&initial,NULL);

    printf("\nPerforming CPU compuatation");

    /*This fuction calculate the shortest path from source node to all node serially*/
    Shortest_Path_Computation_Serial(Graph,Node_Shortest_Dist_1,Parent_Node,Completed_Node,src,Edges_Per_Vertex);
  
    /*Get the stop time for cpu computation*/
    gettimeofday(&final,NULL);
    double diff=0;

    /*Calculate the time taken*/
    diff=timetaken();

    printf("\nTime taken for logic computation by CPU in seconds is %0.6f",diff);

    /*Clear the previous values completed node and parent node*/

    /*This function initializes parent node to a initial value of -1*/
    Initialize_Array(Parent_Node,(int)-1);

    /*This function initializes completed node to a initial value of 0*/
    Initialize_Array(Completed_Node,(int)0);

    /*This function initializes Node_Shortest_Dist_2 to a very high initial value*/
    Initialize_Dist_Array(Node_Shortest_Dist_2,INF_DIST);

    Node_Shortest_Dist_2[src]=0;
    closest_node[0]=-1;
    
    /*Host to device transfer*/

    /*Get the start time for host to device transfer*/
    gettimeofday(&initial,NULL);

    cudaMemcpy(cuda_Graph, Graph, Size_Graph, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_Node_Shortest_Dist, Node_Shortest_Dist_2, Float_Array, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_Parent_Node, Parent_Node, Integer_Array, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_Completed_Node, Completed_Node, Integer_Array, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_closest_node,closest_node, sizeof(int), cudaMemcpyHostToDevice);

    /*Get the stop time for host to device transfer*/
    gettimeofday(&final,NULL);
    double diff2=0;

    /*Calculate the time taken for host to device transfer*/
    diff2=timetaken();

    printf("\nTime taken for host to device transfer in seconds is %0.6f",diff2);

    /*Get the start time for GPU computation*/
    gettimeofday(&initial,NULL);

   
    for(int i=0;i<VERTICES;i++){
    
     /*This kernel launching 1 block of 1 thread at a time, although produced correct result but had a lot of overhead,
      *if we increased the threads and blocks and modified the kernel function aacordingly , it produces an incorrect
      * result,So in order to get correct result with less overhead, we find the next closest node serially, and find 
      * the shortest path from source node to all nodes parallely. This still has an overhead of transfering from 
      * host to device memory and vice versa repeatedly in a for loop. But this overhead is still less than launching
      * kernel of 1 block of 1 thread repeatedly.*/
    
     //Shortest_Distance_Node_CUDA <<<1,1>>>(cuda_Node_Shortest_Dist,cuda_Completed_Node,cuda_closest_node,node_distance,node); 
      
     /*This function calculate the closest node serially*/
     closest_node[0]=Shortest_Distance_Node(Node_Shortest_Dist_2, Completed_Node);
     cudaMemcpy(cuda_closest_node,closest_node, sizeof(int), cudaMemcpyHostToDevice);

     /*This function calculates the shortest distance from source node to all other nodes parallely*/
     Shortest_Path_Computation_CUDA <<<(VERTICES+BLOCKSIZE-1/BLOCKSIZE),BLOCKSIZE>>>(cuda_Graph,cuda_Node_Shortest_Dist,cuda_Parent_Node,cuda_Completed_Node,cuda_closest_node);
     cudaMemcpy(Node_Shortest_Dist_2,cuda_Node_Shortest_Dist, Float_Array, cudaMemcpyDeviceToHost);

    }
   

    /*Get the stop time for GPU computation*/
    gettimeofday(&final,NULL);
    double diff1=0;

    /*Caluclate the time taken for GPU computation*/
    diff1=timetaken();

    
    printf("\nTime taken for logic computation by GPU in seconds is %0.6f",diff1);

    /*Device to host trasfer*/

    /*Get the start time for device to host memory transfer*/
    gettimeofday(&initial,NULL);
    
    cudaMemcpy(Node_Shortest_Dist_2,cuda_Node_Shortest_Dist, Float_Array, cudaMemcpyDeviceToHost);
    cudaMemcpy(Parent_Node, cuda_Parent_Node, Integer_Array, cudaMemcpyDeviceToHost);
    cudaMemcpy(Completed_Node, cuda_Completed_Node, Integer_Array, cudaMemcpyDeviceToHost);
    
    /*Get the stop time for device to host transfer*/
    gettimeofday(&final,NULL);
    
    /*Calculate the time taken for device to host transfer*/
    double diff3=timetaken();

    printf("\nTime taken for memory transfer from device to host in  seconds is %0.6f",diff3);
    printf("\nTotal time taken for memory transfer is %0.6f",(diff2+diff3));
    
    /*Compare CPU and GPU result*/
    int k=0;
    int match=0;
	
    for(k=0;k<VERTICES;k++){
        if(Node_Shortest_Dist_1[k]==Node_Shortest_Dist_2[k]){
            match++;		
	}
    }
 
    if(match==VERTICES){
        printf("\nThe cpu and gpu results match\n");
    }

    /*Free host memory*/
    free(Graph);
    free(Node_Shortest_Dist_1);
    free(Node_Shortest_Dist_2);
    free(Parent_Node);
    free(Completed_Node);
    free(closest_node);
    
    /*Free device memory*/
    cudaFree(cuda_Graph);
    cudaFree(cuda_Node_Shortest_Dist);
    cudaFree(cuda_Parent_Node);
    cudaFree(cuda_Completed_Node);
    cudaFree(cuda_closest_node);
}

/*This function initializes graph*/
void Initialize_Graph(float* Graph,float Value){
    int i,j;
    for(i=0;i<VERTICES;i++){
        for(j=0;j<VERTICES;j++){
            Graph[i*VERTICES + j] = Value;
        }
    }
}

/*This function initializes array to a int value*/
void Initialize_Array(int* Input_Array,int Value){
    int i;
    for(i=0;i<VERTICES;i++){
        Input_Array[i]=Value;
    }
}

/*This function initializes array to a float value*/
void Initialize_Dist_Array(float* Input_Array,float Value){
    int i;
    for(i=0;i<VERTICES;i++){
        Input_Array[i]=Value;
    }
}

/*Ths function assigns random distance between nodes with a minimum of 25 edges per vertex*/
void Set_Graph_Dist_Random(float* Graph, int* Edges_Per_Vertex){
    int i,Current_Edges,Random_Vertex;
    float Random_Dist;

    for(i=1;i<VERTICES;i++){
        Random_Vertex = (rand() % i);
        Random_Dist =(rand() % MAX_DIST) + 1;
        Graph[Random_Vertex*VERTICES + i] = Random_Dist;
        Graph[Random_Vertex + i*VERTICES] = Random_Dist;
        Edges_Per_Vertex[i] += 1;
        Edges_Per_Vertex[Random_Vertex] += 1;
    }

    for(i=0;i<VERTICES;i++){
        Current_Edges = Edges_Per_Vertex[i];
        while(Current_Edges < MIN_EDGES_VERTEX){
            Random_Vertex = (rand() % VERTICES);
            Random_Dist = (rand() % MAX_DIST) + 1;
            if((Random_Vertex != i)&&(Graph[Random_Vertex + i*VERTICES] == 0)){
                Graph[Random_Vertex + i*VERTICES] = Random_Dist;
                Graph[Random_Vertex*VERTICES + i] = Random_Dist;
                Edges_Per_Vertex[i] += 1;
                Current_Edges += 1;
            }
        }
    }
}

/*This function calculates the shortest path serially*/
void Shortest_Path_Computation_Serial(float* Graph, float* Node_Shortest_Dist, int* Parent_Node, int* Completed_Node, int Source,int* Edges_Per_Vertex){
    
    /*Initialize array and graph*/
    Initialize_Graph(Graph,(float)0);
    Initialize_Array(Edges_Per_Vertex,(int)0);
    Set_Graph_Dist_Random(Graph,Edges_Per_Vertex);
    free(Edges_Per_Vertex);
    Initialize_Array(Parent_Node,(int)-1);
    Initialize_Array(Completed_Node,(int)0);
    Initialize_Dist_Array(Node_Shortest_Dist,INF_DIST);

    Node_Shortest_Dist[Source]=0;
    int i,j;
    for(i=0;i<VERTICES;i++){

        /*This function finds the next closest node and returns it*/
        int current_node=Shortest_Distance_Node(Node_Shortest_Dist,Completed_Node);
        Completed_Node[current_node]=1;
        for(j=0;j<VERTICES;j++){
            int new_distance=Node_Shortest_Dist[current_node] + Graph[current_node*VERTICES + j];
            if ((Completed_Node[j] != 1) && (Graph[current_node*VERTICES + j] != (float)(0)) && (new_distance < Node_Shortest_Dist[j])){
                Node_Shortest_Dist[j] = new_distance;
                Parent_Node[j] = current_node;
            }
        }
    }
}

/*This function calculates the shortest path from source node to all other nodes in parallel*/
__global__ void Shortest_Path_Computation_CUDA(float* Graph, float* Node_Shortest_Dist, int* Parent_Node, int* Completed_Node, int* closest_node){
   
    Completed_Node[closest_node[0]]=1;
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
     
    if(tid>VERTICES)
        return;
    
    int current_node=closest_node[0];
    int new_distance;

    new_distance = Node_Shortest_Dist[current_node] + Graph[current_node*VERTICES + tid];

    if ((Completed_Node[tid] != 1) && (Graph[current_node*VERTICES + tid] != (float)(0)) && (new_distance < Node_Shortest_Dist[tid])){ //each thread get different j & new_distance
        Node_Shortest_Dist[tid] = new_distance;
        Parent_Node[tid] = current_node;
    }
          
}

/*This function calculates the shortest distance node serially*/
int Shortest_Distance_Node(float* Node_Shortest_Dist, int* Completed_Node){
    int node_distance=INF_DIST;
    int node=-1;
    int i;
    for(i=0;i<VERTICES;i++){
        if((Node_Shortest_Dist[i]<node_distance) && (Completed_Node[i]==0)){
            node_distance=Node_Shortest_Dist[i];
            node=i;
        }
    }
    Completed_Node[node]=1;
    return node;
}

/* We tried different ways to find next closest node in parallel using multiple threads but failed to do that
*There is a race condition between threads to modify same variable node_distance and node, so we tried to find 
*smallest distance for each block and then smallest distance among all block, but it gives an incorrect result
*The below function works correctly for 1 block of 1 thread*/

/*__global__ void Shortest_Distance_Node_CUDA(float* Node_Shortest_Dist,int* Completed_Node,int* closest_node){
    int node_distance = INF_DIST;
    int node = -1;
    int tid= threadIdx.x;
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
 
    int i;
    for (i = 0; i < VERTICES; i++) {
        if((Node_Shortest_Dist[gid] < node_distance[tid]) && (Completed_Node[gid] == 0)){
            node_distance= Node_Shortest_Dist[gid];
            node = gid;
        }
    }
      
    Completed_Node[node]=1;
    closest_node[0]=node;   
 
}
*/

/*This function calculates the time difference in seconds*/
double timetaken(){
    double initial_s,final_s;
    double diff_s;
    initial_s= (double)initial.tv_sec*1000000 + (double)initial.tv_usec;
    final_s= (double)final.tv_sec*1000000 + (double)final.tv_usec;
    diff_s=(final_s-initial_s)/1000000;
    return diff_s;
}

