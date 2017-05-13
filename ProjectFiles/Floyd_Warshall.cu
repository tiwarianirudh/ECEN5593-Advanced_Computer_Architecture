/*Includes*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

/*macros*/
#define VERTICES (1000)           //number of vertices for graph
#define MIN_EDGES_VERTEX (25)     //minimum no. of edges for each vertex
#define INF_DIST (10000000)       //Initial "infinite" distance value for each node
#define MAX_DIST (1000)           // maximum possible distance between nodes
int BLOCKSIZE=512;                //no. of threads per block

/*Fucntion Prototypes*/

/*This functions assigns random distance between nodes with minimum of 25 edges per vertex*/
void Set_Graph_Dist_Random(int* Graph, int* Edges_Per_Vertex);

/*This function initialized graph to a int value*/
void Initialize_Graph(int* Graph,int Value);

/*This function initializes array to a int value*/
void Initialize_Array(int* Input_Array,int Value);

/*This function calculates shortest distance between nodes serially*/
void Serial_Floyd(int* Host_Graph,int *Host_Path);

/*This function calculates shortest distance between nodes parallely*/
__global__ void CUDA_Kernel(int i,int* cuda_Graph, int* cuda_Path);

/*This function calculates the timetaken*/ 
double timetaken();

/*This function is used to pass command line arguments*/
void arguments(int, char**);

/*varibales to calculate time taken*/
struct timeval initial, final;

int main(int argc, char** argv){
   
   arguments(argc, argv);
   
   printf("\nRunning Floyd Warshall's Algorithm");
   srand(8421);

   /*Host memory allocation*/
   int Graph_Size=VERTICES*VERTICES*sizeof(int);
   int *Graph=(int *)malloc(Graph_Size);
   int *Host_Graph=(int *)malloc(Graph_Size);
   int *Host_Path=(int *)malloc(Graph_Size);
   int *Device_Graph=(int *)malloc(Graph_Size);
   int *Device_Path=(int *)malloc(Graph_Size);
   int* Edges_Per_Vertex = (int*)malloc(VERTICES*sizeof(int));
   printf("\nBlocksize :%d",BLOCKSIZE);
   printf("\nVertices  :%d",VERTICES);

   /*Initialize array and graphs*/
   Initialize_Graph(Graph,(int)0);
   Initialize_Array(Edges_Per_Vertex,(int)0);

   /*Assign random distance between nodes*/
   Set_Graph_Dist_Random(Graph,Edges_Per_Vertex);

   /*Free memory for edges_per_Vertex as it will not be used after this*/ 
   free(Edges_Per_Vertex);

   /*Copy value from Graph to Host_Graph for computation*/
   int i;
   for(i=0;i<VERTICES*VERTICES;i++){
       Host_Graph[i]=Graph[i];
       Host_Path[i]=-1;
   }
   printf("\nPerforming CPU computation");
   
  /*get the start time for cpu computation*/
   gettimeofday(&initial,NULL);

   /*This function finds shortest path between all nodes*/
   Serial_Floyd(Host_Graph,Host_Path);

   /*Get the stop time for cpu computation*/
   gettimeofday(&final,NULL);
   double diff=0;

   /*calculate the time taken for cpu computation*/
   diff=timetaken();

   printf("\nTime taken for logic computation by CPU in seconds is %f",diff);

   /*Cpoy value from graph to device_graph for computation*/
   for(i=0;i<VERTICES*VERTICES;i++){
       Device_Graph[i]=Graph[i];
       Device_Path[i]=-1;
   }
   
   /*Variables for device memory allocation*/
   int* cuda_Graph;
   int* cuda_Path;
  
   /*Device memory allocation*/
   cudaMalloc((void**)&cuda_Graph,Graph_Size);
   cudaMalloc((void**)&cuda_Path,Graph_Size);
   
   /*Host to memory transfer*/

   /*Get the start time for transfer from host to memory*/ 
   gettimeofday(&initial,NULL);

   cudaMemcpy(cuda_Graph, Device_Graph, Graph_Size, cudaMemcpyHostToDevice);
   cudaMemcpy(cuda_Path, Device_Path, Graph_Size,cudaMemcpyHostToDevice);
    
   /*Get the stop time for host to memory transfer*/
   gettimeofday(&final,NULL);

   double diff2=0;

   /*Calculate the time taken for host to memory transfer*/
   diff2=timetaken();

   printf("\nTime taken for memory transfer from host to device in seconds is %f",diff2);

   dim3 dimGrid((VERTICES+BLOCKSIZE-1)/BLOCKSIZE,VERTICES);   
   
   printf("\nPerforming GPU computation");
   
   /*get the start time for GPU computation*/
   gettimeofday(&initial,NULL);
   
    for(i=0;i<VERTICES;i++){
        
       /*This function finds the shortest path between all nodes parallely*/
       CUDA_Kernel<<<dimGrid,BLOCKSIZE>>>(i,cuda_Graph,cuda_Path);
       cudaThreadSynchronize();
    }

    /*Get the stop time for GPU computation*/
    gettimeofday(&final,NULL);
    
    double diff1=0;
    
    /*Calculate the time taken for gpu computation*/
    diff1=timetaken();

    printf("\nTime taken for GPU kernel execution in seconds is %f\n",diff1);

    /*Device to host transfer*/

    /*get the start time for device to host transfer*/
    gettimeofday(&initial,NULL);
    
    cudaMemcpy(Device_Graph,cuda_Graph, Graph_Size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Device_Path,cuda_Path, Graph_Size, cudaMemcpyDeviceToHost);
    
    /*Get the stop time for device to host transfer*/
    gettimeofday(&final,NULL);
    
    double diff3=0;
    
    /*Calculate the time taken for device to host transfer*/
    diff3=timetaken();
    
    printf("\nTime taken for memory transfer from device to host  in seconds is %f\n",diff3);
    printf("\nTime taken for total memory transfer in seconds is %f\n",(diff2+diff3));

   /*Compare CPU and GPU result*/
   int match=0;
   for(i=0;i<VERTICES*VERTICES;i++){
       if(Host_Graph[i]==Device_Graph[i]){
           match++;

       }

   }
   if(match==(VERTICES*VERTICES)){
       printf("\nThe CPU and GPU results match\n");
   }

   /*free host memory*/
   free(Graph);
   free(Host_Graph);
   free(Device_Graph);
   free(Host_Path);
   free(Device_Path);

   /*free device memory*/
   cudaFree(cuda_Graph);
   cudaFree(cuda_Path);

}

/*This function initializes graph to a value*/
void Initialize_Graph(int* Graph,int Value){
    uint32_t i,j;
    for(i=0;i<VERTICES;i++){
        for(j=0;j<VERTICES;j++){
            Graph[i*VERTICES + j] = Value;
        }
    }
}

/*This function initializes array to a value*/
void Initialize_Array(int* Input_Array,int Value){
    uint32_t i;
    for(i=0;i<VERTICES;i++){
        Input_Array[i]=Value;
    }
}

/*This function assigns random distance between nodes with a minimum of 25 edges per vertex*/
void Set_Graph_Dist_Random(int* Graph, int* Edges_Per_Vertex){
    uint32_t i,Current_Edges,Random_Vertex;
    int Random_Dist;

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

/*This function computes shortest distance between all nodes serially*/
void Serial_Floyd(int *Graph1,int *Graph_Path){
    int x,y,z;
    for(x=0;x<VERTICES;++x){
        for(y=0;y<VERTICES;++y){
            for(z=0;z<VERTICES;++z){
		        int current_node=y*VERTICES+z;
		        int Node_i=y*VERTICES+x;
		        int Node_j=x*VERTICES+z;
                if(Graph1[current_node]>(Graph1[Node_i]+Graph1[Node_j])){
                    Graph1[current_node]=(Graph1[Node_i]+Graph1[Node_j]);
                    Graph_Path[current_node]=x;
                }
            }
        }
    }
}

/*This function computes shortest distance between all nodes parallely*/
__global__ void CUDA_Kernel(int i,int* cuda_Graph, int* cuda_Path){

     int tid=threadIdx.x;
     int gid=blockIdx.x*blockDim.x +threadIdx.x;
     if(gid>=VERTICES){
         return;
     }
 
int idx=VERTICES*blockIdx.y + gid;
__shared__  int shortest_distance;

if(tid==0){
    shortest_distance=cuda_Graph[VERTICES*blockIdx.y+i];
}

__syncthreads();

    int node_distance=cuda_Graph[i*VERTICES+gid];

    int total_distance=shortest_distance+node_distance;

    if (cuda_Graph[idx]>total_distance){

   cuda_Graph[idx]=total_distance;
   cuda_Path[idx]=i;
   }

}

/*This function calculates time taken in seconds*/
double timetaken(){
    double initial_s,final_s;
    double diff_s;
    initial_s= (double)initial.tv_sec*1000000 + (double)initial.tv_usec;
    final_s= (double)final.tv_sec*1000000 + (double)final.tv_usec;
    diff_s=(final_s-initial_s)/1000000;
    return diff_s;
}

/*This function is used to pass command line arguments*/
void arguments(int argc, char** argv){
    for (int h = 0; h < argc; ++h) {
        if (strcmp(argv[h], "--blocksize") == 0 || strcmp(argv[h], "-blocksize") == 0) {
            BLOCKSIZE = atoi(argv[h+1]);
            h = h + 1;
        }
      
    }
}
