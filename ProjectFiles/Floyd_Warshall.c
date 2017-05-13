/*Includes*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

/*Macros*/
#define VERTICES (1000)          //number of vertices for graph
#define MIN_EDGES_VERTEX (25)    //minimum no. of edges for each vertex
#define INF_DIST (10000000)      //Initial "infinite" distance value for each node
#define MAX_DIST (1000)          //maximum possible distance betwwen nodes


/*Function Prototypes*/

/*This function assign random distance between nodes*/
void Set_Graph_Dist_Random(int* Graph, int* Edges_Per_Vertex);

/*This function is used to initialize graph to 0 value*/
void Initialize_Graph(int* Graph,int Value);

/*This function is used initialize array to a value*/
void Initialize_Array(int* Input_Array,int Value);

/*This function calculates shortest distance between nodes*/
void Serial_Floyd(int* Host_Graph,int *Host_Path);

/*This function uses openMP and finds shortest distance between nodes in parallel*/
void Parallel_Floyd_openMP(int* Host_Graph,int *Host_Path);

/*This function is used to calculate the time difference*/
double timetaken();

/*Initial and final are used to calculate time difference*/
struct timeval initial, final;

int main(){
 
    printf("\nRunning Floyd Warshall's Algorithm\n");
    printf("VERTICES: %d\n",VERTICES);
    srand(8321);
    int Graph_Size=VERTICES*VERTICES*sizeof(int);
  
    /*Host memory allocation*/
    int *Graph=(int *)malloc(Graph_Size);
    int *Host_Graph=(int *)malloc(Graph_Size);
    int *Host_Path=(int *)malloc(Graph_Size);
    int *Host_Graph1=(int *)malloc(Graph_Size);
    int *Host_Path1=(int *)malloc(Graph_Size);
    int* Edges_Per_Vertex = (int*)malloc(VERTICES*sizeof(int));

    /*This function initializes graph to value 0*/
    Initialize_Graph(Graph,(int)0);

    /*This function initializes Edges_Per_Vertex array to 0*/
    Initialize_Array(Edges_Per_Vertex,(int)0);
    
    /*This function assign random distance between nodes*/
    Set_Graph_Dist_Random(Graph,Edges_Per_Vertex);
    
    /*Free the memory as Edges_Per_Vertex will not be used*/
    free(Edges_Per_Vertex);

    /*Graph is copied to Host_Graph for computation*/
    int i;
    for(i=0;i<VERTICES*VERTICES;i++){
       Host_Graph[i]=Graph[i];
       Host_Path[i]=-1;
    }
   
    /*Get the start time for serial computation*/
    gettimeofday(&initial,NULL);

    /*This function finds shortest distance between nodes serially*/
    Serial_Floyd(Host_Graph,Host_Path);

    /*Get the stop time for serial compuation*/
    gettimeofday(&final,NULL);

    double diff=0;

    /*Calculate time for serial computation*/
    diff=timetaken();

    printf("Time taken for logic computation by CPU in seconds is %f\n",diff);

    /*Set no. of threads for openMP*/
    omp_set_num_threads(2);

    /*Copy the value of graph to Host_Graph1 for Computation*/
    for(i=0;i<VERTICES*VERTICES;i++){
       Host_Graph1[i]=Graph[i];
       Host_Path1[i]=-1;
    }
    
    /*Get the start time for openMP computation*/
    gettimeofday(&initial,NULL);

    /*This function is used to find shortest path parallely*/
    Parallel_Floyd_openMP(Host_Graph1,Host_Path1);

    /*Get the stop time openMP computation*/
    gettimeofday(&final,NULL);

    double diff1=0;

    /*Calculate time difference for openMP implementation*/
    diff1=timetaken();

    printf("Time taken for logic computation by CPU in parallel in seconds is %f\n",diff1);

    /*Check if serial and openmp implementation match*/
    int match=0;
    for(i=0;i<VERTICES*VERTICES;i++){
        if(Host_Graph[i]==Host_Graph1[i]){
           match++;

        } 

    }
    if(match==VERTICES*VERTICES){
        printf("The serial and openMP output match\n");
    }
    
    /*Free host memory*/
    free(Graph);
    free(Host_Graph);
    free(Host_Graph1);
    free(Host_Path);
    free(Host_Path1);

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

/*This function assigns random distance between graph node with a minimum of 25 nodes per vertex*/
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

/*This function computes shortest distance between nodes serially*/
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

/*This function computes shortest distance between nodes parallely*/
void Parallel_Floyd_openMP(int *Graph1,int *Graph_Path){
    int x,y,z;
    int current_node,Node_i,Node_j;
    #pragma omp parallel shared(Graph1,Graph_Path)
    {
    #pragma omp for private(x,y,z,current_node,Node_i,Node_j)
    for(x=0;x<VERTICES;++x){
        for(y=0;y<VERTICES;++y){

                for(z=0;z<VERTICES;++z){
                    
      		    current_node=y*VERTICES+z;
		    Node_i=y*VERTICES+x;
		    Node_j=x*VERTICES+z;
                    if(Graph1[current_node]>(Graph1[Node_i]+Graph1[Node_j])){
                        Graph1[current_node]=(Graph1[Node_i]+Graph1[Node_j]);
                        Graph_Path[current_node]=x;
                    }
                 }
           }
       
    }
    #pragma omp barrier
    }
}

/*This function calculates the time difference between initial and final in seconds*/
double timetaken(){
    double initial_s,final_s;
    double diff_s;
    initial_s= (double)initial.tv_sec*1000000 + (double)initial.tv_usec;
    final_s= (double)final.tv_sec*1000000 + (double)final.tv_usec;
    diff_s=(final_s-initial_s)/1000000;
    return diff_s;
}

