#include <stdio.h>
#include <stdlib.h>

__global__
void calculateX(int *rowArr, int *colArr, double *valueArr,double *xArr,int n, int m)
{
  int dist = n/blockDim.x;
  if(n%blockDim.x > threadIdx.x)
    dist = dist + 1;

  int s = ((n%blockDim.x>(threadIdx.x-1))? dist : n/blockDim.x)*threadIdx.x;

  int i;
  double sum = 0;
  for(i = s; i< s + dist; i++)
    for(int j = rowArr[i]; j < ((i+1 < n)? rowArr[i+1] : m); j++)
      sum = sum + valueArr[j] * xArr[colArr[j]];
    xArr[i] = sum;
    sum = 0;
  }   

int main(int argc, char *argv[])
{
  int threads = atoi(argv[1]);
  int iterations = atoi(argv[2]);
  int answer = atoi(argv[3]);


  FILE *file = fopen(argv[4], "r");
  int num;
  double num2;
  int row;
  int col;

  fscanf(file, "%d", &num);
  int sizeOfMatrix = num;

  fscanf (file, "%d", &num);

  fscanf(file, "%d", &num);
  int numOfNumbers =  num;

  static double matrix[15000][15000];

  for(row = 0; row < sizeOfMatrix; row++)
    for(col = 0; col < sizeOfMatrix; col++)
      matrix[row][col] = 0;

  while(!feof (file))
  {
    fscanf(file, "%d", &num);
    row = num-1;
    fscanf(file, "%d", &num);
    col = num-1;
    fscanf(file, "%lf", &num2);

    matrix[row][col] = num2;
  }
  double *x = (double *)malloc(sizeOfMatrix*sizeof(double));
  
  for(row = 0; row < sizeOfMatrix; row++)
    x[row] = 1;

   int *row_ptr = (int *)malloc(sizeOfMatrix*sizeof(int));
   int *col_ind = (int *)malloc(numOfNumbers*sizeof(int));
   double *values = (double *)malloc(numOfNumbers*sizeof(double));

   int count = 0;
   int first = 0;

  for(row = 0; row < sizeOfMatrix; row++)
  {
    for(col = 0; col < sizeOfMatrix; col++)
    {
      if(matrix[row][col] != 0 && first == 0)
      {
        row_ptr[row] = count;
        col_ind[count] = col;
        values [count] = matrix[row][col];
        count ++;
        first = 1;
      }

      else if(matrix[row][col] != 0 && first == 1)
      {
        col_ind [count] = col;
        values [count] = matrix[row][col];
        count++;
      }

    }
    if(first == 0)
      row_ptr[row] = -1;
    
    first = 0;
  }

  int i = 1;
  for(row = 0; row < sizeOfMatrix; row++)
    if(row_ptr[row] == -1)
    {
      while(1)
      {
        if(row_ptr[row + i] != -1)
        {
          row_ptr[row] = row_ptr[row+i];
          break;
        }
        i++;
      }
      i = 1;
    }

  int *rowArr, *colArr;
  double *valueArr, *xArr;
  cudaMalloc(&rowArr, sizeOfMatrix*sizeof(int));
  cudaMalloc(&colArr, numOfNumbers*sizeof(int)); 
  cudaMalloc(&valueArr, numOfNumbers*sizeof(double)); 
  cudaMalloc(&xArr, sizeOfMatrix*sizeof(double));

  cudaMemcpy(rowArr, row_ptr, sizeOfMatrix*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(colArr, col_ind, numOfNumbers*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(valueArr, values, numOfNumbers*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(xArr, x, sizeOfMatrix*sizeof(double), cudaMemcpyHostToDevice);

  for(row = 0; row < iterations; row++)
  {
    calculateX<<<1, threads>>>(rowArr, colArr, valueArr, xArr, sizeOfMatrix, numOfNumbers);
    cudaThreadSynchronize();
  }

  cudaMemcpy(x, xArr, sizeOfMatrix*sizeof(double), cudaMemcpyDeviceToHost);


  if(answer == 1)
  {
    printf("ROW");
    for(row = 0; row < sizeOfMatrix; row++)
      printf("%d ",row_ptr[row]);

    printf("\n");
    printf("COL");
    printf("\n");
    for(row = 0; row < numOfNumbers; row++)
      printf("%d ",col_ind[row]);

    printf("\n");
    printf("VALUES");
    printf("\n");
    for(row = 0; row < numOfNumbers; row++)
      printf("%lf ",values[row]);

    printf("\n");
    printf("X ARRAY");
    printf("\n");
    for(row = 0; row < sizeOfMatrix; row++)
      printf("%lf ",x[row]);
  }

  cudaFree(rowArr);
  cudaFree(colArr);
  cudaFree(valueArr);
  cudaFree(xArr);

  free(row_ptr);
  free(col_ind);
  free(values);
  free(x);
}