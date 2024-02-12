#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


// The cuda kernel
__global__ void 
conv2d_kernel(const float *d_input, const float *d_filter, float *d_output, int f_dimen, int h, int w, int padded_width, int numElements, int pad_total)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < pad_total && ((i % padded_width) < w))
  {
    int a = 0;
    float sum = (float)a;
    int g = 0;
    int r = 0;
      
    for(int j = 0; j < f_dimen; j++){
      for(int k = 0; k < f_dimen; k++){
        g = i + k + padded_width*j;
        r = k + f_dimen*j;
        sum += d_filter[r] * d_input[g];
      }
    }
       
    int L = w*(i / padded_width) + i % padded_width;

    if(d_output[L] == 0)    
    d_output[L] = sum;
  }
}


int main(int argc, char *argv[]) {

  // Read the inputs from command line

  // Allocate/move data using cudaMalloc and cudaMemCpy

  // Launch the kernel

  // Print the output

  // Clean up the memory

  
  cudaError_t err = cudaSuccess;
  
	char *trace_file1;
  char *trace_file2;  
  trace_file1 = argv[1];
  trace_file2 = argv[2];
  std::ifstream file_in(trace_file1);
  std::ifstream file_filter(trace_file2);

	
	//std::ifstream        file_in, file_filter;          
	//std::vector<float> inp;
	int *hw= (int *)malloc(2*sizeof(int));
	int numElements=0;
	
	//file_in.open("input.txt", std::ifstream::in);
	std::vector<float> fliter_v;
  float f_v;
  int f_dimen; 
  file_filter >> f_dimen;
	while (file_filter >> f_v)
	{
		fliter_v.push_back(f_v);
	}
	for(int i=0; i<2; i++)
	{
		file_in >> hw[i];
	}

  int pad_amount = f_dimen - 1;
  int h_p_am = pad_amount / 2;

  int h = hw[0];
  int w = hw[1];

  //int in_total = h * w;

	std::vector<float> input_v;
  float v;
	while (file_in >>v)
	{
		input_v.push_back(v);
		numElements++;
	}
	
	//file_in.close();

  int padded_width = w + pad_amount; 
  int padded_height = h + pad_amount;
  float* padded_image = (float*)malloc(padded_width * padded_height * sizeof(float));

  for (int i = 0; i < padded_height; i++) {
    for (int j = 0; j < padded_width; j++) {
        padded_image[i * padded_width + j] = 0;
    }
  }

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
        padded_image[(y + h_p_am) * padded_width + (x + h_p_am)] = input_v[y * w + x];
    }
  }
  
  //file_filter.open("filter0.txt", std::ifstream::in);

  
  

	//file_filter.close();  

  int pad_total = padded_width * padded_height;

	size_t size = numElements * sizeof(float);
  size_t pad_size = pad_total * sizeof(float);
	float *A_in= (float *)malloc(pad_size);
	float *A_out= (float *)malloc(size);
  int filter_total = f_dimen * f_dimen;
  size_t filter_size = filter_total * sizeof(float);
  float *A_fil= (float *)malloc(filter_size);

  for(int i=0; i<pad_total; i++) A_in[i] = padded_image[i];
  for(int i=0; i<filter_total; i++) A_fil[i] = fliter_v[i];
  
  float *d_input = NULL;
    err = cudaMalloc((void **)&d_input, pad_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  float *d_filter = NULL;
    err = cudaMalloc((void **)&d_filter, filter_size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device d_filter (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  float *d_output = NULL;
  err = cudaMalloc((void **)&d_output, size);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device d_output (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  
  err = cudaMemcpy(d_input, A_in, pad_size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy input array d_input from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_filter, A_fil, filter_size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy qbit gate d_filter from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  int threadsPerBlock = 256;
  int blocksPerGrid =(pad_total + threadsPerBlock - 1) / threadsPerBlock;

  conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_filter, d_output, f_dimen, h, w, padded_width, numElements, pad_total);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch conv2d_kernel kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(A_out, d_output, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy out array d_output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  for (int i = 0; i < numElements; ++i)
    {
        printf("%0.3f\n", A_out[i]);
    }

  /*for (int i = 0; i < pad_total; ++i)
  {
      printf(" %0.3f \n", A_in[i]);
  }*/

  err = cudaFree(d_input);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_filter);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_filter (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device d_output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  free(A_in);
  free(A_out);
  free(A_fil);

  err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  return 0;
}