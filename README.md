# Image compression using K-means clustering

OpenCL image compression alghoritm for a parallel computing college course. 

## Compile
1. `module load CUDA`
2. `gcc -o gpu gpu.c -fopenmp -O2 -lm -lOpenCL -Wl,-rpath,./ -L./ -l:libfreeimage.so.3`

## Run 
`./gpu input_image.png`

## Program arguments
`input_image [output_image] [-K clusters] [-I iterations] [-d device_index] [-s]`

* K - number of clusters used, number of colors in the output image (64 by default)
* I - number of iterations (50 by default)
* d - selected device (GPU) (0 by default)
* s - show available devices 

The input image should be in PNG format.

## Examples

<figure>
    <img align = "center" src="examples/seville_64.png" alt="Example image - 64 colors"/>
    <figcaption align = "center"><b>K = 64</b></figcaption>
</figure>

<figure>
    <img align = "center" src="examples/seville_32.png" alt="Example image - 32 colors"/>
    <figcaption align = "center"><b>K = 32</b></figcaption>
</figure>

<figure>
    <img align = "center" src="examples/seville_16.png" alt="Example image - 16 colors"/>
    <figcaption align = "center"><b>K = 16</b></figcaption>
</figure>
