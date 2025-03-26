#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Gaussian function
__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

// fonction appelée depuis le CPU mais exécutée sur le GPU ! 
// ici chaque pixel va être traiter par un thread 
__global__ void bilateral_filter_kernel(
    const unsigned char* d_src,    // image source sur le GPU
    unsigned char* d_dst,          // image destination sur le GPU
    int width, int height, int channels,
    const float* d_spatial_weights, // poids spatiaux sur le GPU
    int d, float sigma_color) 
{   
    // on optimise l'accès à la mémoire globale en lisant les données de manière colonnales lorsqu'on travail sur les colonnes d'une matrice
    // calcul des indices des threads dans une grille 2d de blocs en cuda ces indices permettent de ref les élém d'une matrice stockée en mémoire globale
    // on regarde les pixels voisins dans une petite fenêtre d × d, on calcule les poids et on fait une moyenne pondérée.
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // colonne
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // ligne
    int radius = d / 2;

    // Limiter aux pixels valides (on ne veut pas les bords)
    if (x < radius || x >= width - radius || y < radius || y >= height - radius) return;

    int center_idx = (y * width + x) * channels;

    for (int c = 0; c < channels; c++) {
        float filtered_value = 0.0f;
        float weight_sum = 0.0f;

        float center_val = d_src[center_idx + c];

        // Boucle sur la fenêtre locale
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                // Positon des voisins 
                int nx = x + j - radius;
                int ny = y + i - radius;

                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                // on lie la couleur du voisin 
                int neighbor_idx = (ny * width + nx) * channels + c;
                float neighbor_val = d_src[neighbor_idx];
                // on calcule et on fait la moy pondérée
                float spatial = d_spatial_weights[i * d + j];
                float range = gaussian(fabsf(neighbor_val - center_val), sigma_color);
                float weight = spatial * range;

                filtered_value += weight * neighbor_val;
                weight_sum += weight;
            }
        }
        // Et on n'oublie pas de NORMALISER 
        d_dst[center_idx + c] = (unsigned char)(filtered_value / (weight_sum + 1e-6f));
    }
}


// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        // prend le premier argument qui est donc l'image
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }
    // le cpu va charger l'image
    int width, height, channels;
    unsigned char *h_src = stbi_load(argv[1], &width, &height, &channels, 0); // CPU
    if (!h_src) {
        printf("Error loading image!\n");
        return 1;
    }

    if (width <= 5 || height <= 5) {
        printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
        stbi_image_free(h_src);
        return 1;
    }

    // Allocation pour image filtrée (CPU) (Host)
    size_t img_size = width * height * channels * sizeof(unsigned char);
    unsigned char *h_dst = (unsigned char *)malloc(img_size);
    if (!h_dst) {
        printf("Memory allocation for filtered image failed!\n");
        stbi_image_free(h_src);
        return 1;
    }

    // Préparation GPU 

    // cudaMalloc : Alloue des octets de taille de mémoire linéaire sur le périphérique et renvoie dans *devPtr un pointeur à la mémoire allouée. 
    // La mémoire allouée est convenablement alignée pour tout type de variable. 
    // La mémoire n’est pas effacée. cudaMalloc() renvoie cudaErrorMemoryAllocation en cas d’échec.

    unsigned char *d_src = nullptr;
    unsigned char *d_dst = nullptr;
    cudaMalloc(&d_src, img_size);
    cudaMalloc(&d_dst, img_size);
    // permet de copier des données entre la mémoire cpu et la mémoire gpu
    cudaMemcpy(d_src, h_src, img_size, cudaMemcpyHostToDevice);

    // Préparation des poids spatiaux

    int filter_d = 5;
    float sigma_color = 75.0f;
    float sigma_space = 75.0f;
    int radius = filter_d / 2;

    // on va calucler les poids et on les calcule sur le CPU (car c’est simple), puis on les envoie au GPU
    float *h_spatial_weights = (float *)malloc(filter_d * filter_d * sizeof(float));
    if (!h_spatial_weights) {
        printf("Memory allocation for spatial weights failed!\n");
        stbi_image_free(h_src);
        free(h_dst);
        cudaFree(d_src);
        cudaFree(d_dst);
        return 1;
    }

    for (int i = 0; i < filter_d; i++) {
        for (int j = 0; j < filter_d; j++) {
            int x = i - radius;
            int y = j - radius;
            h_spatial_weights[i * filter_d + j] = expf(-(x * x + y * y) / (2.0f * sigma_space * sigma_space));
        }
    }

    float *d_spatial_weights = nullptr;
    cudaMalloc(&d_spatial_weights, filter_d * filter_d * sizeof(float));
    cudaMemcpy(d_spatial_weights, h_spatial_weights, filter_d * filter_d * sizeof(float), cudaMemcpyHostToDevice);

    // Lancement du kernel 
    // On divise l’image en blocs de 16x16 threads et puis on appelle le kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    bilateral_filter_kernel<<<gridSize, blockSize>>>(
        d_src, d_dst, width, height, channels,
        d_spatial_weights, filter_d, sigma_color
    );
    cudaDeviceSynchronize();

    // Récupération du résultat

    cudaMemcpy(h_dst, d_dst, img_size, cudaMemcpyDeviceToHost);

    // Sauvegarde

    if (!stbi_write_png(argv[2], width, height, channels, h_dst, width * channels)) {
        printf("Error saving the image!\n");
        free(h_dst);
        stbi_image_free(h_src);
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_spatial_weights);
        free(h_spatial_weights);
        return 1;
    }

    // Nettoyage

    // cudaFree : Libère l’espace mémoire indiqué par devPtr, qui doit avoir été retourné par un appel précédent à cudaMalloc() ou cudaMallocPitch().
    //Sinon, ou si cudaFree(devPtr) a déjà été appelé auparavant, une erreur est renvoyée. 
    //Si devPtr est 0, aucune opération n’est effectuée. cudaFree() renvoie cudaErrorInvalidDevicePointer en cas d’échec. (la doc nvidia)

    stbi_image_free(h_src);
    free(h_dst);
    free(h_spatial_weights);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_spatial_weights);

    printf("Bilateral filtering (CUDA) complete. Output saved as %s\n", argv[2]);
    return 0;
}

