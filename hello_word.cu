#include <stdio.h>
__global__ void kernel(void) {

}

int main(void) {
    kernel<<<1,1>>>();
        printf("Hello Word ! \n");
        return 0; 
}