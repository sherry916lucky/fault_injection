// gemm.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 4
#define N 4
#define K 4

void gemm(float A[M][K], float B[K][N], float C[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main() {
    float A[M][K];
    float B[K][N];
    float C[M][N];

    // 初始化 A
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A[i][k] = (float)(i + k + 1);
        }
    }

    // 初始化 B
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            B[k][j] = (float)(k - j + 1);
        }
    }

    gemm(A, B, C);

    // 保存 golden output
    FILE *fp = fopen("golden_output.txt", "w");
    if (fp == NULL) {
        printf("Failed to open file.\n");
        return 1;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%.6f ", C[i][j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    printf("Golden output saved to golden_output.txt\n");

    return 0;
}