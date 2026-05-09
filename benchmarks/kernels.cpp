#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

namespace {

void write_output(const char *path, const std::vector<float> &values) {
  FILE *fp = std::fopen(path, "w");
  if (!fp) {
    std::perror(path);
    std::exit(2);
  }

  for (size_t i = 0; i < values.size(); ++i) {
    std::fprintf(fp, "%.9g%c", values[i],
                 (i + 1 == values.size()) ? '\n' : ' ');
  }
  std::fclose(fp);
}

NOINLINE size_t run_gemm(float *out) {
  constexpr int M = 4;
  constexpr int N = 4;
  constexpr int K = 4;
  float A[M][K];
  float B[K][N];
  float C[M][N];

  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k)
      A[i][k] = static_cast<float>(i + k + 1);

  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      B[k][j] = static_cast<float>(k - j + 1);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k)
        sum += A[i][k] * B[k][j];
      C[i][j] = sum;
    }
  }

  size_t index = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      out[index++] = C[i][j];
  return index;
}

NOINLINE size_t run_softmax(float *out) {
  constexpr int N = 16;
  float x[N];
  for (int i = 0; i < N; ++i)
    x[i] = 0.25f * static_cast<float>((i % 7) - 3);

  float max_value = x[0];
  for (int i = 1; i < N; ++i)
    if (x[i] > max_value)
      max_value = x[i];

  float sum = 0.0f;
  float y[N];
  for (int i = 0; i < N; ++i) {
    y[i] = std::exp(x[i] - max_value);
    sum += y[i];
  }

  for (int i = 0; i < N; ++i)
    out[i] = y[i] / sum;
  return N;
}

NOINLINE size_t run_layernorm(float *out) {
  constexpr int Rows = 4;
  constexpr int Cols = 8;
  constexpr float Eps = 1.0e-5f;
  float x[Rows][Cols];
  float gamma[Cols];
  float beta[Cols];

  for (int j = 0; j < Cols; ++j) {
    gamma[j] = 1.0f + 0.02f * static_cast<float>(j);
    beta[j] = -0.05f * static_cast<float>(j % 3);
  }

  for (int i = 0; i < Rows; ++i)
    for (int j = 0; j < Cols; ++j)
      x[i][j] = 0.1f * static_cast<float>((i + 1) * (j - 3));

  size_t index = 0;
  for (int i = 0; i < Rows; ++i) {
    float mean = 0.0f;
    for (int j = 0; j < Cols; ++j)
      mean += x[i][j];
    mean /= static_cast<float>(Cols);

    float variance = 0.0f;
    for (int j = 0; j < Cols; ++j) {
      float centered = x[i][j] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<float>(Cols);

    float inv_std = 1.0f / std::sqrt(variance + Eps);
    for (int j = 0; j < Cols; ++j)
      out[index++] = (x[i][j] - mean) * inv_std * gamma[j] + beta[j];
  }
  return index;
}

NOINLINE size_t run_attention(float *out) {
  constexpr int Seq = 4;
  constexpr int Dim = 8;
  float q[Seq][Dim];
  float k[Seq][Dim];
  float v[Seq][Dim];
  float scores[Seq][Seq];
  float probs[Seq][Seq];

  for (int i = 0; i < Seq; ++i) {
    for (int d = 0; d < Dim; ++d) {
      q[i][d] = 0.05f * static_cast<float>((i + 1) * (d + 1));
      k[i][d] = 0.04f * static_cast<float>((i - d) + 2);
      v[i][d] = 0.03f * static_cast<float>((i + d) % 5 - 2);
    }
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(Dim));
  for (int i = 0; i < Seq; ++i) {
    for (int j = 0; j < Seq; ++j) {
      float dot = 0.0f;
      for (int d = 0; d < Dim; ++d)
        dot += q[i][d] * k[j][d];
      scores[i][j] = dot * scale;
    }
  }

  for (int i = 0; i < Seq; ++i) {
    float max_value = scores[i][0];
    for (int j = 1; j < Seq; ++j)
      if (scores[i][j] > max_value)
        max_value = scores[i][j];

    float sum = 0.0f;
    for (int j = 0; j < Seq; ++j) {
      probs[i][j] = std::exp(scores[i][j] - max_value);
      sum += probs[i][j];
    }
    for (int j = 0; j < Seq; ++j)
      probs[i][j] /= sum;
  }

  size_t index = 0;
  for (int i = 0; i < Seq; ++i) {
    for (int d = 0; d < Dim; ++d) {
      float value = 0.0f;
      for (int j = 0; j < Seq; ++j)
        value += probs[i][j] * v[j][d];
      out[index++] = value;
    }
  }
  return index;
}

} // namespace

int main(int argc, char **argv) {
  const char *kernel = argc > 1 ? argv[1] : "gemm";
  const char *output = argc > 2 ? argv[2] : "output.txt";

  std::vector<float> values(64);
  size_t count = 0;
  if (std::strcmp(kernel, "gemm") == 0) {
    count = run_gemm(values.data());
  } else if (std::strcmp(kernel, "softmax") == 0) {
    count = run_softmax(values.data());
  } else if (std::strcmp(kernel, "layernorm") == 0) {
    count = run_layernorm(values.data());
  } else if (std::strcmp(kernel, "attention") == 0) {
    count = run_attention(values.data());
  } else {
    std::fprintf(stderr, "unknown kernel: %s\n", kernel);
    return 2;
  }

  values.resize(count);
  write_output(output, values);
  return 0;
}
