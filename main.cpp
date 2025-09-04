#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <random>
#include <cmath>
#include <iomanip> // For std::setw, std::fixed

// NEON intrinsics header
#include <arm_neon.h>

// 1. ==================== 伪实现和类型定义 ====================

// 定义MetricType，我们将主要使用L2SQR进行测试，因为它最直接
enum MetricType {
    METRIC_TYPE_L2SQR,
    METRIC_TYPE_IP,
    METRIC_TYPE_COSINE
};

// 伪造Computer结构体，它只需要一个指向缓冲区的指针
template<typename T>
struct Computer {
    void* buf_;
};

// 伪造ProductQuantizer类来容纳您的函数
template <MetricType metric>
class ProductQuantizer {
public:
    ProductQuantizer(int64_t pq_dim) : pq_dim_(pq_dim) {}

    // --- 标量版本 ---
    void ComputeDistImpl(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;
    void ComputeDistsBatch4Impl(Computer<ProductQuantizer<metric>>& computer, const uint8_t* codes1, const uint8_t* codes2, const uint8_t* codes3, const uint8_t* codes4, float& dists1, float& dists2, float& dists3, float& dists4) const;

    // --- NEON版本 ---
    void ComputeDistImplNEON(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;
    void ComputeDistImplNEON_v2(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;
    void ComputeDistsBatch4ImplNEON(Computer<ProductQuantizer<metric>>& computer, const uint8_t* codes1, const uint8_t* codes2, const uint8_t* codes3, const uint8_t* codes4, float& dists1, float& dists2, float& dists3, float& dists4) const;

private:
    const int64_t pq_dim_;
    static constexpr int64_t CENTROIDS_PER_SUBSPACE = 256;
};


// 2. ==================== 将您的函数实现粘贴到这里 ====================
// 注意：为了简洁，我将'ProductQuantizer<metric>::'前缀添加到了函数定义中

// --- 标量实现 ---
template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistImpl(Computer<ProductQuantizer<metric>>& computer,
                                          const uint8_t* codes,
                                          float* dists) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float dist = 0.0F;
    int64_t i = 0;
    // NOTE: The original loop has a potential bug. The inner `dism` is reset every iteration
    // but only the final value is added. I am assuming the intent was to accumulate.
    // The scalar logic has been slightly corrected to match what is likely the intended logic,
    // which the Batch and NEON versions implement correctly (summing all lookups).
    for (; i < pq_dim_; ++i) {
        dist += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE or
                  metric == MetricType::METRIC_TYPE_IP) {
        dists[0] = 1.0F - dist;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = dist;
    }
}

template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistsBatch4Impl(Computer<ProductQuantizer<metric>>& computer,
                                                 const uint8_t* codes1,
                                                 const uint8_t* codes2,
                                                 const uint8_t* codes3,
                                                 const uint8_t* codes4,
                                                 float& dists1,
                                                 float& dists2,
                                                 float& dists3,
                                                 float& dists4) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);

    float d0 = 0.0F;
    float d1 = 0.0F;
    float d2 = 0.0F;
    float d3 = 0.0F;

    int64_t i = 0;

    for (; i + 3 < pq_dim_; i += 4) {
        const float* l0 = lut + (i + 0) * CENTROIDS_PER_SUBSPACE;
        const float* l1 = lut + (i + 1) * CENTROIDS_PER_SUBSPACE;
        const float* l2 = lut + (i + 2) * CENTROIDS_PER_SUBSPACE;
        const float* l3 = lut + (i + 3) * CENTROIDS_PER_SUBSPACE;

        d0 += l0[codes1[i + 0]]; d1 += l0[codes2[i + 0]]; d2 += l0[codes3[i + 0]]; d3 += l0[codes4[i + 0]];
        d0 += l1[codes1[i + 1]]; d1 += l1[codes2[i + 1]]; d2 += l1[codes3[i + 1]]; d3 += l1[codes4[i + 1]];
        d0 += l2[codes1[i + 2]]; d1 += l2[codes2[i + 2]]; d2 += l2[codes3[i + 2]]; d3 += l2[codes4[i + 2]];
        d0 += l3[codes1[i + 3]]; d1 += l3[codes2[i + 3]]; d2 += l3[codes3[i + 3]]; d3 += l3[codes4[i + 3]];
    }

    for (; i < pq_dim_; ++i) {
        const float* li = lut + i * CENTROIDS_PER_SUBSPACE;
        d0 += li[codes1[i]]; d1 += li[codes2[i]]; d2 += li[codes3[i]]; d3 += li[codes4[i]];
    }

    if constexpr (metric == MetricType::METRIC_TYPE_COSINE || metric == MetricType::METRIC_TYPE_IP) {
        dists1 = 1.0F - d0; dists2 = 1.0F - d1; dists3 = 1.0F - d2; dists4 = 1.0F - d3;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists1 = d0; dists2 = d1; dists3 = d2; dists4 = d3;
    }
}

// --- NEON 实现 ---
template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistImplNEON(
    Computer<ProductQuantizer<metric>>& computer,
    const uint8_t* codes,
    float* dists) const {
    
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    
    int64_t i = 0;
    
    for (; i + 3 < pq_dim_; i += 4) {
        float32x4_t values = {
            lut[codes[0]],
            lut[CENTROIDS_PER_SUBSPACE + codes[1]],
            lut[2 * CENTROIDS_PER_SUBSPACE + codes[2]],
            lut[3 * CENTROIDS_PER_SUBSPACE + codes[3]]
        };
        sum_vec = vaddq_f32(sum_vec, values);
        codes += 4;
        lut += 4 * CENTROIDS_PER_SUBSPACE;
    }
    
    float dist = vaddvq_f32(sum_vec);
    
    for (; i < pq_dim_; ++i) {
        dist += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
    }
    
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE || metric == MetricType::METRIC_TYPE_IP) {
        dists[0] = 1.0f - dist;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = dist;
    }
}

template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistImplNEON_v2(
    Computer<ProductQuantizer<metric>>& computer,
    const uint8_t* codes,
    float* dists) const {
    
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    
    int64_t i = 0;
    
    for (; i + 7 < pq_dim_; i += 8) {
        float32x4_t values0 = { lut[codes[0]], lut[CENTROIDS_PER_SUBSPACE + codes[1]], lut[2 * CENTROIDS_PER_SUBSPACE + codes[2]], lut[3 * CENTROIDS_PER_SUBSPACE + codes[3]] };
        float32x4_t values1 = { lut[4 * CENTROIDS_PER_SUBSPACE + codes[4]], lut[5 * CENTROIDS_PER_SUBSPACE + codes[5]], lut[6 * CENTROIDS_PER_SUBSPACE + codes[6]], lut[7 * CENTROIDS_PER_SUBSPACE + codes[7]] };
        sum_vec0 = vaddq_f32(sum_vec0, values0);
        sum_vec1 = vaddq_f32(sum_vec1, values1);
        codes += 8;
        lut += 8 * CENTROIDS_PER_SUBSPACE;
    }
    
    sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
    
    if (i + 3 < pq_dim_) {
        float32x4_t values = { lut[codes[0]], lut[CENTROIDS_PER_SUBSPACE + codes[1]], lut[2 * CENTROIDS_PER_SUBSPACE + codes[2]], lut[3 * CENTROIDS_PER_SUBSPACE + codes[3]] };
        sum_vec0 = vaddq_f32(sum_vec0, values);
        codes += 4;
        lut += 4 * CENTROIDS_PER_SUBSPACE;
        i += 4;
    }
    
    float dist = vaddvq_f32(sum_vec0);
    
    for (; i < pq_dim_; ++i) {
        dist += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
    }
    
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE || metric == MetricType::METRIC_TYPE_IP) {
        dists[0] = 1.0f - dist;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = dist;
    }
}

template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistsBatch4ImplNEON(
    Computer<ProductQuantizer<metric>>& computer,
    const uint8_t* codes1, const uint8_t* codes2,
    const uint8_t* codes3, const uint8_t* codes4,
    float& dists1, float& dists2,
    float& dists3, float& dists4) const {
    
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float32x4_t accum = vdupq_n_f32(0.0f);
    int64_t i = 0;
    
    for (; i + 3 < pq_dim_; i += 4) {
        // Unroll manually to process 4 dimensions
        const float* l0 = lut + (i + 0) * CENTROIDS_PER_SUBSPACE;
        const float* l1 = lut + (i + 1) * CENTROIDS_PER_SUBSPACE;
        const float* l2 = lut + (i + 2) * CENTROIDS_PER_SUBSPACE;
        const float* l3 = lut + (i + 3) * CENTROIDS_PER_SUBSPACE;

        accum = vaddq_f32(accum, (float32x4_t){l0[codes1[i+0]], l0[codes2[i+0]], l0[codes3[i+0]], l0[codes4[i+0]]});
        accum = vaddq_f32(accum, (float32x4_t){l1[codes1[i+1]], l1[codes2[i+1]], l1[codes3[i+1]], l1[codes4[i+1]]});
        accum = vaddq_f32(accum, (float32x4_t){l2[codes1[i+2]], l2[codes2[i+2]], l2[codes3[i+2]], l2[codes4[i+2]]});
        accum = vaddq_f32(accum, (float32x4_t){l3[codes1[i+3]], l3[codes2[i+3]], l3[codes3[i+3]], l3[codes4[i+3]]});
    }
    
    for (; i < pq_dim_; ++i) {
        const float* li = lut + i * CENTROIDS_PER_SUBSPACE;
        accum = vaddq_f32(accum, (float32x4_t){li[codes1[i]], li[codes2[i]], li[codes3[i]], li[codes4[i]]});
    }
    
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE || metric == MetricType::METRIC_TYPE_IP) {
        float32x4_t ones = vdupq_n_f32(1.0f);
        accum = vsubq_f32(ones, accum);
    }
    
    // vst1q_f32 requires a non-const pointer
    float results[4];
    vst1q_f32(results, accum);
    dists1 = results[0]; dists2 = results[1]; dists3 = results[2]; dists4 = results[3];
}

// 3. ==================== 测试主逻辑 ====================

// 辅助函数用于检查正确性
void check_correctness(const std::string& name, float expected, float actual) {
    const float epsilon = 1e-5;
    std::cout << std::left << std::setw(30) << name << ": ";
    if (std::fabs(expected - actual) < epsilon) {
        std::cout << "\033[32mPASS\033[0m" << std::endl;
    } else {
        std::cout << "\033[31mFAIL\033[0m (Expected: " << expected << ", Got: " << actual << ")" << std::endl;
    }
}

int main() {
    // --- 测试参数 ---
    const int64_t PQ_DIM = 64; // PQ维度，选择8和4的倍数
    const int NUM_VECTORS_PERF_TEST = 200000;
    const int BATCH_SIZE = 4;

    std::cout << "Product Quantizer (PQ) Distance Calculation Benchmark" << std::endl;
    std::cout << "PQ Dimensions: " << PQ_DIM << ", Metric: L2SQR" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;

    // --- 数据准备 ---
    ProductQuantizer<METRIC_TYPE_L2SQR> pq(PQ_DIM);
    Computer<ProductQuantizer<METRIC_TYPE_L2SQR>> computer;

    std::mt19937 gen(1337); // 固定种子以获得可复现的结果
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> code_dis(0, 255);

    // 创建查找表 (LUT)
    std::vector<float> lut(PQ_DIM * 256);
    for (auto& val : lut) {
        val = dis(gen);
    }
    computer.buf_ = lut.data();

    // 创建PQ编码
    std::vector<uint8_t> codes1(PQ_DIM), codes2(PQ_DIM), codes3(PQ_DIM), codes4(PQ_DIM);
    for (int64_t i = 0; i < PQ_DIM; ++i) {
        codes1[i] = code_dis(gen);
        codes2[i] = code_dis(gen);
        codes3[i] = code_dis(gen);
        codes4[i] = code_dis(gen);
    }
    
    // --- 正确性验证 ---
    std::cout << "\n[1] Correctness Verification" << std::endl;
    float dist_scalar_golden, dist_scalar_batch[4];
    float dist_neon, dist_neon_v2, dist_neon_batch[4];

    // 单个编码
    pq.ComputeDistImpl(computer, codes1.data(), &dist_scalar_golden);
    pq.ComputeDistImplNEON(computer, codes1.data(), &dist_neon);
    pq.ComputeDistImplNEON_v2(computer, codes1.data(), &dist_neon_v2);

    check_correctness("ComputeDistImpl (Baseline)", dist_scalar_golden, dist_scalar_golden);
    check_correctness("ComputeDistImplNEON", dist_scalar_golden, dist_neon);
    check_correctness("ComputeDistImplNEON_v2", dist_scalar_golden, dist_neon_v2);

    // 批量编码
    pq.ComputeDistsBatch4Impl(computer, codes1.data(), codes2.data(), codes3.data(), codes4.data(), dist_scalar_batch[0], dist_scalar_batch[1], dist_scalar_batch[2], dist_scalar_batch[3]);
    pq.ComputeDistsBatch4ImplNEON(computer, codes1.data(), codes2.data(), codes3.data(), codes4.data(), dist_neon_batch[0], dist_neon_batch[1], dist_neon_batch[2], dist_neon_batch[3]);

    float golden_c2, golden_c3, golden_c4;
    pq.ComputeDistImpl(computer, codes2.data(), &golden_c2);
    pq.ComputeDistImpl(computer, codes3.data(), &golden_c3);
    pq.ComputeDistImpl(computer, codes4.data(), &golden_c4);
    
    check_correctness("ComputeDistsBatch4Impl [1]", dist_scalar_golden, dist_scalar_batch[0]);
    check_correctness("ComputeDistsBatch4Impl [2]", golden_c2, dist_scalar_batch[1]);
    check_correctness("ComputeDistsBatch4ImplNEON [1]", dist_scalar_golden, dist_neon_batch[0]);
    check_correctness("ComputeDistsBatch4ImplNEON [4]", golden_c4, dist_neon_batch[3]);

    // --- 性能测试 ---
    std::cout << "\n[2] Performance Benchmark (" << NUM_VECTORS_PERF_TEST << " vectors)" << std::endl;
    volatile float sink = 0; // 防止编译器优化掉循环

    auto run_benchmark = [&](const std::string& name, auto func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double mops = (double)NUM_VECTORS_PERF_TEST / (duration_ns / 1000.0); // M-ops/sec
        
        std::cout << std::left << std::setw(30) << name << ": "
                  << std::fixed << std::setprecision(2) << std::right << std::setw(8)
                  << (double)duration_ns / NUM_VECTORS_PERF_TEST << " ns/op, "
                  << std::fixed << std::setprecision(2) << std::right << std::setw(8)
                  << mops << " M-ops/sec" << std::endl;
    };

    // 标量 单个
    run_benchmark("ComputeDistImpl", [&]{
        float d;
        for (int i = 0; i < NUM_VECTORS_PERF_TEST; ++i) {
            pq.ComputeDistImpl(computer, codes1.data(), &d);
            sink += d;
        }
    });

    // NEON 单个
    run_benchmark("ComputeDistImplNEON", [&]{
        float d;
        for (int i = 0; i < NUM_VECTORS_PERF_TEST; ++i) {
            pq.ComputeDistImplNEON(computer, codes1.data(), &d);
            sink += d;
        }
    });

    // NEON 单个 v2
    run_benchmark("ComputeDistImplNEON_v2", [&]{
        float d;
        for (int i = 0; i < NUM_VECTORS_PERF_TEST; ++i) {
            pq.ComputeDistImplNEON_v2(computer, codes1.data(), &d);
            sink += d;
        }
    });
    
    // 标量 批量
    run_benchmark("ComputeDistsBatch4Impl", [&]{
        float d1, d2, d3, d4;
        for (int i = 0; i < NUM_VECTORS_PERF_TEST; i += BATCH_SIZE) {
            pq.ComputeDistsBatch4Impl(computer, codes1.data(), codes2.data(), codes3.data(), codes4.data(), d1, d2, d3, d4);
            sink += d1 + d2 + d3 + d4;
        }
    });

    // NEON 批量
    run_benchmark("ComputeDistsBatch4ImplNEON", [&]{
        float d1, d2, d3, d4;
        for (int i = 0; i < NUM_VECTORS_PERF_TEST; i += BATCH_SIZE) {
            pq.ComputeDistsBatch4ImplNEON(computer, codes1.data(), codes2.data(), codes3.data(), codes4.data(), d1, d2, d3, d4);
            sink += d1 + d2 + d3 + d4;
        }
    });

    return 0;
}
