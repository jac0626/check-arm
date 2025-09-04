#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <random>
#include <cmath>
#include <iomanip>

// SIMD headers
#include <arm_neon.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

// ==================== 伪实现和类型定义 ====================
enum MetricType { METRIC_TYPE_L2SQR, METRIC_TYPE_IP, METRIC_TYPE_COSINE };
template<typename T> struct Computer { void* buf_; };

template <MetricType metric>
class ProductQuantizer {
public:
    ProductQuantizer(int64_t pq_dim) : pq_dim_(pq_dim) {}

    // --- 标量版本 ---
    void ComputeDistImpl(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;
    void ComputeDistsBatch4Impl(Computer<ProductQuantizer<metric>>& computer, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, const uint8_t* c4, float& d1, float& d2, float& d3, float& d4) const;

    // --- NEON版本 ---
    void ComputeDistImplNEON_v2(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;
    void ComputeDistsBatch4ImplNEON(Computer<ProductQuantizer<metric>>& computer, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, const uint8_t* c4, float& d1, float& d2, float& d3, float& d4) const;
    
#ifdef __ARM_FEATURE_SVE
    // --- SVE版本 ---
    void ComputeDistImplSVE(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;
    void ComputeDistsBatch4ImplSVE(Computer<ProductQuantizer<metric>>& computer, const uint8_t* c1, const uint8_t* c2, const uint8_t* c3, const uint8_t* c4, float& d1, float& d2, float& d3, float& d4) const;
#endif

private:
    const int64_t pq_dim_;
    static constexpr int64_t CENTROIDS_PER_SUBSPACE = 256;
};

// ==================== 函数实现 (包括 SVE) ====================
// (粘贴之前的标量和NEON实现于此...)
// 为了简洁，这里只展示SVE的实现和修改后的main

// --- 标量实现 ---
template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistImpl(Computer<ProductQuantizer<metric>>& computer,
                                          const uint8_t* codes,
                                          float* dists) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float dist = 0.0F;
    for (int64_t i = 0; i < pq_dim_; ++i) {
        dist += lut[i * CENTROIDS_PER_SUBSPACE + codes[i]];
    }
    dists[0] = dist; // Simplified for L2SQR
}

template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistsBatch4Impl(Computer<ProductQuantizer<metric>>& computer,
                                                 const uint8_t* codes1, const uint8_t* codes2,
                                                 const uint8_t* codes3, const uint8_t* codes4,
                                                 float& dists1, float& dists2,
                                                 float& dists3, float& dists4) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float d0 = 0.0F, d1 = 0.0F, d2 = 0.0F, d3 = 0.0F;
    for (int64_t i = 0; i < pq_dim_; ++i) {
        const float* li = lut + i * CENTROIDS_PER_SUBSPACE;
        d0 += li[codes1[i]]; d1 += li[codes2[i]];
        d2 += li[codes3[i]]; d3 += li[codes4[i]];
    }
    dists1 = d0; dists2 = d1; dists3 = d2; dists4 = d3;
}

// --- NEON 实现 ---
template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistImplNEON_v2(
    Computer<ProductQuantizer<metric>>& computer,
    const uint8_t* codes,
    float* dists) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float32x4_t sum_vec0 = vdupq_n_f32(0.0f), sum_vec1 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 7 < pq_dim_; i += 8) {
        sum_vec0 = vaddq_f32(sum_vec0, (float32x4_t){lut[i*256+codes[i]], lut[(i+1)*256+codes[i+1]], lut[(i+2)*256+codes[i+2]], lut[(i+3)*256+codes[i+3]]});
        sum_vec1 = vaddq_f32(sum_vec1, (float32x4_t){lut[(i+4)*256+codes[i+4]], lut[(i+5)*256+codes[i+5]], lut[(i+6)*256+codes[i+6]], lut[(i+7)*256+codes[i+7]]});
    }
    sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
    if (i + 3 < pq_dim_) {
        sum_vec0 = vaddq_f32(sum_vec0, (float32x4_t){lut[i*256+codes[i]], lut[(i+1)*256+codes[i+1]], lut[(i+2)*256+codes[i+2]], lut[(i+3)*256+codes[i+3]]});
        i += 4;
    }
    float dist = vaddvq_f32(sum_vec0);
    for (; i < pq_dim_; ++i) { dist += lut[i * 256 + codes[i]]; }
    dists[0] = dist;
}

template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistsBatch4ImplNEON(
    Computer<ProductQuantizer<metric>>& computer,
    const uint8_t* c1, const uint8_t* c2,
    const uint8_t* c3, const uint8_t* c4,
    float& d1, float& d2,
    float& d3, float& d4) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float32x4_t accum = vdupq_n_f32(0.0f);
    for (int64_t i = 0; i < pq_dim_; ++i) {
        const float* li = lut + i * 256;
        accum = vaddq_f32(accum, (float32x4_t){li[c1[i]], li[c2[i]], li[c3[i]], li[c4[i]]});
    }
    float results[4]; vst1q_f32(results, accum);
    d1 = results[0]; d2 = results[1]; d3 = results[2]; d4 = results[3];
}


#ifdef __ARM_FEATURE_SVE
template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistImplSVE(
    Computer<ProductQuantizer<metric>>& computer,
    const uint8_t* codes,
    float* dists) const {

    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float dist = 0.0f;
    svfloat32_t sum_vec = svdup_n_f32(0.0f);
    svbool_t pg_true = svptrue_b32();
    uint64_t VEC_LEN = svcntw();

    int64_t i = 0;
    // As noted, this isn't a great use case for SVE due to gather load limitations.
    // We primarily vectorize the accumulation.
    for (; i + VEC_LEN <= pq_dim_; i += VEC_LEN) {
        float temp_vals[VEC_LEN];
        for (uint64_t j = 0; j < VEC_LEN; ++j) {
            temp_vals[j] = lut[(i + j) * CENTROIDS_PER_SUBSPACE + codes[i + j]];
        }
        svfloat32_t current_vals = svld1_f32(pg_true, temp_vals);
        sum_vec = svadd_f32_m(pg_true, sum_vec, current_vals);
    }
    
    dist = svaddv_f32(pg_true, sum_vec);

    for (; i < pq_dim_; ++i) {
        dist += lut[i * CENTROIDS_PER_SUBSPACE + codes[i]];
    }
    
    dists[0] = dist; // Simplified for L2SQR
}

template <MetricType metric>
void ProductQuantizer<metric>::ComputeDistsBatch4ImplSVE(
    Computer<ProductQuantizer<metric>>& computer,
    const uint8_t* codes1, const uint8_t* codes2,
    const uint8_t* codes3, const uint8_t* codes4,
    float& dists1, float& dists2,
    float& dists3, float& dists4) const {

    auto* lut = reinterpret_cast<float*>(computer.buf_);
    svfloat32_t accum_vec = svdup_n_f32(0.0f);
    svbool_t pg_4 = svwhilelt_b32(0, 4); // Predicate for first 4 lanes

    for (int64_t i = 0; i < pq_dim_; ++i) {
        const float* li = lut + i * CENTROIDS_PER_SUBSPACE;
        
        float vals_to_load[4] = {
            li[codes1[i]], li[codes2[i]], li[codes3[i]], li[codes4[i]]
        };
        
        svfloat32_t current_vals = svld1_f32(pg_4, vals_to_load);
        accum_vec = svadd_f32_m(pg_4, accum_vec, current_vals);
    }

    float results[4];
    svst1_f32(pg_4, results, accum_vec);
    dists1 = results[0]; dists2 = results[1];
    dists3 = results[2]; dists4 = results[3];
}
#endif

// ==================== 测试主逻辑 (更新后) ====================
// (辅助函数 check_correctness 和 main 函数的结构保持不变, 只增加SVE的调用)
void check_correctness(const std::string& name, float expected, float actual) {
    const float epsilon = 1e-4; // Increase epsilon slightly for cross-SIMD checks
    std::cout << std::left << std::setw(32) << name << ": ";
    if (std::fabs(expected - actual) < epsilon) {
        std::cout << "\033[32mPASS\033[0m" << std::endl;
    } else {
        std::cout << "\033[31mFAIL\033[0m (Expected: " << expected << ", Got: " << actual << ")" << std::endl;
    }
}

int main() {
    // --- 测试参数 ---
    const int64_t PQ_DIM = 64;
    const int NUM_VECTORS_PERF_TEST = 200000;
    const int BATCH_SIZE = 4;

    std::cout << "PQ Distance Calculation Benchmark" << std::endl;
    std::cout << "PQ Dimensions: " << PQ_DIM << ", Metric: L2SQR" << std::endl;
#ifdef __ARM_FEATURE_SVE
    std::cout << "SVE available. Vector length: " << svcntb() * 8 << " bits." << std::endl;
#else
    std::cout << "SVE not available." << std::endl;
#endif
    std::cout << "----------------------------------------------------" << std::endl;

    // --- 数据准备 ---
    ProductQuantizer<METRIC_TYPE_L2SQR> pq(PQ_DIM);
    Computer<ProductQuantizer<METRIC_TYPE_L2SQR>> computer;
    std::mt19937 gen(1337);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_int_distribution<> code_dis(0, 255);

    std::vector<float> lut(PQ_DIM * 256);
    for (auto& val : lut) { val = dis(gen); }
    computer.buf_ = lut.data();

    std::vector<uint8_t> c1(PQ_DIM), c2(PQ_DIM), c3(PQ_DIM), c4(PQ_DIM);
    for (int64_t i = 0; i < PQ_DIM; ++i) {
        c1[i] = code_dis(gen); c2[i] = code_dis(gen);
        c3[i] = code_dis(gen); c4[i] = code_dis(gen);
    }
    
    // --- 正确性验证 ---
    std::cout << "\n[1] Correctness Verification" << std::endl;
    float dist_scalar_golden, dist_scalar_batch[4];
    float dist_neon_v2, dist_neon_batch[4];
    pq.ComputeDistImpl(computer, c1.data(), &dist_scalar_golden);

    check_correctness("ComputeDistImpl (Baseline)", dist_scalar_golden, dist_scalar_golden);

    pq.ComputeDistImplNEON_v2(computer, c1.data(), &dist_neon_v2);
    check_correctness("ComputeDistImplNEON_v2", dist_scalar_golden, dist_neon_v2);

    pq.ComputeDistsBatch4Impl(computer, c1.data(), c2.data(), c3.data(), c4.data(), dist_scalar_batch[0], dist_scalar_batch[1], dist_scalar_batch[2], dist_scalar_batch[3]);
    check_correctness("ComputeDistsBatch4Impl [1]", dist_scalar_golden, dist_scalar_batch[0]);
    
    pq.ComputeDistsBatch4ImplNEON(computer, c1.data(), c2.data(), c3.data(), c4.data(), dist_neon_batch[0], dist_neon_batch[1], dist_neon_batch[2], dist_neon_batch[3]);
    check_correctness("ComputeDistsBatch4ImplNEON [1]", dist_scalar_golden, dist_neon_batch[0]);

#ifdef __ARM_FEATURE_SVE
    float dist_sve, dist_sve_batch[4];
    pq.ComputeDistImplSVE(computer, c1.data(), &dist_sve);
    check_correctness("ComputeDistImplSVE", dist_scalar_golden, dist_sve);

    pq.ComputeDistsBatch4ImplSVE(computer, c1.data(), c2.data(), c3.data(), c4.data(), dist_sve_batch[0], dist_sve_batch[1], dist_sve_batch[2], dist_sve_batch[3]);
    check_correctness("ComputeDistsBatch4ImplSVE [1]", dist_scalar_golden, dist_sve_batch[0]);
#endif

    // --- 性能测试 ---
    std::cout << "\n[2] Performance Benchmark (" << NUM_VECTORS_PERF_TEST << " vectors)" << std::endl;
    volatile float sink = 0; 
    auto run_benchmark = [&](const std::string& name, auto func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        double mops = (double)NUM_VECTORS_PERF_TEST / (duration_ns / 1000.0);
        std::cout << std::left << std::setw(32) << name << ": "
                  << std::fixed << std::setprecision(2) << std::right << std::setw(8)
                  << (double)duration_ns / NUM_VECTORS_PERF_TEST << " ns/op, "
                  << std::fixed << std::setprecision(2) << std::right << std::setw(8)
                  << mops << " M-ops/sec" << std::endl;
    };
    
    // (Existing benchmarks for scalar and NEON)
    run_benchmark("ComputeDistImpl (Scalar)", [&]{ float d; for (int i = 0; i < NUM_VECTORS_PERF_TEST; ++i) { pq.ComputeDistImpl(computer, c1.data(), &d); sink += d; }});
    run_benchmark("ComputeDistImplNEON_v2", [&]{ float d; for (int i = 0; i < NUM_VECTORS_PERF_TEST; ++i) { pq.ComputeDistImplNEON_v2(computer, c1.data(), &d); sink += d; }});
    run_benchmark("ComputeDistsBatch4Impl (Scalar)", [&]{ float d1,d2,d3,d4; for (int i = 0; i < NUM_VECTORS_PERF_TEST; i += 4) { pq.ComputeDistsBatch4Impl(computer, c1.data(), c2.data(), c3.data(), c4.data(), d1, d2, d3, d4); sink += d1; }});
    run_benchmark("ComputeDistsBatch4ImplNEON", [&]{ float d1,d2,d3,d4; for (int i = 0; i < NUM_VECTORS_PERF_TEST; i += 4) { pq.ComputeDistsBatch4ImplNEON(computer, c1.data(), c2.data(), c3.data(), c4.data(), d1, d2, d3, d4); sink += d1; }});


#ifdef __ARM_FEATURE_SVE
    run_benchmark("ComputeDistImplSVE", [&]{
        float d;
        for (int i = 0; i < NUM_VECTORS_PERF_TEST; ++i) {
            pq.ComputeDistImplSVE(computer, c1.data(), &d);
            sink += d;
        }
    });

    run_benchmark("ComputeDistsBatch4ImplSVE", [&]{
        float d1, d2, d3, d4;
        for (int i = 0; i < NUM_VECTORS_PERF_TEST; i += BATCH_SIZE) {
            pq.ComputeDistsBatch4ImplSVE(computer, c1.data(), c2.data(), c3.data(), c4.data(), d1, d2, d3, d4);
            sink += d1;
        }
    });
#endif

    return 0;
}
