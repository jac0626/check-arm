// sve_fp16_bf16_complete.cpp - Complete SVE implementation with test
#include <arm_sve.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdint>

#ifndef RESTRICT
#define RESTRICT __restrict
#endif

// ========== SVE Implementation ==========
union FP32Struct {
    uint32_t int_value;
    float float_value;
};

// Scalar conversion functions
float BF16ToFloat(const uint16_t bf16_value) {
    FP32Struct fp32;
    fp32.int_value = (static_cast<uint32_t>(bf16_value) << 16);
    return fp32.float_value;
}

uint16_t FloatToBF16(const float fp32_value) {
    FP32Struct fp32;
    fp32.float_value = fp32_value;
    return static_cast<uint16_t>((fp32.int_value + 0x8000) >> 16);
}

float FP16ToFloat(const uint16_t fp16_value) {
    uint32_t sign = (fp16_value >> 15) & 0x1;
    int32_t exp = ((fp16_value >> 10) & 0x1F) - 15;
    uint32_t mantissa = (fp16_value & 0x3FF) << 13;
    FP32Struct fp32;
    // Handle special cases
    if (exp == 16) { // Infinity or NaN
        exp = 128;
    } else if (exp == -15) { // Zero or denormal
        if (mantissa == 0) {
            fp32.int_value = sign << 31;
            return fp32.float_value;
        }
        // Handle denormal
        exp = -14;
    }
    fp32.int_value = (sign << 31) | ((exp + 127) << 23) | mantissa;
    return fp32.float_value;
}

uint16_t FloatToFP16(const float fp32_value) {
    FP32Struct fp32;
    fp32.float_value = fp32_value;
    uint16_t sign = (fp32.int_value >> 31) & 0x1;
    int32_t exp = ((fp32.int_value >> 23) & 0xFF) - 127;
    uint32_t mantissa = fp32.int_value & 0x007FFFFF;

    if (exp > 15) {
        exp = 15;
        mantissa = 0; // Infinity
    } else if (exp < -14) {
        exp = -15;
        mantissa = 0; // Zero
    }
    return (sign << 15) | ((exp + 15) << 10) | (mantissa >> 13);
}

// Simplified SVE implementations
namespace sve {

float BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
    
    svfloat32_t sum_vec = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t vl_f32 = svcntw();
    
    for (; i + vl_f32 <= dim; i += vl_f32) {
        svbool_t pg = svptrue_b32();
        
        svuint32_t query_u32 = svld1uh_u32(pg, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(pg, &codes_bf16[i]);
        
        query_u32 = svlsl_n_u32_x(pg, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(pg, codes_u32, 16);
        
        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);
        
        sum_vec = svmla_f32_x(pg, sum_vec, query_f32, codes_f32);
    }
    
    if (i < dim) {
        svbool_t pg = svwhilelt_b32(i, dim);
        
        svuint32_t query_u32 = svld1uh_u32(pg, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(pg, &codes_bf16[i]);
        
        query_u32 = svlsl_n_u32_x(pg, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(pg, codes_u32, 16);
        
        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);
        
        sum_vec = svmla_f32_x(pg, sum_vec, query_f32, codes_f32);
    }
    
    return svaddv_f32(svptrue_b32(), sum_vec);
}

float BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
    auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
    
    svfloat32_t sum_vec = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t vl_f32 = svcntw();
    
    for (; i + vl_f32 <= dim; i += vl_f32) {
        svbool_t pg = svptrue_b32();
        
        svuint32_t query_u32 = svld1uh_u32(pg, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(pg, &codes_bf16[i]);
        
        query_u32 = svlsl_n_u32_x(pg, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(pg, codes_u32, 16);
        
        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);
        
        svfloat32_t diff = svsub_f32_x(pg, query_f32, codes_f32);
        sum_vec = svmla_f32_x(pg, sum_vec, diff, diff);
    }
    
    if (i < dim) {
        svbool_t pg = svwhilelt_b32(i, dim);
        
        svuint32_t query_u32 = svld1uh_u32(pg, &query_bf16[i]);
        svuint32_t codes_u32 = svld1uh_u32(pg, &codes_bf16[i]);
        
        query_u32 = svlsl_n_u32_x(pg, query_u32, 16);
        codes_u32 = svlsl_n_u32_x(pg, codes_u32, 16);
        
        svfloat32_t query_f32 = svreinterpret_f32_u32(query_u32);
        svfloat32_t codes_f32 = svreinterpret_f32_u32(codes_u32);
        
        svfloat32_t diff = svsub_f32_x(pg, query_f32, codes_f32);
        sum_vec = svmla_f32_x(pg, sum_vec, diff, diff);
    }
    
    return svaddv_f32(svptrue_b32(), sum_vec);
}

float FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);
    
    svfloat32_t sum_vec = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t vl_f16 = svcnth();
    
    for (; i + vl_f16 <= dim; i += vl_f16) {
        svbool_t pg = svptrue_b16();
        
        // Load FP16 values directly
        svfloat16_t query_f16 = svld1_f16(pg, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(pg, &codes_fp16[i]);
        
        // Convert first half to FP32
        svbool_t pg_half = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_lo = svcvt_f32_f16_x(pg_half, query_f16);
        svfloat32_t codes_f32_lo = svcvt_f32_f16_x(pg_half, codes_f16);
        
        // Convert second half to FP32
        svfloat16_t query_f16_hi = svreinterpret_f16_u32(svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_hi = svreinterpret_f16_u32(svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_hi = svcvt_f32_f16_x(pg_half, query_f16_hi);
        svfloat32_t codes_f32_hi = svcvt_f32_f16_x(pg_half, codes_f16_hi);
        
        // Accumulate
        sum_vec = svmla_f32_x(svptrue_b32(), sum_vec, query_f32_lo, codes_f32_lo);
        sum_vec = svmla_f32_x(svptrue_b32(), sum_vec, query_f32_hi, codes_f32_hi);
    }
    
    // Handle remaining elements
    if (i < dim) {
        svbool_t pg = svwhilelt_b16(i, dim);
        
        svfloat16_t query_f16 = svld1_f16(pg, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(pg, &codes_fp16[i]);
        
        // Process elements that fit in a single FP32 vector
        uint64_t remaining = dim - i;
        uint64_t vl_f32 = svcntw();
        uint64_t to_process = (remaining < vl_f32) ? remaining : vl_f32;
        
        svbool_t pg_f32 = svwhilelt_b32(uint64_t(0), to_process);
        svfloat32_t query_f32 = svcvt_f32_f16_x(pg_f32, query_f16);
        svfloat32_t codes_f32 = svcvt_f32_f16_x(pg_f32, codes_f16);
        
        sum_vec = svmla_f32_x(pg_f32, sum_vec, query_f32, codes_f32);
    }
    
    return svaddv_f32(svptrue_b32(), sum_vec);
}

float FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    auto* query_fp16 = reinterpret_cast<const __fp16*>(query);
    auto* codes_fp16 = reinterpret_cast<const __fp16*>(codes);
    
    svfloat32_t sum_vec = svdup_n_f32(0.0f);
    uint64_t i = 0;
    uint64_t vl_f16 = svcnth();
    
    for (; i + vl_f16 <= dim; i += vl_f16) {
        svbool_t pg = svptrue_b16();
        
        // Load FP16 values directly
        svfloat16_t query_f16 = svld1_f16(pg, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(pg, &codes_fp16[i]);
        
        // Convert first half to FP32
        svbool_t pg_half = svptrue_pat_b16(SV_POW2);
        svfloat32_t query_f32_lo = svcvt_f32_f16_x(pg_half, query_f16);
        svfloat32_t codes_f32_lo = svcvt_f32_f16_x(pg_half, codes_f16);
        
        // Convert second half to FP32
        svfloat16_t query_f16_hi = svreinterpret_f16_u32(svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(query_f16), 16));
        svfloat16_t codes_f16_hi = svreinterpret_f16_u32(svlsr_n_u32_x(svptrue_b32(), svreinterpret_u32_f16(codes_f16), 16));
        svfloat32_t query_f32_hi = svcvt_f32_f16_x(pg_half, query_f16_hi);
        svfloat32_t codes_f32_hi = svcvt_f32_f16_x(pg_half, codes_f16_hi);
        
        // Compute differences and accumulate
        svfloat32_t diff_lo = svsub_f32_x(svptrue_b32(), query_f32_lo, codes_f32_lo);
        svfloat32_t diff_hi = svsub_f32_x(svptrue_b32(), query_f32_hi, codes_f32_hi);
        
        sum_vec = svmla_f32_x(svptrue_b32(), sum_vec, diff_lo, diff_lo);
        sum_vec = svmla_f32_x(svptrue_b32(), sum_vec, diff_hi, diff_hi);
    }
    
    // Handle remaining elements
    if (i < dim) {
        svbool_t pg = svwhilelt_b16(i, dim);
        
        svfloat16_t query_f16 = svld1_f16(pg, &query_fp16[i]);
        svfloat16_t codes_f16 = svld1_f16(pg, &codes_fp16[i]);
        
        // Process elements that fit in a single FP32 vector
        uint64_t remaining = dim - i;
        uint64_t vl_f32 = svcntw();
        uint64_t to_process = (remaining < vl_f32) ? remaining : vl_f32;
        
        svbool_t pg_f32 = svwhilelt_b32(uint64_t(0), to_process);
        svfloat32_t query_f32 = svcvt_f32_f16_x(pg_f32, query_f16);
        svfloat32_t codes_f32 = svcvt_f32_f16_x(pg_f32, codes_f16);
        
        svfloat32_t diff = svsub_f32_x(pg_f32, query_f32, codes_f32);
        sum_vec = svmla_f32_x(pg_f32, sum_vec, diff, diff);
    }
    
    return svaddv_f32(svptrue_b32(), sum_vec);
}

} // namespace sve

// ========== Scalar Implementation ==========
namespace scalar {
    float BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
        float result = 0.0f;
        auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
        auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
        for (uint64_t i = 0; i < dim; ++i) {
            result += BF16ToFloat(query_bf16[i]) * BF16ToFloat(codes_bf16[i]);
        }
        return result;
    }

    float BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
        float result = 0.0f;
        auto* query_bf16 = reinterpret_cast<const uint16_t*>(query);
        auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);
        for (uint64_t i = 0; i < dim; ++i) {
            auto val = BF16ToFloat(query_bf16[i]) - BF16ToFloat(codes_bf16[i]);
            result += val * val;
        }
        return result;
    }

    float FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
        float result = 0.0f;
        auto* query_fp16 = reinterpret_cast<const uint16_t*>(query);
        auto* codes_fp16 = reinterpret_cast<const uint16_t*>(codes);
        for (uint64_t i = 0; i < dim; ++i) {
            result += FP16ToFloat(query_fp16[i]) * FP16ToFloat(codes_fp16[i]);
        }
        return result;
    }

    float FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
        float result = 0.0f;
        auto* query_fp16 = reinterpret_cast<const uint16_t*>(query);
        auto* codes_fp16 = reinterpret_cast<const uint16_t*>(codes);
        for (uint64_t i = 0; i < dim; ++i) {
            auto val = FP16ToFloat(query_fp16[i]) - FP16ToFloat(codes_fp16[i]);
            result += val * val;
        }
        return result;
    }
}

// ========== Test Functions ==========

void generate_random_data(std::vector<uint16_t>& data, bool is_bf16 = true) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < data.size(); ++i) {
        float value = dis(gen);
        if (is_bf16) {
            data[i] = FloatToBF16(value);
        } else {
            data[i] = FloatToFP16(value);
        }
    }
}

template<typename Func>
double benchmark(Func func, const uint8_t* query, const uint8_t* codes, uint64_t dim, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    float result = 0;
    for (int i = 0; i < iterations; ++i) {
        result += func(query, codes, dim);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    if (result == -1.0f) {
        std::cout << "Never happens" << std::endl;
    }
    
    return duration.count() / 1000.0;
}

bool is_close(float a, float b, float rel_tol = 1e-5f, float abs_tol = 1e-8f) {
    return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
}

int main() {
    std::cout << "=== SVE BF16/FP16 Operations Test ===" << std::endl;
    
    // Check SVE support
    std::cout << "SVE vector length: " << svcntb() << " bytes" << std::endl;
    std::cout << "SVE FP32 elements per vector: " << svcntw() << std::endl;
    
    std::vector<uint64_t> test_dims = {16, 64, 128, 256, 512, 1024, 2048, 4096};
    
    for (auto dim : test_dims) {
        std::cout << "\n--- Testing dimension: " << dim << " ---" << std::endl;
        
        std::vector<uint16_t> query_bf16(dim);
        std::vector<uint16_t> codes_bf16(dim);
        std::vector<uint16_t> query_fp16(dim);
        std::vector<uint16_t> codes_fp16(dim);
        
        generate_random_data(query_bf16, true);
        generate_random_data(codes_bf16, true);
        generate_random_data(query_fp16, false);
        generate_random_data(codes_fp16, false);
        
        auto* query_bf16_ptr = reinterpret_cast<const uint8_t*>(query_bf16.data());
        auto* codes_bf16_ptr = reinterpret_cast<const uint8_t*>(codes_bf16.data());
        auto* query_fp16_ptr = reinterpret_cast<const uint8_t*>(query_fp16.data());
        auto* codes_fp16_ptr = reinterpret_cast<const uint8_t*>(codes_fp16.data());
        
        // Test BF16 Inner Product
        {
            float scalar_result = scalar::BF16ComputeIP(query_bf16_ptr, codes_bf16_ptr, dim);
            float sve_result = sve::BF16ComputeIP(query_bf16_ptr, codes_bf16_ptr, dim);
            
            std::cout << "BF16 Inner Product:" << std::endl;
            std::cout << "  Scalar result: " << scalar_result << std::endl;
            std::cout << "  SVE result: " << sve_result << std::endl;
            std::cout << "  Match: " << (is_close(scalar_result, sve_result) ? "YES" : "NO") << std::endl;
            
            if (dim >= 256) {
                int iterations = 10000;
                double scalar_time = benchmark(scalar::BF16ComputeIP, query_bf16_ptr, codes_bf16_ptr, dim, iterations);
                double sve_time = benchmark(sve::BF16ComputeIP, query_bf16_ptr, codes_bf16_ptr, dim, iterations);
                
                std::cout << "  Scalar time: " << scalar_time << " ms" << std::endl;
                std::cout << "  SVE time: " << sve_time << " ms" << std::endl;
                std::cout << "  Speedup: " << scalar_time / sve_time << "x" << std::endl;
            }
        }
        
        // Test BF16 L2 Distance
        {
            float scalar_result = scalar::BF16ComputeL2Sqr(query_bf16_ptr, codes_bf16_ptr, dim);
            float sve_result = sve::BF16ComputeL2Sqr(query_bf16_ptr, codes_bf16_ptr, dim);
            
            std::cout << "\nBF16 L2 Squared Distance:" << std::endl;
            std::cout << "  Scalar result: " << scalar_result << std::endl;
            std::cout << "  SVE result: " << sve_result << std::endl;
            std::cout << "  Match: " << (is_close(scalar_result, sve_result) ? "YES" : "NO") << std::endl;
            
            if (dim >= 256) {
                int iterations = 10000;
                double scalar_time = benchmark(scalar::BF16ComputeL2Sqr, query_bf16_ptr, codes_bf16_ptr, dim, iterations);
                double sve_time = benchmark(sve::BF16ComputeL2Sqr, query_bf16_ptr, codes_bf16_ptr, dim, iterations);
                
                std::cout << "  Scalar time: " << scalar_time << " ms" << std::endl;
                std::cout << "  SVE time: " << sve_time << " ms" << std::endl;
                std::cout << "  Speedup: " << scalar_time / sve_time << "x" << std::endl;
            }
        }
        
        // Test FP16 Inner Product
        {
            float scalar_result = scalar::FP16ComputeIP(query_fp16_ptr, codes_fp16_ptr, dim);
            float sve_result = sve::FP16ComputeIP(query_fp16_ptr, codes_fp16_ptr, dim);
            
            std::cout << "\nFP16 Inner Product:" << std::endl;
            std::cout << "  Scalar result: " << scalar_result << std::endl;
            std::cout << "  SVE result: " << sve_result << std::endl;
            std::cout << "  Match: " << (is_close(scalar_result, sve_result) ? "YES" : "NO") << std::endl;
            
            if (dim >= 256) {
                int iterations = 10000;
                double scalar_time = benchmark(scalar::FP16ComputeIP, query_fp16_ptr, codes_fp16_ptr, dim, iterations);
                double sve_time = benchmark(sve::FP16ComputeIP, query_fp16_ptr, codes_fp16_ptr, dim, iterations);
                
                std::cout << "  Scalar time: " << scalar_time << " ms" << std::endl;
                std::cout << "  SVE time: " << sve_time << " ms" << std::endl;
                std::cout << "  Speedup: " << scalar_time / sve_time << "x" << std::endl;
            }
        }
        
        // Test FP16 L2 Distance
        {
            float scalar_result = scalar::FP16ComputeL2Sqr(query_fp16_ptr, codes_fp16_ptr, dim);
            float sve_result = sve::FP16ComputeL2Sqr(query_fp16_ptr, codes_fp16_ptr, dim);
            
            std::cout << "\nFP16 L2 Squared Distance:" << std::endl;
            std::cout << "  Scalar result: " << scalar_result << std::endl;
            std::cout << "  SVE result: " << sve_result << std::endl;
            std::cout << "  Match: " << (is_close(scalar_result, sve_result) ? "YES" : "NO") << std::endl;
            
            if (dim >= 256) {
                int iterations = 10000;
                double scalar_time = benchmark(scalar::FP16ComputeL2Sqr, query_fp16_ptr, codes_fp16_ptr, dim, iterations);
                double sve_time = benchmark(sve::FP16ComputeL2Sqr, query_fp16_ptr, codes_fp16_ptr, dim, iterations);
                
                std::cout << "  Scalar time: " << scalar_time << " ms" << std::endl;
                std::cout << "  SVE time: " << sve_time << " ms" << std::endl;
                std::cout << "  Speedup: " << scalar_time / sve_time << "x" << std::endl;
            }
        }
    }
    
    // Special case tests
    std::cout << "\n=== Special Case Tests ===" << std::endl;
    
    // Test conversion functions
    {
        std::cout << "\nConversion function tests:" << std::endl;
        float test_values[] = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 3.14159f, -3.14159f};
        
        for (float val : test_values) {
            uint16_t bf16 = FloatToBF16(val);
            float recovered = BF16ToFloat(bf16);
            std::cout << "BF16: " << val << " -> 0x" << std::hex << bf16 << std::dec 
                      << " -> " << recovered << " (error: " << std::abs(val - recovered) << ")" << std::endl;
            
            uint16_t fp16 = FloatToFP16(val);
            recovered = FP16ToFloat(fp16);
            std::cout << "FP16: " << val << " -> 0x" << std::hex << fp16 << std::dec 
                      << " -> " << recovered << " (error: " << std::abs(val - recovered) << ")" << std::endl;
        }
    }
    
    // Boundary test
    {
        std::cout << "\nBoundary test (dim=1):" << std::endl;
        uint16_t single_query = FloatToBF16(2.0f);
        uint16_t single_code = FloatToBF16(3.0f);
        
        float result = sve::BF16ComputeIP(reinterpret_cast<const uint8_t*>(&single_query), 
                                          reinterpret_cast<const uint8_t*>(&single_code), 1);
        std::cout << "BF16 IP (2.0 * 3.0) = " << result << " (expected: ~6.0)" << std::endl;
    }
    
    std::cout << "\n=== Test completed ===" << std::endl;
    
    return 0;
}
