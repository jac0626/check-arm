#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath> // for std::abs

// 只有在定义了 __ARM_FEATURE_SVE2 宏时才包含 SVE 头文件和 SVE2 代码
#if defined(__ARM_FEATURE_SVE2)
#include <arm_sve.h>
#endif

/**
 * @brief 标量版本: result[i] = accumulator[i] + abs(vec1[i] - vec2[i])
 */
void saba_example_scalar(int8_t* accumulator, const int8_t* vec1, const int8_t* vec2, uint64_t size) {
    for (uint64_t i = 0; i < size; ++i) {
        // 注意：C++中，int8_t的减法可能会提升到int，需要小心处理
        int32_t diff = static_cast<int32_t>(vec1[i]) - static_cast<int32_t>(vec2[i]);
        // SVE指令遵循的是模运算，我们在这里模拟它
        accumulator[i] = static_cast<int8_t>(accumulator[i] + std::abs(diff));
    }
}

/**
 * @brief SVE2版本: 使用 svaba_s8
 */
void saba_example_sve2(int8_t* accumulator, const int8_t* vec1, const int8_t* vec2, uint64_t size) {
#if defined(__ARM_FEATURE_SVE2)
    uint64_t i = 0;
    uint64_t step = svcntb(); // 每次处理的8位元素数量

    while (i < size) {
        svbool_t pg = svwhilelt_b8(i, size);

        // 加载三个向量
        svint8_t acc_vec = svld1_s8(pg, accumulator + i);
        svint8_t v1_vec = svld1_s8(pg, vec1 + i);
        svint8_t v2_vec = svld1_s8(pg, vec2 + i);
        
        // 关键的 SVE2 指令：有符号绝对差值累加
        // svaba_s8(acc_vec, v1_vec, v2_vec)
        // - 计算 v1_vec 和 v2_vec 逐元素的差的绝对值。
        // - 将这个绝对差值累加到 acc_vec 对应的元素上。
        // - SABA = Signed Absolute aBsolute difference and Accumulate
        // - 注意：这里没有谓词版本(_m或_z)，因为它是一个三元操作，
        //   并且通常会使用 merging 行为，即非激活通道保持acc_vec的原值。
        //   为了代码清晰，我们使用谓词加载/存储来确保正确性。
        svint8_t result_vec = svaba_s8(acc_vec, v1_vec, v2_vec);

        // 将结果写回内存
        svst1_s8(pg, accumulator + i, result_vec);

        i += step;
    }
#else
    // 如果不支持SVE2，则调用标量版本作为后备
    saba_example_scalar(accumulator, vec1, vec2, size);
#endif
}

int main() {
    uint64_t vector_size = 13; // 使用一个小的、非向量长度整数倍的大小

    // 准备三份数据
    std::vector<int8_t> accumulator = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, -120};
    std::vector<int8_t> vec1        = { 5, -5, 15, 25, 35, 45, 55, 65, 75,  85,  95, 105, 127};
    std::vector<int8_t> vec2        = { 2,  2, 20, 20, 40, 40, 60, 60, 80,  80, 100, 100, -128};
    
    // 创建一个副本，用于标量计算以进行比较
    std::vector<int8_t> accumulator_scalar = accumulator;


    std::cout << "--- 原始数据 ---" << std::endl;
    std::cout << "累加器: ";
    for (int8_t val : accumulator) std::cout << static_cast<int>(val) << " ";
    std::cout << std::endl;


    // --- 调用 SVE2 版本 ---
    saba_example_sve2(accumulator.data(), vec1.data(), vec2.data(), vector_size);

    std::cout << "\n--- SVE2 处理后 ---" << std::endl;
    std::cout << "新累加器: ";
    for (int8_t val : accumulator) std::cout << static_cast<int>(val) << " ";
    std::cout << std::endl;


    // --- 调用标量版本以验证 ---
    saba_example_scalar(accumulator_scalar.data(), vec1.data(), vec2.data(), vector_size);
    
    std::cout << "\n--- 标量处理后 (用于验证) ---" << std::endl;
    std::cout << "新累加器: ";
    for (int8_t val : accumulator_scalar) std::cout << static_cast<int>(val) << " ";
    std::cout << std::endl;


    // --- 比较结果 ---
    if (accumulator == accumulator_scalar) {
        std::cout << "\n验证成功: SVE2版本结果与标量版本一致!" << std::endl;
    } else {
        std::cout << "\n验证失败: 结果不一致!" << std::endl;
        return 1;
    }

    return 0;
}
