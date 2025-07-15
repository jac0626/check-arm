// sve2_example.cpp
//
// 一个展示SVE2特性的示例程序。
// 它使用SVE2的整数点积指令 (svdot) 来高效计算两个int8_t向量的内积。

#include <iostream>
#include <vector>
#include <cstdint>
#include <numeric> // 用于 std::inner_product

// 检查是否定义了SVE宏，这是编写可移植代码的良好实践
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * @brief 标量版本：计算int8_t向量的内积。
 * 
 * 这是我们优化的基准，用于验证SVE2版本的正确性。
 * 
 * @param vec1 第一个向量的指针。
 * @param vec2 第二个向量的指针。
 * @param dim  向量的维度。
 * @return int32_t 两个向量的内积。
 */
int32_t
int8_inner_product_scalar(const int8_t* vec1, const int8_t* vec2, uint64_t dim) {
    int32_t result = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        // 将乘积累加到32位整数，以防止溢出
        result += static_cast<int32_t>(vec1[i]) * static_cast<int32_t>(vec2[i]);
    }
    return result;
}


#if defined(__ARM_FEATURE_SVE2)
/**
 * @brief SVE2优化版本：计算int8_t向量的内积。
 *
 * __attribute__((target("sve2"))) 是一个函数属性，它告诉编译器
 * 只为这一个函数启用SVE2指令集。这在混合指令集的项目中很有用。
 * 当然，更常见的方法是为整个文件添加 -march=armv8-a+sve2 编译标志。
 */
__attribute__((target("sve2")))
int32_t
int8_inner_product_sve2(const int8_t* vec1, const int8_t* vec2, uint64_t dim) {
    // 1. 初始化
    // 创建一个32位的向量累加器，所有通道都初始化为0。
    // 我们使用32位累加器是因为svdot指令会将8位乘积累加到32位通道中，以避免溢出。
    svint32_t sum_vec = svdup_s32(0);

    // 获取当前SVE向量长度可以容纳多少个8位元素。
    // svcntb() 返回SVE向量寄存器的字节数。
    const uint64_t step = svcntb();
    uint64_t i = 0;

    // 2. 主循环 (向量长度无关 VLA)
    // svwhilelt_b8会生成一个谓词（掩码），只为 i < dim 的通道激活。
    // 这优雅地处理了任意长度的向量，包括最后不足一个完整向量长度的“尾部”数据。
    svbool_t pg = svwhilelt_b8(i, dim);

    // svptest_any检查谓词中是否至少有一个激活的通道。
    // 只要还有数据需要处理，循环就会继续。
    do {
        // 使用谓词pg进行安全加载。对于无效的通道，不会发生内存访问。
        svint8_t v1_s8 = svld1_s8(pg, vec1 + i);
        svint8_t v2_s8 = svld1_s8(pg, vec2 + i);

        // 关键的SVE2指令: svdot (Signed Dot Product)
        // svdot_s32_m(pg, sum_vec, v1_s8, v2_s8)
        // - 它将v1_s8和v2_s8中每4个相邻的int8_t元素作为一个组。
        // - 在每个组内，它计算点积: (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3])
        // - 然后将这个32位的结果，累加到sum_vec对应的32位通道中。
        // - 这一切都在一条指令中完成，效率极高。
        sum_vec = svdot_s32_m(pg, sum_vec, v1_s8, v2_s8);

        // 步进到下一个向量块
        i += step;
        pg = svwhilelt_b8(i, dim); // 更新下一轮的谓词
    } while (svptest_any(svptrue_b8(), pg));

    // 3. 水平累加 (Reduction)
    // 循环结束后，sum_vec的每个32位通道都包含了一部分总和。
    // svaddv_s32将一个向量中所有通道的值相加，得到最终的标量结果。
    return svaddv_s32(svptrue_b32(), sum_vec);
}
#endif


int main() {
    // 创建两个向量，维度不是16或32的整数倍，以测试尾部处理逻辑。
    const uint64_t dim = 1003; 
    std::vector<int8_t> vec1(dim);
    std::vector<int8_t> vec2(dim);

    // 填充一些测试数据
    for (uint64_t i = 0; i < dim; ++i) {
        vec1[i] = (i % 50) - 25;
        vec2[i] = (i % 40) - 20;
    }

    std::cout << "向量维度: " << dim << std::endl;

    // --- 使用标量版本计算 ---
    int32_t scalar_result = int8_inner_product_scalar(vec1.data(), vec2.data(), dim);
    std::cout << "标量版本结果: " << scalar_result << std::endl;

    // --- 使用SVE2版本计算 ---
#if defined(__ARM_FEATURE_SVE2)
    int32_t sve2_result = int8_inner_product_sve2(vec1.data(), vec2.data(), dim);
    std::cout << "SVE2版本结果:  " << sve2_result << std::endl;

    // --- 验证结果 ---
    if (scalar_result == sve2_result) {
        std::cout << "\n验证成功: SVE2版本结果与标量版本一致!" << std::endl;
    } else {
        std::cout << "\n验证失败: 结果不一致!" << std::endl;
        return 1;
    }
#else
    std::cout << "\n未启用SVE2编译。跳过SVE2版本测试。" << std::endl;
#endif

    return 0;
}
