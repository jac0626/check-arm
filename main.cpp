// sve2_widening_multiply_example.cpp
//
// 一个展示SVE2加宽乘加特性的示例程序。
// 它使用 svmlalt_u32 指令来高效地将两个uint16_t向量的乘积
// 累加到一个uint32_t的向量累加器中。

#include <iostream>
#include <vector>
#include <cstdint>
#include <numeric>

// 检查SVE宏的定义
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/**
 * @brief 标量版本：计算两个uint16_t向量的乘积累加和。
 * 
 * @param vec1 第一个向量的指针。
 * @param vec2 第二个向量的指针。
 * @param dim  向量的维度。
 * @return uint64_t 乘积的总和。
 */
uint64_t
multiply_accumulate_u16_scalar(const uint16_t* vec1, const uint16_t* vec2, uint64_t dim) {
    uint64_t total_sum = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        // 将两个u16相乘得到u32，然后累加到u64，防止总和溢出
        total_sum += static_cast<uint32_t>(vec1[i]) * static_cast<uint32_t>(vec2[i]);
    }
    return total_sum;
}


#if defined(__ARM_FEATURE_SVE2)
/**
 * @brief SVE2优化版本：使用svmlalt_u32计算乘积累加和。
 *
 * __attribute__((target("sve2"))) 确保为这个函数启用SVE2指令。
 */
__attribute__((target("sve2")))
uint64_t
multiply_accumulate_u16_sve2(const uint16_t* vec1, const uint16_t* vec2, uint64_t dim) {
    // 1. 初始化
    // 创建一个32位的向量累加器，所有通道都初始化为0。
    // 因为svmlalt会将16位乘积加宽到32位。
    svuint32_t sum_vec = svdup_u32(0);

    // 获取当前SVE向量长度可以容纳多少个16位元素。
    // svcnth() 返回SVE向量寄存器中的16位元素数量。
    const uint64_t step = svcnth();
    uint64_t i = 0;

    // 2. 主循环 (向量长度无关 VLA)
    // svwhilelt_b16会生成一个16位粒度的谓词（掩码）。
    svbool_t pg = svwhilelt_b16(i, dim);

    do {
        // 使用谓词pg进行安全加载。
        svuint16_t v1_u16 = svld1_u16(pg, vec1 + i);
        svuint16_t v2_u16 = svld1_u16(pg, vec2 + i);

        // 关键的SVE2指令: svmlalt_u32 (Unsigned Multiply-Long Accumulate Top)
        // svmlalt_u32_m(pg, sum_vec, v1_u16, v2_u16);
        // - 它取出v1_u16和v2_u16中的每个uint16_t元素。
        // - 将它们逐元素相乘，结果被“加宽(Long)”成一个uint32_t。
        // - 将这个32位的结果，累加(Accumulate)到sum_vec对应的32位通道中。
        // - "_m" 表示这个操作是受谓词pg控制的（masked）。
        //
        // 这个指令优雅地处理了从低精度到高精度的转换和计算。
        sum_vec = svmlalt_u32_m(pg, sum_vec, v1_u16, v2_u16);

        // 步进到下一个向量块
        i += step;
        pg = svwhilelt_b16(i, dim); // 更新谓词
    } while (svptest_any(svptrue_b16(), pg));

    // 3. 水平累加 (Reduction)
    // 此时，sum_vec的每个通道都包含了一部分和。我们需要将它们全部加起来。
    // 注意：svaddv的结果是32位的，但我们的总和可能超过32位，
    // 所以我们需要将32位的向量累加到一个64位的标量中。
    // 为此，我们可以使用svget32将向量的通道取出并累加。
    // 一个更简单的方法是使用svaddv将u32向量水平相加，然后将结果转换为u64。
    // 在这里我们使用svaddv。
    // svaddv在u32上的结果是一个u32, 我们需要将它转换成u64以匹配标量版本
    svuint64_t sum_vec_u64 = svunpklo_u64(sum_vec); // 将u32向量的低半部分解包成u64
    return svaddv_u64(svptrue_b64(), sum_vec_u64); // 水平累加u64向量
}
#endif


int main() {
    const uint64_t dim = 2055; // 同样使用一个非整数倍的维度
    std::vector<uint16_t> vec1(dim);
    std::vector<uint16_t> vec2(dim);

    // 填充一些测试数据
    for (uint64_t i = 0; i < dim; ++i) {
        vec1[i] = i % 1000;
        vec2[i] = (i * 3) % 1200;
    }

    std::cout << "向量维度: " << dim << std::endl;

    // --- 使用标量版本计算 ---
    uint64_t scalar_result = multiply_accumulate_u16_scalar(vec1.data(), vec2.data(), dim);
    std::cout << "标量版本结果: " << scalar_result << std::endl;

    // --- 使用SVE2版本计算 ---
#if defined(__ARM_FEATURE_SVE2)
    uint64_t sve2_result = multiply_accumulate_u16_sve2(vec1.data(), vec2.data(), dim);
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
