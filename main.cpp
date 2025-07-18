#include <arm_sve.h>
#include <vector>
#include <iostream>
#include <iomanip> // For std::setw

void gather_load_robust_example() {
    // 设定一个不“整齐”的维度，以测试边界情况
    const size_t dim = 13;

    // 创建源数据
    std::vector<float> array(dim);
    for (size_t i = 0; i < dim; ++i) {
        array[i] = i + 1.0f;
    }
    const float* base_ptr = array.data();

    // 获取矢量长度
    uint64_t vl = svcntw();
    if (vl == 0) { return; }

    std::cout << "Vector Length (32-bit floats): " << vl << std::endl;
    std::cout << "Source data dimension (dim): " << dim << std::endl;

    // 一个“管辖”谓词，通常是全true
    svbool_t pg_all = svptrue_b32();

    // 1. 创建基础偏移量矢量 [0, 2, 4, 6, ...]
    svuint32_t z_base_indices = svindex_u32(0, 1);
    svuint32_t z_offsets = svmul_n_u32_x(pg_all, z_base_indices, 2);

    // 2. --- 关键的安全检查 ---
    //    创建一个谓词，仅当 z_offsets[i] < dim 时，通道i才为true。
    svbool_t pg_safe = svcmplt_n_u32(pg_all, z_offsets, dim);

    // 3. 使用这个“安全”的谓词执行收集加载
    svfloat32_t z_result = svld1_gather_u32offset_f32(pg_safe, base_ptr, z_offsets);

    // --- 验证结果 ---
    std::vector<uint32_t> offsets_vec(vl);
    svst1_u32(pg_all, offsets_vec.data(), z_offsets);

    std::vector<float> result_vec(vl);
    svst1_f32(pg_all, result_vec.data(), z_result);

    std::cout << "\n--- Operation Details ---\n";
    std::cout << std::left;
    std::cout << std::setw(8) << "Lane" 
              << std::setw(12) << "Offset" 
              << std::setw(12) << "Offset<dim"
              << std::setw(12) << "Result" << std::endl;
    std::cout << "------------------------------------------\n";

    for (uint64_t i = 0; i < vl; ++i) {
        std::cout << std::setw(8) << i
                  << std::setw(12) << offsets_vec[i]
                  << std::setw(12) << (offsets_vec[i] < dim ? "true" : "false")
                  << std::setw(12) << result_vec[i] << std::endl;
    }
}
int main(){
    gather_load_robust_example();
}
