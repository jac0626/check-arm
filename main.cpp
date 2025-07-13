#include <iostream>

// 检查是否定义了 SVE 特性宏。
// 如果定义了，则包含 SVE 头文件。
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

int main() {
    // 再次使用宏来保护 SVE 相关的代码，
    // 这样即使在不支持 SVE 的编译器或架构上编译，程序也能正常运行（并给出提示）。
#if defined(__ARM_FEATURE_SVE)
    // svcntb() (count bytes) 是一个SVE内联函数，它返回SVE向量寄存器
    // 以字节为单位的长度。这是获取向量长度最直接的方式。
    // 返回值类型是 uint64_t。
    uint64_t sve_bytes = svcntb();

    // 向量长度（以位为单位）等于字节数乘以8。
    uint64_t sve_bits = sve_bytes * 8;

    std::cout << "SVE is enabled on this system." << std::endl;
    std::cout << "SVE vector length: " << sve_bits << " bits (" << sve_bytes << " bytes)." << std::endl;

    // 你也可以使用其他 `svcnt*` 系列函数来获取以不同元素大小为单位的长度，
    // 结果是一样的。例如，使用 `svcntw()` 获取32位（word）元素的数量：
    uint64_t num_floats = svcntw();
    std::cout << "This means it can hold " << num_floats << " 32-bit float/int elements." << std::endl;

#else
    // 如果 __ARM_FEATURE_SVE 宏未定义，说明当前编译环境不支持SVE。
    std::cout << "SVE is not enabled or not supported by the compiler/architecture." << std::endl;
#endif

    return 0;
}