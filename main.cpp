#include <arm_sve.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <sys/mman.h>
#include <stdexcept>
#include <iomanip>

class SVEMemoryTester {
private:
    static constexpr size_t PAGE_SIZE = 4096;
    void* mem_base;
    size_t total_size;
    const int vector_length;
    
public:
    SVEMemoryTester() : mem_base(nullptr), total_size(PAGE_SIZE * 2), 
                        vector_length(svcntw()) {
        // 分配内存
        mem_base = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        
        if (mem_base == MAP_FAILED) {
            throw std::runtime_error("内存映射失败");
        }
    }
    
    ~SVEMemoryTester() {
        if (mem_base && mem_base != MAP_FAILED) {
            munmap(mem_base, total_size);
        }
    }
    
    // 禁用拷贝构造和赋值操作
    SVEMemoryTester(const SVEMemoryTester&) = delete;
    SVEMemoryTester& operator=(const SVEMemoryTester&) = delete;
    
    void run_test() {
        std::cout << "SVE向量长度: " << vector_length << "个32位元素\n\n";
        
        // 准备测试数据
        prepare_test_data();
        
        // 执行测试
        test_first_faulting_load();
        std::cout << std::endl;
        test_non_faulting_load();
        
        std::cout << "\n关键区别：两种指令都能处理越界访问，但首故障加载\n";
        std::cout << "在第一个元素无效时会触发真正的故障。\n";
    }
    
private:
    void prepare_test_data() {
        const size_t data_size = sizeof(float) * vector_length;
        
        // 将数据放在第一页的末尾，这样向量加载会跨越页边界
        float* data = reinterpret_cast<float*>(
            static_cast<char*>(mem_base) + PAGE_SIZE - data_size/2
        );
        
        // 初始化可访问的部分
        for (int i = 0; i < vector_length/2; ++i) {
            data[i] = static_cast<float>(i + 10.0f);
        }
        
        // 使第二页不可访问
        if (mprotect(static_cast<char*>(mem_base) + PAGE_SIZE, 
                     PAGE_SIZE, PROT_NONE) != 0) {
            throw std::runtime_error("设置内存保护失败");
        }
        
    }
    
    void test_first_faulting_load() {
        std::cout << "=== 首故障加载测试 ===\n";
        
        const float* data = reinterpret_cast<float*>(
            static_cast<char*>(mem_base) + PAGE_SIZE - 
            sizeof(float) * vector_length / 2
        );
        
        svbool_t all_true = svptrue_b32();
        
        // 设置首故障寄存器
        svsetffr();
        
        // 执行首故障加载
        svfloat32_t result = svldff1_f32(all_true, data);
        svbool_t ffr_status = svrdffr();
        
        std::cout << "加载完成，检查结果：\n";
        print_results(result, ffr_status);
    }
    
    void test_non_faulting_load() {
        std::cout << "=== 非故障加载测试 ===\n";
        
        const float* data = reinterpret_cast<float*>(
            static_cast<char*>(mem_base) + PAGE_SIZE - 
            sizeof(float) * vector_length / 2
        );
        
        svbool_t all_true = svptrue_b32();
        
        // 设置首故障寄存器
        svsetffr();
        
        // 执行非故障加载
        svfloat32_t result = svldnf1_f32(all_true, data);
        svbool_t ffr_status = svrdffr();
        
        std::cout << "加载完成，检查结果：\n";
        print_results(result, ffr_status);
    }
    
    void print_results(svfloat32_t result, svbool_t ffr_status) {
        svbool_t all_true = svptrue_b32();
        std::vector<float> buffer(vector_length);
        
        // 存储结果到缓冲区
        svst1_f32(all_true, buffer.data(), result);
        
        // 打印每个元素的状态
        for (int i = 0; i < vector_length; ++i) {
            svbool_t element_mask = svwhilelt_b32(i, i + 1);
            svbool_t valid_mask = svand_z(all_true, ffr_status, element_mask);
            
            if (svptest_any(all_true, valid_mask)) {
                std::cout << "  元素[" << std::setw(2) << i << "]: " 
                         << std::fixed << std::setprecision(1) 
                         << buffer[i] << " (有效)\n";
            } else {
                std::cout << "  元素[" << std::setw(2) << i << "]: ? (无效)\n";
            }
        }
    }
};

int main() {
    try {
        SVEMemoryTester tester;
        tester.run_test();
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
