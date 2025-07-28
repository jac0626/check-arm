#include <arm_sve.h>
#include <stdio.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>

void test_ffr_granularity() {
    size_t page_size = 4096;
    
    // 分配两页内存
    void *mem = mmap(NULL, page_size * 2, PROT_READ | PROT_WRITE, 
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    // 初始化第一页
    memset(mem, 0xAA, page_size);
    
    // 保护第二页
    mprotect((char*)mem + page_size, page_size, PROT_NONE);
    
    printf("=== SVE FFR粒度测试 ===\n");
    printf("页边界地址: %p\n\n", (char*)mem + page_size);
    
    // 测试不同偏移量的32位加载
    for (int offset = 16; offset >0; offset -= 4) {
        printf("测试 %d: 从页边界前 %d 字节开始加载32位元素\n", 
               (16 - offset) / 4 + 1, offset);
        
        float *ptr = (float*)((char*)mem + page_size - offset);
        printf("  加载地址: %p\n", ptr);
        
        svsetffr();
        svfloat32_t result = svldff1_f32(svptrue_b32(), ptr);
        svbool_t ffr = svrdffr();
        uint32_t ffr_val = *(uint32_t*)&ffr;
        
        printf("  FFR: 0x%08x = ", ffr_val);
        for (int i = 15; i >= 0; i--) {
            printf("%d", (ffr_val >> i) & 1);
            if (i % 4 == 0 && i > 0) printf(" ");
        }
        
        // 计算有效位数
        int valid_bits = __builtin_popcount(ffr_val & 0xFFFF);
        printf("\n  有效位数: %d", valid_bits);
        
        // 分析
        if (offset > 0) {
            int expected_elements = offset / 4;  // 能完整加载的32位元素数
            int expected_bytes = offset;         // 可访问的字节数
            printf("\n  期望(按元素): %d 个1", expected_elements);
            printf("\n  期望(按字节): %d 个1", expected_bytes);
            
            if (valid_bits == expected_bytes && valid_bits != expected_elements) {
                printf("\n  >>> FFR似乎是按字节计算！");
            } else if (valid_bits == expected_elements) {
                printf("\n  >>> FFR是按元素计算");
            }
        }
        printf("\n\n");
    }
    
    // 测试8位加载作为对比
    printf("=== 8位加载对比测试 ===\n");
    for (int offset = 4; offset > 0; offset--) {
        printf("从页边界前 %d 字节开始加载8位元素\n", offset);
        
        uint8_t *ptr = (uint8_t*)((char*)mem + page_size - offset);
        
        svsetffr();
        svuint8_t result = svldff1_u8(svptrue_b8(), ptr);
        svbool_t ffr = svrdffr();
        uint32_t ffr_val = *(uint32_t*)&ffr;
        
        printf("  FFR: 0x%08x = ", ffr_val);
        for (int i = 7; i >= 0; i--) {
            printf("%d", (ffr_val >> i) & 1);
        }
        
        int valid_bits = __builtin_popcount(ffr_val & 0xFF);
        printf("\n  有效位数: %d (期望: %d)\n\n", valid_bits, offset);
    }
    
    // 测试不同大小元素的FFR行为
    printf("=== 不同元素大小的FFR测试 ===\n");
    printf("从页边界前8字节开始加载：\n");
    
    void *test_ptr = (char*)mem + page_size - 8;
    svbool_t ffr_temp;
    
    // 8位元素
    svsetffr();
    svldff1_u8(svptrue_b8(), (uint8_t*)test_ptr);
    ffr_temp = svrdffr();
    uint32_t ffr8_val = *(uint32_t*)&ffr_temp;
    printf("  8位元素 FFR: 0x%08x (%d个1)\n", ffr8_val, __builtin_popcount(ffr8_val));
    
    // 16位元素
    svsetffr();
    svldff1_u16(svptrue_b16(), (uint16_t*)test_ptr);
    ffr_temp = svrdffr();
    uint32_t ffr16_val = *(uint32_t*)&ffr_temp;
    printf("  16位元素 FFR: 0x%08x (%d个1)\n", ffr16_val, __builtin_popcount(ffr16_val));
    
    // 32位元素
    svsetffr();
    svldff1_f32(svptrue_b32(), (float*)test_ptr);
    ffr_temp = svrdffr();
    uint32_t ffr32_val = *(uint32_t*)&ffr_temp;
    printf("  32位元素 FFR: 0x%08x (%d个1)\n", ffr32_val, __builtin_popcount(ffr32_val));
    
    // 64位元素
    svsetffr();
    svldff1_f64(svptrue_b64(), (double*)test_ptr);
    ffr_temp = svrdffr();
    uint32_t ffr64_val = *(uint32_t*)&ffr_temp;
    printf("  64位元素 FFR: 0x%08x (%d个1)\n", ffr64_val, __builtin_popcount(ffr64_val));
    
    munmap(mem, page_size * 2);
}

int main() {
    printf("SVE向量长度: %ld 字节\n", svcntb());
    printf("32位元素数: %ld\n", svcntw());
    printf("64位元素数: %ld\n\n", svcntd());
    
    test_ffr_granularity();
    
    return 0;
}
