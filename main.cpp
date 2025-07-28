#include <arm_sve.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <signal.h>
#include <setjmp.h>
#include <string.h>
#include <errno.h>

static sigjmp_buf jmpbuf;

void sigsegv_handler(int sig) {
    siglongjmp(jmpbuf, 1);
}

// 打印FFR的详细状态
void print_ffr_bits(svbool_t ffr, int vl) {
    printf("FFR位状态: ");
    for (int i = 0; i < vl; i++) {
        // 创建只有第i位为1的谓词
        svbool_t single_bit = svwhilelt_b32(i, i+1);
        svbool_t test = svand_z(svptrue_b32(), ffr, single_bit);
        
        if (svptest_any(svptrue_b32(), test)) {
            printf("1");
        } else {
            printf("0");
        }
    }
    printf("\n");
}

int main() {
    int vl = svcntw();
    printf("SVE向量长度: %d个32位元素\n", vl);
    
    size_t page_size = sysconf(_SC_PAGESIZE);
    printf("系统页面大小: %zu字节\n\n", page_size);
    
    // 分配3个页面，使用中间页面存放数据
    void *mem = mmap(NULL, page_size * 3, PROT_READ | PROT_WRITE, 
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed");
        return 1;
    }
    
    // 数据放在第二页的中间，确保会跨越到第三页
    float *data = (float*)((char*)mem + page_size + page_size/2);
    
    // 初始化数据（只初始化不会越界的部分）
    int safe_elements = (page_size - page_size/2) / sizeof(float);
    safe_elements = safe_elements < vl ? safe_elements : vl;
    
    printf("安全元素数量: %d\n", safe_elements);
    for (int i = 0; i < safe_elements; i++) {
        data[i] = i + 100.0f;
    }
    
    // 保护第三页
    if (mprotect((char*)mem + 2*page_size, page_size, PROT_NONE) != 0) {
        perror("mprotect failed");
        munmap(mem, page_size * 3);
        return 1;
    }
    
    // 验证保护
    printf("\n验证内存保护...\n");
    signal(SIGSEGV, sigsegv_handler);
    if (sigsetjmp(jmpbuf, 1) == 0) {
        volatile float test = *((float*)((char*)mem + 2*page_size));
        printf("错误：能够访问受保护的页面！\n");
        munmap(mem, page_size * 3);
        return 1;
    } else {
        printf("确认：受保护页面不可访问\n");
    }
    
    svbool_t all_true = svptrue_b32();
    
    // 测试1：首故障加载
    printf("\n=== 首故障加载 (svldff1_f32) ===\n");
    svsetffr();
    svbool_t initial_ffr = svrdffr();
    print_ffr_bits(initial_ffr, vl);
    
    svfloat32_t ff_result = svldff1_f32(all_true, data);
    svbool_t ff_ffr = svrdffr();
    
    printf("加载后的");
    print_ffr_bits(ff_ffr, vl);
    
    // 统计有效元素
    int ff_valid = 0;
    float ff_buf[vl];
    svst1_f32(all_true, ff_buf, ff_result);
    
    for (int i = 0; i < vl; i++) {
        svbool_t bit = svand_z(all_true, ff_ffr, svwhilelt_b32(i, i+1));
        if (svptest_any(all_true, bit)) {
            ff_valid++;
            printf("  元素[%d] = %.1f (FFR=1)\n", i, ff_buf[i]);
        } else {
            printf("  元素[%d] = ? (FFR=0)\n", i);
        }
    }
    printf("有效元素总数: %d\n", ff_valid);
    
    // 测试2：非故障加载
    printf("\n=== 非故障加载 (svldnf1_f32) ===\n");
    svsetffr();
    svfloat32_t nf_result = svldnf1_f32(all_true, data);
    svbool_t nf_ffr = svrdffr();
    
    printf("加载后的");
    print_ffr_bits(nf_ffr, vl);
    
    // 统计有效元素
    int nf_valid = 0;
    float nf_buf[vl];
    svst1_f32(all_true, nf_buf, nf_result);
    
    for (int i = 0; i < vl; i++) {
        svbool_t bit = svand_z(all_true, nf_ffr, svwhilelt_b32(i, i+1));
        if (svptest_any(all_true, bit)) {
            nf_valid++;
            printf("  元素[%d] = %.1f (FFR=1)\n", i, nf_buf[i]);
        } else {
            printf("  元素[%d] = ? (FFR=0)\n", i);
        }
    }
    printf("有效元素总数: %d\n", nf_valid);
    
    // 测试3：从完全非法地址开始的非故障加载
    printf("\n=== 从受保护页面开始的非故障加载 ===\n");
    svsetffr();
    float *protected_addr = (float*)((char*)mem + 2*page_size);
    svfloat32_t nf2_result = svldnf1_f32(all_true, protected_addr);
    svbool_t nf2_ffr = svrdffr();
    
    printf("加载后的");
    print_ffr_bits(nf2_ffr, vl);
    
    int nf2_valid = 0;
    for (int i = 0; i < vl; i++) {
        svbool_t bit = svand_z(all_true, nf2_ffr, svwhilelt_b32(i, i+1));
        if (svptest_any(all_true, bit)) {
            nf2_valid++;
        }
    }
    printf("有效元素总数: %d (预期: 0)\n", nf2_valid);
    
    munmap(mem, page_size * 3);
    return 0;
}
