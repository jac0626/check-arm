#include <iostream>
#include <vector>
#include <numeric>
#include <cstring>
#include <iomanip>
#include <arm_sve.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <signal.h>
#include <setjmp.h>
#include <random>
#include <thread>
#include <atomic>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif  

// For SIGSEGV handling in demo
static jmp_buf jmp_env;
static void sigsegv_handler(int sig) {
    longjmp(jmp_env, 1);
}
// --- Helper functions to print vector contents ---
void print_separator(const std::string& title = "") {
    if (!title.empty()) {
        std::cout << "\n━━━ " << title << " ━━━" << std::endl;
    } else {
        std::cout << "────────────────────────────────────────" << std::endl;
    }
}

void print_array_s8(const std::string& label, const int8_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << (int)data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_u8(const std::string& label, const uint8_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << (unsigned)data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_s16(const std::string& label, const int16_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_u16(const std::string& label, const uint16_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_s32(const std::string& label, const int32_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_u32(const std::string& label, const uint32_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_s64(const std::string& label, const int64_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_u64(const std::string& label, const uint64_t* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_f32(const std::string& label, const float* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << std::fixed << std::setprecision(1) << data[i];
    }
    std::cout << "]" << std::endl;
}

void print_array_f64(const std::string& label, const double* data, size_t count) {
    std::cout << label << ": [";
    for(size_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        if (i >= 16 && count > 20) {
            std::cout << "...";
            break;
        }
        std::cout << std::fixed << std::setprecision(1) << data[i];
    }
    std::cout << "]" << std::endl;
}

// Vector print functions
void print_s8(const std::string& label, svint8_t vec, uint64_t count) {
    std::vector<int8_t> values(count);
    svst1_s8(svptrue_b8(), values.data(), vec);
    print_array_s8(label, values.data(), count);
}

void print_u8(const std::string& label, svuint8_t vec, uint64_t count) {
    std::vector<uint8_t> values(count);
    svst1_u8(svptrue_b8(), values.data(), vec);
    print_array_u8(label, values.data(), count);
}

void print_s16(const std::string& label, svint16_t vec, uint64_t count) {
    std::vector<int16_t> values(count);
    svst1_s16(svptrue_b16(), values.data(), vec);
    print_array_s16(label, values.data(), count);
}

void print_u16(const std::string& label, svuint16_t vec, uint64_t count) {
    std::vector<uint16_t> values(count);
    svst1_u16(svptrue_b16(), values.data(), vec);
    print_array_u16(label, values.data(), count);
}

void print_s32(const std::string& label, svint32_t vec, uint64_t count) {
    std::vector<int32_t> values(count);
    svst1_s32(svptrue_b32(), values.data(), vec);
    print_array_s32(label, values.data(), count);
}

void print_u32(const std::string& label, svuint32_t vec, uint64_t count) {
    std::vector<uint32_t> values(count);
    svst1_u32(svptrue_b32(), values.data(), vec);
    print_array_u32(label, values.data(), count);
}

void print_s64(const std::string& label, svint64_t vec, uint64_t count) {
    std::vector<int64_t> values(count);
    svst1_s64(svptrue_b64(), values.data(), vec);
    print_array_s64(label, values.data(), count);
}

void print_u64(const std::string& label, svuint64_t vec, uint64_t count) {
    std::vector<uint64_t> values(count);
    svst1_u64(svptrue_b64(), values.data(), vec);
    print_array_u64(label, values.data(), count);
}

void print_f32(const std::string& label, svfloat32_t vec, uint64_t count) {
    std::vector<float> values(count);
    svst1_f32(svptrue_b32(), values.data(), vec);
    print_array_f32(label, values.data(), count);
}

void print_f64(const std::string& label, svfloat64_t vec, uint64_t count) {
    std::vector<double> values(count);
    svst1_f64(svptrue_b64(), values.data(), vec);
    print_array_f64(label, values.data(), count);
}

void print_bool(const std::string& label, svbool_t pg, uint64_t count) {
    std::cout << label << ": [";
    for(uint64_t i = 0; i < count; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << (svptest_any(svptrue_b8(), svand_z(svptrue_b8(), pg, svwhilelt_b8(i, i+1))) ? "1" : "0");
    }
    std::cout << "]" << std::endl;
}

void print_predicate_info(svbool_t pg, uint64_t count, const std::string& desc) {
    std::cout << "Predicate (" << desc << "): ";
    print_bool("", pg, count);
}

// --- Category 1: Basic Contiguous/Interleaved Loads ---
void demo_contiguous_interleaved_loads() {
    std::cout << "\n=== Category 1: Contiguous & Interleaved Loads ===" << std::endl;
    uint64_t count = svcntw();
    
    // 1.1 Standard contiguous load
    print_separator("1.1 Standard Contiguous Load");
    std::vector<int32_t> contiguous_data(count);
    std::iota(contiguous_data.begin(), contiguous_data.end(), 100);
    
    std::cout << "Input parameters:" << std::endl;
    svbool_t pg = svptrue_b32();
    print_predicate_info(pg, count, "all true");
    print_array_s32("Source memory", contiguous_data.data(), count);
    
    std::cout << "\nOperation: svld1_s32(pg, ptr)" << std::endl;
    svint32_t vec1 = svld1_s32(pg, contiguous_data.data());
    
    std::cout << "\nResult:" << std::endl;
    print_s32("Loaded vector", vec1, count);

    // 1.2 Vnum offset loads
    print_separator("1.2 Vector-Length Offset (vnum) Loads");
    std::vector<int32_t> vnum_data(count * 4);
    std::iota(vnum_data.begin(), vnum_data.end(), 200);
    
    std::cout << "Input parameters:" << std::endl;
    print_predicate_info(pg, count, "all true");
    print_array_s32("Source memory", vnum_data.data(), vnum_data.size());
    std::cout << "Base address: " << vnum_data.data() << std::endl;
    std::cout << "Vector length in elements: " << count << std::endl;
    
    std::cout << "\nOperations:" << std::endl;
    svint32_t vnum0 = svld1_vnum_s32(pg, vnum_data.data(), 0);
    svint32_t vnum1 = svld1_vnum_s32(pg, vnum_data.data(), 1);
    svint32_t vnum2 = svld1_vnum_s32(pg, vnum_data.data(), 2);
    
    std::cout << "svld1_vnum_s32(pg, ptr, 0) - loads from ptr + 0*VL" << std::endl;
    std::cout << "svld1_vnum_s32(pg, ptr, 1) - loads from ptr + 1*VL" << std::endl;
    std::cout << "svld1_vnum_s32(pg, ptr, 2) - loads from ptr + 2*VL" << std::endl;
    
    std::cout << "\nResults:" << std::endl;
    print_s32("vnum=0 result", vnum0, count);
    print_s32("vnum=1 result", vnum1, count);
    print_s32("vnum=2 result", vnum2, count);

    // 1.3 Interleaved loads (2-way)
    print_separator("1.3 Interleaved Loads (2-way)");
    std::vector<int32_t> interleaved2_data(count * 2);
    for(uint64_t i = 0; i < count; ++i) {
        interleaved2_data[i*2] = 300 + i;     // Channel 0
        interleaved2_data[i*2+1] = 400 + i;   // Channel 1
    }
    
    std::cout << "Input parameters:" << std::endl;
    print_predicate_info(pg, count, "all true");
    std::cout << "Source memory (interleaved format):" << std::endl;
    print_array_s32("  Memory", interleaved2_data.data(), interleaved2_data.size());
    std::cout << "  Format: [ch0[0], ch1[0], ch0[1], ch1[1], ...]" << std::endl;
    
    std::cout << "\nOperation: svld2_s32(pg, ptr) - de-interleaves 2 channels" << std::endl;
    svint32x2_t vecs2 = svld2_s32(pg, interleaved2_data.data());
    
    std::cout << "\nResults:" << std::endl;
    print_s32("Channel 0", svget2_s32(vecs2, 0), count);
    print_s32("Channel 1", svget2_s32(vecs2, 1), count);

    // 1.4 Interleaved loads (3-way)
    print_separator("1.4 Interleaved Loads (3-way)");
    std::vector<int32_t> interleaved3_data(count * 3);
    for(uint64_t i = 0; i < count; ++i) {
        interleaved3_data[i*3] = 500 + i;     // R
        interleaved3_data[i*3+1] = 600 + i;   // G
        interleaved3_data[i*3+2] = 700 + i;   // B
    }
    
    std::cout << "Input parameters:" << std::endl;
    print_predicate_info(pg, count, "all true");
    std::cout << "Source memory (RGB format):" << std::endl;
    print_array_s32("  Memory", interleaved3_data.data(), std::min(interleaved3_data.size(), size_t(24)));
    std::cout << "  Format: [R[0], G[0], B[0], R[1], G[1], B[1], ...]" << std::endl;
    
    std::cout << "\nOperation: svld3_s32(pg, ptr) - de-interleaves RGB channels" << std::endl;
    svint32x3_t vecs3 = svld3_s32(pg, interleaved3_data.data());
    
    std::cout << "\nResults:" << std::endl;
    print_s32("R channel", svget3_s32(vecs3, 0), count);
    print_s32("G channel", svget3_s32(vecs3, 1), count);
    print_s32("B channel", svget3_s32(vecs3, 2), count);

    // 1.5 Interleaved loads (4-way)
    print_separator("1.5 Interleaved Loads (4-way)");
    std::vector<int32_t> interleaved4_data(count * 4);
    for(uint64_t i = 0; i < count; ++i) {
        interleaved4_data[i*4] = 800 + i;     // R
        interleaved4_data[i*4+1] = 900 + i;   // G
        interleaved4_data[i*4+2] = 1000 + i;  // B
        interleaved4_data[i*4+3] = 1100 + i;  // A
    }
    
    std::cout << "Input parameters:" << std::endl;
    print_predicate_info(pg, count, "all true");
    std::cout << "Source memory (RGBA format):" << std::endl;
    print_array_s32("  Memory", interleaved4_data.data(), std::min(interleaved4_data.size(), size_t(32)));
    std::cout << "  Format: [R[0], G[0], B[0], A[0], R[1], G[1], B[1], A[1], ...]" << std::endl;
    
    std::cout << "\nOperation: svld4_s32(pg, ptr) - de-interleaves RGBA channels" << std::endl;
    svint32x4_t vecs4 = svld4_s32(pg, interleaved4_data.data());
    
    std::cout << "\nResults:" << std::endl;
    print_s32("R channel", svget4_s32(vecs4, 0), count);
    print_s32("G channel", svget4_s32(vecs4, 1), count);
    print_s32("B channel", svget4_s32(vecs4, 2), count);
    print_s32("A channel", svget4_s32(vecs4, 3), count);
}

// --- Category 2: Loads with Data Conversion ---
void demo_load_with_conversion() {
    std::cout << "\n=== Category 2: Loads with Data Conversion (Load-and-Extend) ===" << std::endl;
    
    // 2.1 8-bit to 32-bit conversions
    print_separator("2.1 8-bit to 32-bit Conversions");
    uint64_t count32 = svcntw();
    std::vector<int8_t> signed_bytes(count32);
    std::vector<uint8_t> unsigned_bytes(count32);
    std::iota(signed_bytes.begin(), signed_bytes.end(), -5);
    std::iota(unsigned_bytes.begin(), unsigned_bytes.end(), 250);

    svbool_t pg32 = svptrue_b32();
    
    std::cout << "Input parameters:" << std::endl;
    print_predicate_info(pg32, count32, "all true");
    print_array_s8("Signed bytes", signed_bytes.data(), signed_bytes.size());
    print_array_u8("Unsigned bytes", unsigned_bytes.data(), unsigned_bytes.size());
    
    std::cout << "\nOperations:" << std::endl;
    std::cout << "1. svld1sb_s32: Load signed bytes, sign-extend to 32-bit" << std::endl;
    svint32_t vec_sb_s32 = svld1sb_s32(pg32, signed_bytes.data());
    print_s32("  Result", vec_sb_s32, count32);
    
    std::cout << "\n2. svld1ub_u32: Load unsigned bytes, zero-extend to 32-bit" << std::endl;
    svuint32_t vec_ub_u32 = svld1ub_u32(pg32, unsigned_bytes.data());
    print_u32("  Result", vec_ub_u32, count32);
    
    std::cout << "\n3. svld1sb_s32 on unsigned data (reinterpret as signed)" << std::endl;
    svint32_t vec_sb_s32_2 = svld1sb_s32(pg32, (int8_t*)unsigned_bytes.data());
    print_s32("  Result", vec_sb_s32_2, count32);
    
    std::cout << "\n4. svld1ub_u32 on signed data (reinterpret as unsigned)" << std::endl;
    svuint32_t vec_ub_u32_2 = svld1ub_u32(pg32, (uint8_t*)signed_bytes.data());
    print_u32("  Result", vec_ub_u32_2, count32);

    // 2.2 16-bit to 32-bit conversions
    print_separator("2.2 16-bit to 32-bit Conversions");
    std::vector<int16_t> signed_shorts(count32);
    std::vector<uint16_t> unsigned_shorts(count32);
    std::iota(signed_shorts.begin(), signed_shorts.end(), -10);
    std::iota(unsigned_shorts.begin(), unsigned_shorts.end(), 65530);

    std::cout << "Input parameters:" << std::endl;
    print_array_s16("Signed shorts", signed_shorts.data(), signed_shorts.size());
    print_array_u16("Unsigned shorts", unsigned_shorts.data(), unsigned_shorts.size());
    
    std::cout << "\nOperations:" << std::endl;
    std::cout << "1. svld1sh_s32: Load signed shorts, sign-extend to 32-bit" << std::endl;
    svint32_t vec_sh_s32 = svld1sh_s32(pg32, signed_shorts.data());
    print_s32("  Result", vec_sh_s32, count32);
    
    std::cout << "\n2. svld1uh_u32: Load unsigned shorts, zero-extend to 32-bit" << std::endl;
    svuint32_t vec_uh_u32 = svld1uh_u32(pg32, unsigned_shorts.data());
    print_u32("  Result", vec_uh_u32, count32);

    // 2.3 8-bit to 64-bit conversions
    print_separator("2.3 8-bit to 64-bit Conversions");
    uint64_t count64 = svcntd();
    svbool_t pg64 = svptrue_b64();
    
    std::cout << "Input parameters:" << std::endl;
    print_predicate_info(pg64, count64, "all true");
    std::cout << "Using same byte arrays as before (first " << count64 << " elements)" << std::endl;
    
    std::cout << "\nOperations:" << std::endl;
    std::cout << "1. svld1sb_s64: Load signed bytes, sign-extend to 64-bit" << std::endl;
    svint64_t vec_sb_s64 = svld1sb_s64(pg64, signed_bytes.data());
    print_s64("  Result", vec_sb_s64, count64);
    
    std::cout << "\n2. svld1ub_u64: Load unsigned bytes, zero-extend to 64-bit" << std::endl;
    svuint64_t vec_ub_u64 = svld1ub_u64(pg64, unsigned_bytes.data());
    print_u64("  Result", vec_ub_u64, count64);

    // 2.6 Data conversion with vnum
    print_separator("2.6 Data Conversion with vnum");
    std::vector<int16_t> vnum_shorts(count32 * 3);
    std::iota(vnum_shorts.begin(), vnum_shorts.end(), 2000);
    
    std::cout << "Input parameters:" << std::endl;
    print_array_s16("Source array", vnum_shorts.data(), std::min(vnum_shorts.size(), size_t(48)));
    std::cout << "Vector length for 32-bit: " << count32 << std::endl;
    
    std::cout << "\nOperations:" << std::endl;
    std::cout << "svld1sh_vnum_s32(pg, ptr, vnum) - loads 16-bit values from ptr+vnum*VL/2" << std::endl;
    
    svint32_t vnum_sh0 = svld1sh_vnum_s32(pg32, vnum_shorts.data(), 0);
    svint32_t vnum_sh1 = svld1sh_vnum_s32(pg32, vnum_shorts.data(), 1);
    
    std::cout << "\nResults:" << std::endl;
    print_s32("vnum=0 (offset 0)", vnum_sh0, count32);
    print_s32("vnum=1 (offset VL/2)", vnum_sh1, count32);
}

// --- Category 3: Gather Loads ---
void demo_gather_loads() {
    std::cout << "\n=== Category 3: Gather Loads (Non-Contiguous Access) ===" << std::endl;
    uint64_t count32 = svcntw();
    uint64_t count64 = svcntd();
    
    // Prepare source data
    std::vector<int32_t> data_source(count32 * 4);
    std::iota(data_source.begin(), data_source.end(), 1000);
    
    std::vector<float> float_source(count32 * 4);
    for(size_t i = 0; i < float_source.size(); ++i) {
        float_source[i] = 100.0f + i * 0.5f;
    }

    // 3.1 Offset-based gather (32-bit)
    print_separator("3.1 Gather with 32-bit Byte Offsets");
    std::vector<uint32_t> offsets32(count32);
    for(uint64_t i = 0; i < count32; ++i) {
        offsets32[i] = i * 8; // Byte offsets: 0, 8, 16, ...
    }
    
    svbool_t pg32 = svptrue_b32();
    svuint32_t offsets32_vec = svld1_u32(pg32, offsets32.data());
    
    std::cout << "Input parameters:" << std::endl;
    print_predicate_info(pg32, count32, "all true");
    print_array_s32("Source array", data_source.data(), std::min(data_source.size(), size_t(32)));
    print_u32("Byte offsets", offsets32_vec, count32);
    std::cout << "Base address: " << data_source.data() << std::endl;
    
    std::cout << "\nOperation: svld1_gather_u32offset_s32(pg, base, offsets)" << std::endl;
    std::cout << "Loads: base[offsets[i]/4] for each i" << std::endl;
    
    svint32_t gather_offset32 = svld1_gather_u32offset_s32(pg32, data_source.data(), offsets32_vec);
    
    std::cout << "\nResult:" << std::endl;
    print_s32("Gathered values", gather_offset32, count32);

    // 3.3 Index-based gather (32-bit)
    print_separator("3.3 Gather with 32-bit Element Indices");
    std::vector<uint32_t> indices32(count32);
    for(uint64_t i = 0; i < count32; ++i) {
        indices32[i] = i * 2; // Element indices: 0, 2, 4, ...
    }
    
    svuint32_t indices32_vec = svld1_u32(pg32, indices32.data());
    
    std::cout << "Input parameters:" << std::endl;
    print_array_s32("Source array", data_source.data(), std::min(data_source.size(), size_t(32)));
    print_u32("Element indices", indices32_vec, count32);
    
    std::cout << "\nOperation: svld1_gather_u32index_s32(pg, base, indices)" << std::endl;
    std::cout << "Loads: base[indices[i]] for each i" << std::endl;
    
    svint32_t gather_index32 = svld1_gather_u32index_s32(pg32, data_source.data(), indices32_vec);
    
    std::cout << "\nResult:" << std::endl;
    print_s32("Gathered values", gather_index32, count32);

    // 3.5 Base-address gather (64-bit addresses)
    print_separator("3.5 Gather with 64-bit Base Addresses");
    std::vector<int64_t> data64_source(count64 * 4);
    std::iota(data64_source.begin(), data64_source.end(), 5000);
    
    std::vector<uint64_t> addresses64(count64);
    for(uint64_t i = 0; i < count64; ++i) {
        addresses64[i] = (uint64_t)(uintptr_t)&data64_source[i * 2];
    }
    
    svbool_t pg64 = svptrue_b64();
    svuint64_t addresses64_vec = svld1_u64(pg64, addresses64.data());
    
    std::cout << "Input parameters:" << std::endl;
    print_array_s64("Source array", data64_source.data(), std::min(data64_source.size(), size_t(16)));
    std::cout << "Address vector (pointing to elements 0, 2, 4, ...):" << std::endl;
    print_u64("Addresses", addresses64_vec, count64);
    
    std::cout << "\nOperation: svld1_gather_u64base_s64(pg, addresses)" << std::endl;
    std::cout << "Loads: *(int64_t*)addresses[i] for each i" << std::endl;
    
    svint64_t gather_base64 = svld1_gather_u64base_s64(pg64, addresses64_vec);
    
    std::cout << "\nResult:" << std::endl;
    print_s64("Gathered values", gather_base64, count64);

    // 3.7 Gather with data conversion (8-bit to 32-bit)
    print_separator("3.7 Gather with Data Conversion (8→32)");
    std::vector<int8_t> byte_source(count32 * 4);
    std::iota(byte_source.begin(), byte_source.end(), -20);
    
    std::cout << "Input parameters:" << std::endl;
    print_array_s8("Source bytes", byte_source.data(), std::min(byte_source.size(), size_t(32)));
    print_u32("Byte offsets", offsets32_vec, count32);
    
    std::cout << "\nOperations:" << std::endl;
    std::cout << "1. svld1sb_gather_u32offset_s32: Gather bytes, sign-extend to 32-bit" << std::endl;
    svint32_t gather_sb = svld1sb_gather_u32offset_s32(pg32, byte_source.data(), offsets32_vec);
    print_s32("  Result", gather_sb, count32);
    
    std::cout << "\n2. svld1ub_gather_u32offset_u32: Gather bytes, zero-extend to 32-bit" << std::endl;
    svuint32_t gather_ub = svld1ub_gather_u32offset_u32(pg32, (uint8_t*)byte_source.data(), offsets32_vec);
    print_u32("  Result", gather_ub, count32);
}

// --- Category 4: Load and Broadcast ---
void demo_load_and_broadcast() {
    std::cout << "\n=== Category 4: Load and Broadcast ===" << std::endl;
    
    // 4.1 Load and replicate 128-bit block
    print_separator("4.1 Load and Replicate 128-bit (svld1rq)");
    
    float pattern_128[4] = {1.1f, 2.2f, 3.3f, 4.4f};
    int32_t pattern_128_i[4] = {10, 20, 30, 40};
    
    std::cout << "Input parameters:" << std::endl;
    print_array_f32("128-bit float pattern", pattern_128, 4);
    print_array_s32("128-bit int pattern", pattern_128_i, 4);
    
    std::cout << "\nOperation: svld1rq_* - Loads 128 bits and replicates across vector" << std::endl;
    
    svfloat32_t broadcast_128 = svld1rq_f32(svptrue_b32(), pattern_128);
    svint32_t broadcast_128_i = svld1rq_s32(svptrue_b32(), pattern_128_i);
    
    std::cout << "\nResults:" << std::endl;
    print_f32("Replicated floats", broadcast_128, svcntw());
    print_s32("Replicated ints", broadcast_128_i, svcntw());

    // 4.2 Load and replicate 256-bit block (if supported)
    print_separator("4.2 Load and Replicate 256-bit (svld1ro)");
    
    float pattern_256[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    
    std::cout << "Input parameters:" << std::endl;
    print_array_f32("256-bit pattern", pattern_256, 8);
    
    std::cout << "\nOperation: svld1ro_f32 - Loads 256 bits and replicates" << std::endl;
    
    svfloat32_t broadcast_256 = svld1ro_f32(svptrue_b32(), pattern_256);
    
    std::cout << "\nResult:" << std::endl;
    print_f32("Replicated pattern", broadcast_256, svcntw());
   
}

// --- Category 5: Fault-Tolerant Loads ---

void demo_strlen_performance() {
    std::cout <<"Test 3: First-Faulting Loads (FF) - Vectorized strlen Performance"<<std::endl;
    
    // SVE strlen implementation
    auto sve_strlen_ff = [](const char* str) -> size_t {
        size_t len = 0;
        svbool_t pg = svptrue_b8();
        
        svsetffr();  // Initialize FFR once
        
        while (true) {
            svuint8_t data = svldff1_u8(pg, (const uint8_t*)(str + len));
            svbool_t valid = svrdffr();
            svbool_t zeros = svcmpeq_n_u8(valid, data, 0);
            
            if (svptest_any(pg, zeros)) {
                len += svcntp_b8(pg, svbrkb_z(pg, zeros));
                break;
            }
            
            uint64_t valid_count = svcntp_b8(pg, valid);
            len += valid_count;
            
            if (valid_count < svcntp_b8(pg, pg)) {
                break;
            }
        }
        return len;
    };
    
    // Test with multiple string lengths
    std::vector<std::pair<size_t, std::string>> test_cases = {
        {16, "Short string"},
        {64, "Medium length string for initial testing"},
        {256, "Longer string to start seeing SVE benefits"},
        {1024, "1KB string"},
        {4096, "4KB string"},
        {16384, "16KB string"},
        {65536, "64KB string"},
        {262144, "256KB string"},
        {1048576, "1MB string"}
    };
    
    // Generate test strings
    std::cout << "String Length Performance Comparison:\n" << std::endl;
    std::cout << std::setw(12) << "Length" 
              << std::setw(15) << "Standard (ns)" 
              << std::setw(15) << "SVE FF (ns)" 
              << std::setw(12) << "Speedup" 
              << std::setw(20) << "Throughput GB/s" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& test_case : test_cases) {
    size_t target_len = test_case.first;
    const std::string& desc = test_case.second;
    
    // Create string of target length
    std::string test_str;
    test_str.reserve(target_len + 1);
    
    // Fill with readable pattern
    const char* pattern = "The quick brown fox jumps over the lazy dog. ";
    size_t pattern_len = strlen(pattern);
    
    while (test_str.length() < target_len) {
        size_t remaining = target_len - test_str.length();
        if (remaining >= pattern_len) {
            test_str += pattern;
        } else {
            test_str += std::string(pattern, remaining);
        }
    }
        
        // Ensure it's exactly the target length
        test_str.resize(target_len);
        
        // Warm-up runs
        volatile size_t dummy = 0;
        for (int i = 0; i < 100; ++i) {
            dummy += strlen(test_str.c_str());
            dummy += sve_strlen_ff(test_str.c_str());
        }
        
        // Benchmark iterations - more for shorter strings
        int iterations = std::max(1000, (int)(1000000 / target_len));
        
        // Standard strlen
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile size_t len = strlen(test_str.c_str());
            (void)len;
        }
        auto end = std::chrono::high_resolution_clock::now();
        double standard_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)iterations;
        
        // SVE strlen
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            volatile size_t len = sve_strlen_ff(test_str.c_str());
            (void)len;
        }
        end = std::chrono::high_resolution_clock::now();
        double sve_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double)iterations;
        
        // Calculate metrics
        double speedup = standard_ns / sve_ns;
        double throughput_gb_s = (target_len / sve_ns); // GB/s
        
        // Display results
        std::cout << std::setw(12) << target_len
                  << std::setw(15) << std::fixed << std::setprecision(1) << standard_ns
                  << std::setw(15) << std::fixed << std::setprecision(1) << sve_ns
                  << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x"
                  << std::setw(15) << std::fixed << std::setprecision(2) << throughput_gb_s
                  << std::endl;
    }
    
}
void demo_fault_tolerant_loads() {
    std::cout << "\n=== Category 5: Fault-Tolerant Loads ===" << std::endl;
    
    // 5.1 First-Faulting loads with practical strlen example
    print_separator("5.1 First-Faulting Loads (FF) - Vectorized strlen");
    
    // Implementation of vectorized strlen using FF loads
    auto sve_strlen_ff = [](const char* str) -> size_t {
        size_t len = 0;
        svbool_t pg = svptrue_b8();
        
        svsetffr();  // Initialize FFR once
        
        while (true) {
            // Load with first-faulting
            svuint8_t data = svldff1_u8(pg, (const uint8_t*)(str + len));
            svbool_t valid = svrdffr();
            
            // Check for zeros in valid elements
            svbool_t zeros = svcmpeq_n_u8(valid, data, 0);
            
            if (svptest_any(pg, zeros)) {
                // Found a zero - count up to it
                len += svcntp_b8(pg, svbrkb_z(pg, zeros));
                break;
            }
            
            // Count valid elements loaded
            len += svcntp_b8(pg, valid);
            
            // If not all elements were valid, we hit a fault
            if (svcntp_b8(pg, valid)<svcntb()) {
                break;
            }
        }
        return len;
    };
    
    // Standard strlen for comparison
    auto standard_strlen = [](const char* str) -> size_t {
        return strlen(str);
    };
    
    // Test 1: Normal string with null terminator
    std::cout << "Test 1: Normal null-terminated string" << std::endl;
    const char* test_string = "Hello, this is a test string for SVE first-faulting loads demonstration!";
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t len_standard = standard_strlen(test_string);
    auto end = std::chrono::high_resolution_clock::now();
    auto standard_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    size_t len_sve = sve_strlen_ff(test_string);
    end = std::chrono::high_resolution_clock::now();
    auto sve_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    std::cout << "  String: \"" << test_string << "\"" << std::endl;
    std::cout << "  Standard strlen: " << len_standard << " (time: " << standard_time << " ns)" << std::endl;
    std::cout << "  SVE FF strlen:  " << len_sve << " (time: " << sve_time << " ns)" << std::endl;
    std::cout << "  Match: " << (len_standard == len_sve ? "YES" : "NO") << std::endl;
    
    // Test 2: String at page boundary without null terminator
    std::cout << "\nTest 2: String at page boundary (no null terminator)" << std::endl;
    
    // Allocate memory at page boundary
    size_t page_size = sysconf(_SC_PAGESIZE);
    void* pages = mmap(nullptr, page_size * 2, PROT_READ | PROT_WRITE, 
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) {
        std::cerr << "mmap failed" << std::endl;
        return;
    }
    
    // Make second page inaccessible
    if (mprotect((char*)pages + page_size, page_size, PROT_NONE) != 0) {
        std::cerr << "mprotect failed" << std::endl;
        munmap(pages, page_size * 2);
        return;
    }
    
    // Place string at end of first page without null terminator
    size_t test_len = 65;
    char* boundary_str = (char*)pages + page_size - test_len;
    memset(boundary_str, 'A', test_len);  // No null terminator!
    
    std::cout << "  String placed at: " << (void*)boundary_str << std::endl;
    std::cout << "  Page boundary at: " << (void*)((char*)pages + page_size) << std::endl;
    std::cout << "  String length: " << test_len << " (no null terminator)" << std::endl;
    
    // SVE FF strlen - should handle gracefully
    size_t ff_result = sve_strlen_ff(boundary_str);
    std::cout << "  SVE FF strlen result: " << ff_result << " (stopped at page boundary)" << std::endl;
    
    // Standard strlen would crash - demonstrate with signal handler
    struct sigaction sa, old_sa;
    sa.sa_handler = sigsegv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGSEGV, &sa, &old_sa);
    
    std::cout << "  Standard strlen: ";
    if (setjmp(jmp_env) == 0) {
        size_t std_result = strlen(boundary_str);  // This will segfault
        std::cout << std_result << std::endl;
    } else {
        std::cout << "SEGMENTATION FAULT (as expected)" << std::endl;
    }
    
    sigaction(SIGSEGV, &old_sa, nullptr);
    
    // Cleanup
    munmap(pages, page_size * 2);
    demo_strlen_performance();
    // 5.2 Non-Faulting loads with protected memory
    print_separator("5.2 Non-Faulting Loads (NF) - Protected Memory Access");
    
    // Allocate pages again for NF demonstration
    pages = mmap(nullptr, page_size * 3, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (pages == MAP_FAILED) {
        std::cerr << "mmap failed" << std::endl;
        return;
    }
    
    // Fill first page with data
    int32_t* data_page = (int32_t*)pages;
    for (size_t i = 0; i < page_size/sizeof(int32_t); ++i) {
        data_page[i] = 1000 + i;
    }
    
    // Make second page inaccessible
    if (mprotect((char*)pages + page_size, page_size, PROT_NONE) != 0) {
        std::cerr << "mprotect failed" << std::endl;
        munmap(pages, page_size * 3);
        return;
    }
    
    // Fill third page with data
    int32_t* third_page = (int32_t*)((char*)pages + page_size * 2);
    for (size_t i = 0; i < page_size/sizeof(int32_t); ++i) {
        third_page[i] = 2000 + i;
    }
    
    std::cout << "Memory layout:" << std::endl;
    std::cout << "  Page 1 (accessible): " << pages << " - " << (void*)((char*)pages + page_size) << std::endl;
    std::cout << "  Page 2 (protected):  " << (void*)((char*)pages + page_size) << " - " << (void*)((char*)pages + page_size * 2) << std::endl;
    std::cout << "  Page 3 (accessible): " << (void*)((char*)pages + page_size * 2) << " - " << (void*)((char*)pages + page_size * 3) << std::endl;
    
    // Test NF loads across page boundaries
    svbool_t pg = svptrue_b32();
    uint64_t count = svcntw();
    
    // Position load to span across protected page
    int32_t* load_ptr = (int32_t*)((char*)pages + page_size - count * 2);
    
    std::cout << "\nAttempting to load " << count << " elements starting at offset " 
              << ((char*)load_ptr - (char*)pages) << " bytes" << std::endl;
    std::cout << "This will span into the protected page." << std::endl;
    
    // Non-faulting load - should handle gracefully
    std::cout << "\nNon-Faulting load (svldnf1_s32):" << std::endl;
    svsetffr();
    svint32_t nf_result = svldnf1_s32(pg, load_ptr);
    svbool_t nf_valid = svrdffr();
    
    std::vector<int32_t> nf_values(count);
    svst1_s32(pg, nf_values.data(), nf_result);
    
    std::cout << "  Loaded values: ";
    uint64_t valid_count = svcntp_b32(pg, nf_valid);
    for (uint64_t i = 0; i < count; ++i) {
        if (i < valid_count) {
            std::cout << nf_values[i] << " ";
        } else {
            std::cout << "(invalid) ";
        }
    }
    std::cout << std::endl;
    std::cout << "  Valid elements: " << valid_count << " out of " << count << std::endl;
    
    // Regular load would crash - demonstrate
    std::cout << "\nRegular load (svld1_s32):" << std::endl;
    sigaction(SIGSEGV, &sa, &old_sa);
    
    if (setjmp(jmp_env) == 0) {
        //set jump can not capture svld1 segment fault
        //remove this note to see core dump here
        //svint32_t regular_result = svld1_s32(pg, load_ptr);
        std::cout << "  Unexpectedly succeeded!" << std::endl;
    } else {
        std::cout << "  SEGMENTATION FAULT (as expected)" << std::endl;
    }
    
    sigaction(SIGSEGV, &old_sa, nullptr);
    
    // Cleanup
    munmap(pages, page_size * 3);
    
    // Performance comparison for large buffer processing
    print_separator("5.3 Performance Comparison - Safe Buffer Processing");
    
    size_t buffer_size = 1024 * 1024;  // 1MB
    std::vector<char> buffer(buffer_size);
    for (size_t i = 0; i < buffer_size; ++i) {
        buffer[i] = 'A' + (i % 26);
    }
    
    // Insert some null terminators
    for (size_t i = 1000; i < buffer_size; i += 1000) {
        buffer[i] = '\0';
    }
    
    std::cout << "Processing " << buffer_size << " byte buffer with periodic null terminators" << std::endl;
    
    // Count strings using FF
    auto count_strings_ff = [](const char* buf, size_t size) -> size_t {
        size_t count = 0;
        size_t pos = 0;
        
        while (pos < size) {
            size_t len = 0;
            svbool_t pg = svptrue_b8();
            svsetffr();
            
            while (pos + len < size) {
                svuint8_t data = svldff1_u8(pg, (const uint8_t*)(buf + pos + len));
                svbool_t valid = svrdffr();
                svbool_t zeros = svcmpeq_n_u8(valid, data, 0);
                
                if (svptest_any(pg, zeros)) {
                    len += svcntp_b8(pg, svbrkb_z(pg, zeros));
                    break;
                }
                
                len += svcntp_b8(pg, valid);
                if (svcntp_b8(pg, valid)<svcntb()) break;
            }
            
            if (pos + len < size && buf[pos + len] == '\0') {
                count++;
                pos += len + 1;
            } else {
                break;
            }
        }
        return count;
    };
    
    // Count strings using standard method
    auto count_strings_standard = [](const char* buf, size_t size) -> size_t {
        size_t count = 0;
        size_t pos = 0;
        
        while (pos < size) {
            size_t len = 0;
            while (pos + len < size && buf[pos + len] != '\0') {
                len++;
            }
            if (pos + len < size && buf[pos + len] == '\0') {
                count++;
                pos += len + 1;
            } else {
                break;
            }
        }
        return count;
    };
    
    // Benchmark
    const int iterations = 100;
    
    start = std::chrono::high_resolution_clock::now();
    size_t std_count = 0;
    for (int i = 0; i < iterations; ++i) {
        std_count = count_strings_standard(buffer.data(), buffer_size);
    }
    end = std::chrono::high_resolution_clock::now();
    auto std_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    size_t ff_count = 0;
    for (int i = 0; i < iterations; ++i) {
        ff_count = count_strings_ff(buffer.data(), buffer_size);
    }
    end = std::chrono::high_resolution_clock::now();
    auto ff_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "\nResults (" << iterations << " iterations):" << std::endl;
    std::cout << "  Standard method: " << std_count << " strings, " << std_duration << " μs total" << std::endl;
    std::cout << "  SVE FF method:   " << ff_count << " strings, " << ff_duration << " μs total" << std::endl;
    std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
              << (double)std_duration / ff_duration << "x" << std::endl;
}

void demo_non_temporal_loads() {
    std::cout << "\n=== Category 6: Non-Temporal (Streaming) Loads ===" << std::endl;
    
    // 6.1 Single-threaded baseline (保留原有测试作为基准)
    print_separator("6.1 Single-threaded Baseline");
    
    size_t single_buffer_size = 32 * 1024 * 1024;  // 32MB
    std::vector<float> source_buffer(single_buffer_size / sizeof(float));
    std::vector<float> dest_buffer(single_buffer_size / sizeof(float));
    
    // Initialize with data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 100.0);
    for (size_t i = 0; i < source_buffer.size(); ++i) {
        source_buffer[i] = dis(gen);
    }
    
    std::cout << "Single-threaded test with " << single_buffer_size / (1024*1024) << " MB buffer" << std::endl;
    
    // Lambda for regular processing
    auto process_regular = [](const float* src, float* dst, size_t count) {
        svbool_t pg = svptrue_b32();
        size_t vl = svcntw();
        svfloat32_t scale = svdup_f32(2.5f);
        
        for (size_t i = 0; i < count; i += vl) {
            svbool_t pred = svwhilelt_b32(i, count);
            svfloat32_t data = svld1_f32(pred, &src[i]);
            svfloat32_t result = svmul_f32_x(pred, data, scale);
            svst1_f32(pred, &dst[i], result);
        }
    };
    
    // Lambda for non-temporal processing
    auto process_streaming = [](const float* src, float* dst, size_t count) {
        svbool_t pg = svptrue_b32();
        size_t vl = svcntw();
        svfloat32_t scale = svdup_f32(2.5f);
        
        for (size_t i = 0; i < count; i += vl) {
            svbool_t pred = svwhilelt_b32(i, count);
            svfloat32_t data = svldnt1_f32(pred, &src[i]);
            svfloat32_t result = svmul_f32_x(pred, data, scale);
            svstnt1_f32(pred, &dst[i], result);
        }
    };
    
    // Benchmark single-threaded
    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        process_regular(source_buffer.data(), dest_buffer.data(), source_buffer.size());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        process_streaming(source_buffer.data(), dest_buffer.data(), source_buffer.size());
    }
    end = std::chrono::high_resolution_clock::now();
    auto streaming_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Results (single-threaded):" << std::endl;
    std::cout << "  Regular loads:      " << regular_time << " ms" << std::endl;
    std::cout << "  Non-temporal loads: " << streaming_time << " ms" << std::endl;
    std::cout << "  Difference: " << std::fixed << std::setprecision(1) 
              << ((double)(regular_time - streaming_time) / regular_time * 100) << "%" << std::endl;
    
    // 6.2 Multi-threaded cache competition test
    print_separator("6.2 Multi-threaded Cache Competition");
    
    const int num_threads = 4;  // Adjust based on your CPU
    const size_t thread_buffer_size = 16 * 1024 * 1024;  // 16MB per thread
    const size_t hot_data_size = 1 * 1024 * 1024;  // 1MB hot data
    
    std::cout << "Multi-threaded test setup:" << std::endl;
    std::cout << "  Threads: " << num_threads << std::endl;
    std::cout << "  Buffer per thread: " << thread_buffer_size / (1024*1024) << " MB" << std::endl;
    std::cout << "  Hot data size: " << hot_data_size / (1024*1024) << " MB" << std::endl;
    
    // Results storage
    struct ThreadResult {
        double hot_data_latency;
        double streaming_throughput;
        size_t cache_misses;  // Simulated metric
    };
    
    // Test function for mixed workload
    auto run_mixed_workload = [&](bool use_non_temporal) {
        std::vector<ThreadResult> results(num_threads);
        std::vector<std::thread> threads;
        std::atomic<bool> start_flag{false};
        std::atomic<int> ready_count{0};
        
        // Hot data shared by thread 0
        std::vector<float> hot_data(hot_data_size / sizeof(float));
        for (size_t i = 0; i < hot_data.size(); ++i) {
            hot_data[i] = dis(gen);
        }
        
        // Thread 0: Hot data access (simulating latency-sensitive workload)
        threads.emplace_back([&]() {
            // CPU affinity (optional, platform-specific)
            #ifdef __linux__
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(0, &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
            #endif
            
            ready_count++;
            while (!start_flag) { std::this_thread::yield(); }
            
            // Repeatedly access hot data
            const int hot_iterations = 1000;
            auto start = std::chrono::high_resolution_clock::now();
            
            float sum = 0;
            for (int iter = 0; iter < hot_iterations; ++iter) {
                // Random access pattern to stress cache
                for (size_t i = 0; i < hot_data.size(); i += 64/sizeof(float)) {
                    size_t idx = (i * 997) % hot_data.size();  // Prime number for scatter
                    sum += hot_data[idx];
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            results[0].hot_data_latency = 
                std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 
                (double)hot_iterations;
            
            // Prevent optimization
            volatile float v = sum;
            (void)v;
        });
        
        // Threads 1-3: Streaming data processing
        for (int tid = 1; tid < num_threads; ++tid) {
            threads.emplace_back([&, tid]() {
                #ifdef __linux__
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(tid, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
                #endif
                
                // Create thread-local buffers
                std::vector<float> src(thread_buffer_size / sizeof(float));
                std::vector<float> dst(thread_buffer_size / sizeof(float));
                
                // Initialize with random data
                for (size_t i = 0; i < src.size(); ++i) {
                    src[i] = tid * 1000.0f + i * 0.01f;
                }
                
                ready_count++;
                while (!start_flag) { std::this_thread::yield(); }
                
                // Process streaming data
                auto start = std::chrono::high_resolution_clock::now();
                
                if (use_non_temporal) {
                    process_streaming(src.data(), dst.data(), src.size());
                } else {
                    process_regular(src.data(), dst.data(), src.size());
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                results[tid].streaming_throughput = 
                    (thread_buffer_size / (1024.0 * 1024.0)) / (time_ms / 1000.0);  // GB/s
            });
        }
        
        // Wait for all threads to be ready
        while (ready_count < num_threads) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        // Start all threads simultaneously
        start_flag = true;
        
        // Wait for completion
        for (auto& t : threads) {
            t.join();
        }
        
        return results;
    };
    
    // Run tests
    std::cout << "\nRunning mixed workload with regular loads..." << std::endl;
    auto regular_results = run_mixed_workload(false);
    
    std::cout << "Running mixed workload with non-temporal loads..." << std::endl;
    auto streaming_results = run_mixed_workload(true);
    
    // Display results
    std::cout << "\nResults comparison:" << std::endl;
    std::cout << "Hot data access (Thread 0) - Lower is better:" << std::endl;
    std::cout << "  With regular loads:      " << std::fixed << std::setprecision(2) 
              << regular_results[0].hot_data_latency << " μs/iteration" << std::endl;
    std::cout << "  With non-temporal loads: " << std::fixed << std::setprecision(2) 
              << streaming_results[0].hot_data_latency << " μs/iteration" << std::endl;
    std::cout << "  Improvement: " << std::fixed << std::setprecision(1)
              << ((regular_results[0].hot_data_latency - streaming_results[0].hot_data_latency) / 
                  regular_results[0].hot_data_latency * 100) << "%" << std::endl;
    
    std::cout << "\nStreaming throughput (Threads 1-3 avg) - Higher is better:" << std::endl;
    double avg_regular = 0, avg_streaming = 0;
    for (int i = 1; i < num_threads; ++i) {
        avg_regular += regular_results[i].streaming_throughput;
        avg_streaming += streaming_results[i].streaming_throughput;
    }
    avg_regular /= (num_threads - 1);
    avg_streaming /= (num_threads - 1);
    
    std::cout << "  With regular loads:      " << std::fixed << std::setprecision(2) 
              << avg_regular << " GB/s" << std::endl;
    std::cout << "  With non-temporal loads: " << std::fixed << std::setprecision(2) 
              << avg_streaming << " GB/s" << std::endl;
    
    // 6.3 Cache pollution demonstration
    print_separator("6.3 Cache Pollution Measurement");
    
    std::cout << "Demonstrating cache pollution effects:" << std::endl;
    
    // Create a small working set that fits in L2/L3 cache
    const size_t working_set_size = 256 * 1024;  // 256KB
    std::vector<float> working_set(working_set_size / sizeof(float));
    for (size_t i = 0; i < working_set.size(); ++i) {
        working_set[i] = dis(gen);
    }
    
    // Lambda to measure access time to working set
    auto measure_working_set_latency = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        float sum = 0;
        for (int iter = 0; iter < 100; ++iter) {
            for (size_t i = 0; i < working_set.size(); ++i) {
                sum += working_set[i];
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        volatile float v = sum;
        (void)v;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 100.0;
    };
    
    // Warm up working set
    std::cout << "Warming up working set..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        measure_working_set_latency();
    }
    
    double baseline_latency = measure_working_set_latency();
    std::cout << "Baseline working set access time: " << baseline_latency / 1000.0 << " μs" << std::endl;
    
    // Process large buffer with regular loads (pollutes cache)
    std::cout << "\nProcessing 64MB with regular loads..." << std::endl;
    std::vector<float> large_buffer(64 * 1024 * 1024 / sizeof(float));
    std::vector<float> large_dest(64 * 1024 * 1024 / sizeof(float));
    process_regular(large_buffer.data(), large_dest.data(), large_buffer.size());
    
    double after_regular_latency = measure_working_set_latency();
    std::cout << "Working set access time after regular loads: " 
              << after_regular_latency / 1000.0 << " μs ("
              << std::fixed << std::setprecision(1)
              << (after_regular_latency / baseline_latency - 1) * 100 << "% slower)" << std::endl;
    
    // Re-warm working set
    for (int i = 0; i < 10; ++i) {
        measure_working_set_latency();
    }
    
    // Process large buffer with non-temporal loads (should not pollute cache)
    std::cout << "\nProcessing 64MB with non-temporal loads..." << std::endl;
    process_streaming(large_buffer.data(), large_dest.data(), large_buffer.size());
    
    double after_streaming_latency = measure_working_set_latency();
    std::cout << "Working set access time after non-temporal loads: " 
              << after_streaming_latency / 1000.0 << " μs ("
              << std::fixed << std::setprecision(1)
              << (after_streaming_latency / baseline_latency - 1) * 100 << "% slower)" << std::endl;
    
    std::cout << "\nCache pollution effect:" << std::endl;
    std::cout << "  Regular loads caused " 
              << std::fixed << std::setprecision(1)
              << (after_regular_latency / baseline_latency - 1) * 100 << "% slowdown" << std::endl;
    std::cout << "  Non-temporal loads caused " 
              << std::fixed << std::setprecision(1)
              << (after_streaming_latency / baseline_latency - 1) * 100 << "% slowdown" << std::endl;
}
// --- Advanced Examples ---
void demo_advanced_examples() {
    std::cout << "\n=== Advanced Examples ===" << std::endl;
    
    // Example 1: Complex number processing
    print_separator("Example 1: Complex Number Processing");
    uint64_t count = svcntw();
    std::vector<float> complex_data(count * 2);
    for(uint64_t i = 0; i < count; ++i) {
        complex_data[i*2] = 1.0f + i;      // Real part
        complex_data[i*2+1] = 10.0f + i;   // Imaginary part
    }
    
    std::cout << "Input: Complex numbers in interleaved format" << std::endl;
    print_array_f32("Interleaved data", complex_data.data(), std::min(complex_data.size(), size_t(16)));
    std::cout << "Format: [real[0], imag[0], real[1], imag[1], ...]" << std::endl;
    
    svbool_t pg = svptrue_b32();
    svfloat32x2_t complex_vecs = svld2_f32(pg, complex_data.data());
    svfloat32_t real = svget2_f32(complex_vecs, 0);
    svfloat32_t imag = svget2_f32(complex_vecs, 1);
    
    std::cout << "\nAfter svld2_f32 de-interleaving:" << std::endl;
    print_f32("Real parts", real, count);
    print_f32("Imaginary parts", imag, count);
    
    // Compute magnitude squared: real^2 + imag^2
    svfloat32_t mag_sq = svmla_f32_x(pg, svmul_f32_x(pg, real, real), imag, imag);
    
    std::cout << "\nComputed magnitude squared (real² + imag²):" << std::endl;
    print_f32("Magnitudes²", mag_sq, count);
    
    // Example 2: Structure of Arrays to Array of Structures
    print_separator("Example 2: SoA to AoS Conversion");
    
    // Simulate gathering from different arrays
    std::vector<uint8_t> types(count);
    std::vector<int8_t> priorities(count);
    std::vector<uint16_t> flags(count);
    std::vector<float> values(count);
    
    for(uint64_t i = 0; i < count; ++i) {
        types[i] = i % 4;
        priorities[i] = -10 + i;
        flags[i] = 0x100 + i;
        values[i] = 100.0f + i * 0.5f;
    }
    
    std::cout << "Input: Separate arrays (Structure of Arrays)" << std::endl;
    print_array_u8("Types (u8)", types.data(), std::min(types.size(), size_t(16)));
    print_array_s8("Priorities (i8)", priorities.data(), std::min(priorities.size(), size_t(16)));
    print_array_u16("Flags (u16)", flags.data(), std::min(flags.size(), size_t(16)));
    print_array_f32("Values (f32)", values.data(), std::min(values.size(), size_t(16)));
    
    // Load and extend different data types
    svuint32_t type_vec = svld1ub_u32(pg, types.data());
    svint32_t priority_vec = svld1sb_s32(pg, priorities.data());
    svuint32_t flags_vec = svld1uh_u32(pg, flags.data());
    svfloat32_t value_vec = svld1_f32(pg, values.data());
    
    std::cout << "\nAfter loading with type conversion to 32-bit:" << std::endl;
    print_u32("Types (u8→u32)", type_vec, count);
    print_s32("Priorities (i8→i32)", priority_vec, count);
    print_u32("Flags (u16→u32)", flags_vec, count);
    print_f32("Values", value_vec, count);
}

int main() {
    std::cout << "SVE Load Instructions Demonstration" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "SVE Implementation Details:" << std::endl;
    std::cout << "  Vector length: " << svcntb() << " bytes" << std::endl;
    std::cout << "  Elements per vector:" << std::endl;
    std::cout << "    8-bit:  " << svcntb() << std::endl;
    std::cout << "    16-bit: " << svcnth() << std::endl;
    std::cout << "    32-bit: " << svcntw() << std::endl;
    std::cout << "    64-bit: " << svcntd() << std::endl;

    demo_contiguous_interleaved_loads();
    demo_load_with_conversion();
    demo_gather_loads();
    demo_load_and_broadcast();
    demo_fault_tolerant_loads();
    demo_non_temporal_loads();
    demo_advanced_examples();

    return 0;
}
