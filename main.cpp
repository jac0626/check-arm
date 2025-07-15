 #include <arm_sve.h>
   int main() {
       svbool_t pg = svptrue_b16();
       svuint16_t u16_a, u16_b;
       svuint32_t u32_c = svdup_u32(0);
       u32_c = svmlalt_u32(u32_c, u16_a, u16_b);
       return 0;
   }
