#include "ninfer_glm53/exl3_decode.hpp"
#include "exl3_test_oracle.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

static void expect(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
template<class F> void rejects(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return; }
    expect(false, "invalid linear input was accepted");
}
int main() {
    using namespace ninfer::glm53;
    std::vector<std::uint16_t> packed(128*128/4), scales(128,0x3c00);
    std::vector<float> x(128);x[0]=1;
    Exl3LinearView all_zero{128,128,packed,scales,scales};
    const auto y=exl3_linear_reference(all_zero,x);
    expect(std::fabs(y[0]-236.0)<1e-10,"all-zero trellis requires H128 on both sides");
    for(unsigned i=1;i<128;++i) expect(std::fabs(y[i])<1e-10,"all-zero basis tail");
    for (const auto& dims : {std::pair{128u,128u},std::pair{256u,128u},std::pair{128u,256u}}) {
        const auto [k,n]=dims;
        std::vector<std::uint16_t> codes(k*n/4),su(k),sv(n),bias(n,0x3400);
        for(unsigned i=0;i<codes.size();++i) codes[i]=static_cast<std::uint16_t>(i*40503u+7919u);
        for(unsigned i=0;i<k;++i)su[i]=(i%3==0?0xbc00:i%3==1?0x3800:0x4000);
        for(unsigned i=0;i<n;++i)sv[i]=(i%2==0?0x3c00:0xb800);
        std::vector<float> input(2*k);
        for(unsigned i=0;i<input.size();++i)input[i]=float(int(i%17)-8)/8;
        Exl3LinearView m{k,n,codes,su,sv,bias};const auto actual=exl3_linear_reference(m,input,2);
        for(unsigned row=0;row<2;++row) {
            std::vector<double> a(k),b(n);
            for(unsigned i=0;i<k;++i)a[i]=input[row*k+i]*oracle_half(su[i]);
            a=oracle_h128(a);
            for(unsigned i=0;i<k;++i)for(unsigned j=0;j<n;++j) {
                const auto offset=((i/16)*(n/16)+j/16)*64;
                b[j]+=a[i]*oracle_mcg(oracle_state(codes.data()+offset,oracle_order(i%16,j%16)));
            }
            b=oracle_h128(b);
            for(unsigned j=0;j<n;++j)
                expect(std::fabs(actual[row*n+j]-(b[j]*oracle_half(sv[j])+oracle_half(bias[j])))<1e-8,"multi-tile dense Sylvester oracle");
        }
    }
    auto invalid=all_zero;invalid.in_features=16;
    rejects([&]{(void)exl3_linear_reference(invalid,x);});
    invalid=all_zero;invalid.trellis=std::span(packed).first(packed.size()-1);
    rejects([&]{(void)exl3_linear_reference(invalid,x);});
    rejects([&]{(void)exl3_linear_reference(all_zero,x,0);});
    rejects([&]{(void)exl3_linear_reference(all_zero,x,33);});
    rejects([&]{(void)exl3_linear_reference(all_zero,x,1,16383);});
    rejects([&]{(void)exl3_linear_reference(all_zero,std::span(x).first(127));});
    x[0]=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{(void)exl3_linear_reference(all_zero,x);});x[0]=1;
    scales[0]=0x7c00;
    rejects([&]{(void)exl3_linear_reference(all_zero,x);});
    std::cout << "Analytic H128, three multi-tile/two-row shapes and eight rejection cases passed\n";
}
