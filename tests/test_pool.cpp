#include "cpso/thread_pool.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    cpso::ThreadPool pool(4);
    for (std::size_t size : {0U,1U,3U,4U,17U,64U,67U,1003U}) {
        std::vector<std::atomic<unsigned>> seen(size);
        for(auto& x:seen) x.store(0);
        for(int rep=0;rep<30;rep++) {
            pool.parallel_for(size,[&](std::size_t i) { seen[i].fetch_add(1); });
        }
        for (auto& x:seen) assert(x.load()==30);
    }
    bool caught=false;
    try { pool.parallel_for(100,[](std::size_t i) { if(i==7) throw std::runtime_error("sentinel"); }); }
    catch(const std::runtime_error&) { caught=true; }
    assert(caught);
    std::cout<<"Thread pool: every index once, nondivisible work, exception propagation PASS\n";
}
