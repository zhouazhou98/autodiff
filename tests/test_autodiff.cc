#include "zhou.h"
#include <iostream>
#include <memory>

int main() {
    using namespace autodiff;

    // 栈上三个 AutoDiff 实例 a b c
    auto a = std::make_shared<AutoDiff>(8.0);
    auto b = std::make_shared<AutoDiff>(10.0);
    auto c = std::make_shared<AutoDiff>(3.0);

    // 计算 a * b * c
    AutoDiff::ptr result = a * b * c;
    result = result * c * a;

    // 输出初始值
    std::cout << "a value = " << a->m_value << std::endl;
    std::cout << "b value = " << b->m_value << std::endl;
    std::cout << "c value = " << c->m_value << std::endl;
    std::cout << "result value = " << result->m_value << std::endl;

    // 反向传播
    result->backward();

    // 输出梯度
    std::cout << "a gradient = " << a->m_gradient << std::endl;
    std::cout << "b gradient = " << b->m_gradient << std::endl;
    std::cout << "c gradient = " << c->m_gradient << std::endl;
    std::cout << "result value = " << result->m_value << std::endl;


    std::cout << "-----------------------------------" << std::endl;


    
    auto x = std::make_shared<AutoDiff>(M_PI);
    auto re = sin(x);
    std::cout << "x value = " << x->m_value << std::endl;
    std::cout << "x gradient = " << x->m_gradient << std::endl;
    re->backward();
    std::cout << "x value = " << x->m_value << std::endl;
    std::cout << "x gradient = " << x->m_gradient << std::endl;
    std::cout << "result value of sin(" << x->m_value << ") = " << re->m_value << std::endl;


    return 0;
}
