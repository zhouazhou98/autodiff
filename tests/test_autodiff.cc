#include "zhou.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace autodiff;

void test_mul() {
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



}

void test_sin() {
    auto x = std::make_shared<AutoDiff>(M_PI);
    auto re = sin(x);
    std::cout << "x value = " << x->m_value << std::endl;
    std::cout << "x gradient = " << x->m_gradient << std::endl;
    re->backward();
    std::cout << "x value = " << x->m_value << std::endl;
    std::cout << "x gradient = " << x->m_gradient << std::endl;
    std::cout << "result value of sin(" << x->m_value << ") = " << re->m_value << std::endl;
}

void test_nn() {
    std::vector<AutoDiff::ptr> X = { 
                            std::make_shared<AutoDiff>(0.1), 
                            std::make_shared<AutoDiff>(0.2), 
                            std::make_shared<AutoDiff>(0.3), 
                        };

    std::vector<AutoDiff::ptr> Y = { 
                            std::make_shared<AutoDiff>(0.3), 
                            std::make_shared<AutoDiff>(0.6), 
                            std::make_shared<AutoDiff>(0.9), 
                        };



    auto w = std::make_shared<AutoDiff>(1);
    auto b = std::make_shared<AutoDiff>(1);

    double learning_rate = 0.1;

    for (int epoch = 0; epoch < 100000; ++ epoch) {
        std::vector<AutoDiff::ptr> Y_pred = {
                                            X[0] * w + b,
                                            X[1] * w + b,
                                            X[2] * w + b,
                                        };
        
        auto loss = 
                        (Y_pred[0] - Y[0]) * (Y_pred[0] - Y[0]) +
                        (Y_pred[1] - Y[1]) * (Y_pred[1] - Y[1]) +
                        (Y_pred[2] - Y[2]) * (Y_pred[2] - Y[2])
                        ;

        loss->backward();

        w->m_value -= learning_rate * w->m_gradient;
        b->m_value -= learning_rate * b->m_gradient;

        w->m_gradient = 0;
        b->m_gradient = 0;

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " - Loss: " << loss->m_value << std::endl;
            double learning_rate = 0.1 / 2;
        } 
    }

    std::cout << "w: " << w->m_value << ", b: " << b->m_value << std::endl;
}


int main() {
    test_mul();

    std::cout << "-----------------------------------" << std::endl;
    test_sin();

    std::cout << "-----------------------------------" << std::endl;
    test_nn();

    
    return 0;
}
