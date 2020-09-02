#include "main.hpp"

#include <iostream>
#include <random>
#include <vector>

#define AND

int main(int argc, char** argv) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> idist(0, 3);

    int population = 4;
    double trainedRate = 0;
    double untrainedRate = 0;

    int epoch = 10000;

    Perceptron perceptron(3, 0.01);

    std::vector<Gate> gates;
    gates.resize(population);

#ifdef AND
    gates[0] = Gate(0, 0, 0);
    gates[1] = Gate(0, 1, 0);
    gates[2] = Gate(1, 0, 0);
    gates[3] = Gate(1, 1, 1);
#endif

#ifdef OR
    gates[0] = Gate(0, 0, 0);
    gates[1] = Gate(0, 1, 1);
    gates[2] = Gate(1, 0, 1);
    gates[3] = Gate(1, 1, 1);
#endif

#ifdef NAND
    gates[0] = Gate(0, 0, 1);
    gates[1] = Gate(0, 1, 1);
    gates[2] = Gate(1, 0, 1);
    gates[3] = Gate(1, 1, 0);
#endif

#ifdef NOR
    gates[0] = Gate(0, 0, 1);
    gates[1] = Gate(0, 1, 0);
    gates[2] = Gate(1, 0, 0);
    gates[3] = Gate(1, 1, 0);
#endif

    /*
     *  Note: XOR and XNOR are not linear functions, so a perceptron can't predict them
     * 
     *    XOR    XNOR
     *  |#|0|1| |#|0|1|
     *  |0|F|T| |0|T|F|
     *  |1|T|F| |1|F|T|
     * 
     *  At top and left represent the inputs and T stands for true and F for false
     * 
     *  There no way to draw a line that separate True and False. however in AND and OR its
     *  possible, so percptron can predic correctly.
     * 
    */

#ifdef XOR
    gates[0] = Gate(0, 0, 0);
    gates[1] = Gate(0, 1, 1);
    gates[2] = Gate(1, 0, 1);
    gates[3] = Gate(1, 1, 0);
#endif

#ifdef XNOR
    gates[0] = Gate(0, 0, 1);
    gates[1] = Gate(0, 1, 0);
    gates[2] = Gate(1, 0, 0);
    gates[3] = Gate(1, 1, 1);
#endif

    for (int i = 0; i < epoch; i++) {
        std::vector<double> inputs;

        int gate = idist(rng);

        inputs.push_back(gates[gate].a);
        inputs.push_back(gates[gate].b);
        inputs.push_back(gates[gate].bias);

        double target = gates[gate].label;

        perceptron.train(inputs, target);
    }

    for (auto gate : gates) {
        std::vector<double> inputs;
        inputs.push_back(gate.a);
        inputs.push_back(gate.b);
        inputs.push_back(gate.bias);

        double target = gate.label;

        double result = perceptron.guess(inputs);

        std::cout << gate.a << " and " << gate.b << " = " << result << " Expected: " << gate.label << std::endl;
    }

    std::cout << "Done" << std::endl;
    return 0;
}