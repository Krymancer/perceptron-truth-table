#pragma once

#include <vector>
#include <iostream>
#include <random>

// Header file to main

const char* message = "CPP-PROJECT TEMPLATE";

class Perceptron {
   public:
    std::vector<double> weights;
    double lr;

    Perceptron(int inputLength, double learningRate) {
        this->weights.resize(inputLength, 0);

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < this->weights.size(); i++) {
            this->weights[i] = dist(rng);
        }

        this->lr = learningRate;
    }

    double sing(double value) {
        return value > 0 ? 1 : 0;
    }

    double guess(std::vector<double> inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * this->weights[i];
        }

        return sing(sum);
    }

    void train(std::vector<double> inputs, double target) {
        double attempt = guess(inputs);
        double error = target - attempt;

        for (int i = 0; i < weights.size(); i++) {
            this->weights[i] += error * inputs[i] * this->lr;
        }
    }
};

class Gate {
   public:
    double a, b, label;
    double bias = 1;

    Gate() {}

    Gate(int a, int b, int label) {
        this->a = a;
        this->b = b;
        this->label = label;
    }
};