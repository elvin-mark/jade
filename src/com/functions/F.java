package com.functions;

import com.data.Tensor;

public class F {
    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        return input.mm(weight).add(bias);
    }

    public static Tensor linear(Tensor input, Tensor weight) {
        return input.mm(weight);
    }

    public static Tensor sigmoid(Tensor input) {
        return input.sigmoid();
    }

    public static Tensor tanh(Tensor input) {
        return input.tanh();
    }

    public static Tensor relu(Tensor input) {
        return input.relu();
    }

    public static Tensor leaky_relu(Tensor input, float alpha) {
        return input.leaky_relu(alpha);
    }

    public static Tensor mse_loss(Tensor input, Tensor target) {
        return input.sub(target).pow(2).mean();
    }
}
