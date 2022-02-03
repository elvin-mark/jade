package com.functions;

import com.data.Tensor;

public class F {
    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        return input.mm(weight).add(bias);
    }

    public static Tensor linear(Tensor input, Tensor weight) {
        return input.mm(weight);
    }

    public static Tensor conv1d(Tensor input, Tensor weight, Tensor bias, int stride, int padding) {
        return input.conv1d(weight, stride, padding).add(bias);
    }

    public static Tensor conv1d(Tensor input, Tensor weight, int stride, int padding) {
        return input.conv1d(weight, stride, padding);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int[] stride, int[] padding) {
        return input.conv2d(weight, stride, padding).add(bias);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, int[] stride, int[] padding) {
        return input.conv2d(weight, stride, padding);
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

    public static Tensor softmax(Tensor input) {
        return input.softmax();
    }

    public static Tensor mse_loss(Tensor input, Tensor target) {
        return input.sub(target).pow(2).mean();
    }

    public static Tensor cross_entropy_loss(Tensor input, Tensor target) {
        // TODO: Fix this
        // return -target.mul(input.log()).mean();
        return null;
    }

    public static Tensor nllloss(Tensor input, Tensor target) {
        // TODO: Fix this
        // return -target.mul(input.log()).mean();
        return null;
    }
}
