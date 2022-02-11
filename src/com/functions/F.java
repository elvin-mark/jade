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

    public static Tensor maxpool2d(Tensor input, int[] kernel) {
        return input.maxpool2d(kernel);
    }

    public static Tensor dropout2d(Tensor input, float p) {
        return input.dropout2d(p);
    }

    public static Tensor batchnorm2d(Tensor input, Tensor runningMean, Tensor runningVar, Tensor gamma, Tensor beta,
            float momentum) {
        // TODO: implement
        // Calculate the mean and variance of the input
        Tensor mean = input.mean(new int[] { 0, 2, 3 });
        Tensor var = input.var(new int[] { 0, 2, 3 }, mean);
        // Normalize the input
        mean.requires_grad(false);
        var.requires_grad(false);

        runningMean.data = runningMean.mul(new Tensor(1 - momentum)).add(mean.mul(new Tensor(momentum))).data;
        runningVar.data = runningVar.mul(new Tensor(1 - momentum)).add(var.mul(new Tensor(momentum))).data;

        Tensor normalized = input.sub(new Tensor(mean)).div(var.add(new Tensor(1e-5f)).pow(0.5f));
        // Scale and shift the normalized input
        return normalized.mul(gamma).add(beta);
    }

    public static Tensor batchnorm2d(Tensor input, Tensor runningMean, Tensor runningVar, Tensor gamma, Tensor beta) {
        // TODO: implement
        // Calculate the mean and variance of the input
        Tensor normalized = input.sub(runningMean).div(runningVar.add(new Tensor(1e-5f)).pow(0.5f));
        // Scale and shift the normalized input
        return normalized.mul(gamma).add(beta);
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

    public static Tensor logsoftmax(Tensor input) {
        return input.logsoftmax();
    }

    public static Tensor mse_loss(Tensor input, Tensor target) {
        return input.sub(target).pow(2).mean();
    }

    public static Tensor nll_loss(Tensor input, Tensor target) {
        /*
         * input: [N, C, ... ]
         * target: [N, 1, ... ]
         */

        return input.nll(target);
    }

    public static Tensor cross_entropy_loss(Tensor input, Tensor target) {
        /*
         * input: [N, C, ... ]
         * target: [N, 1, ... ]
         */
        return nll_loss(input.logsoftmax(), target);
    }

}
