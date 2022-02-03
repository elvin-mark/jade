package com.data;

public class PowBackward extends Node {
    float exponent;

    public PowBackward(Tensor t1, float exponent, Tensor result) {
        super();
        this.add_child(t1.node);
        this.exponent = exponent;
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        // TODO
        Tensor t1 = this.children.get(0).tensor;
        Tensor t1_grad = t1.grad;
        if (t1_grad == null) {
            t1_grad = new Tensor(t1);
            t1.grad = t1_grad;
        }
        for (int i = 0; i < t1_grad.size; i++) {
            t1_grad.data[i] = this.exponent * this.tensor.data[i] * loss.data[i];
        }
    }
}
