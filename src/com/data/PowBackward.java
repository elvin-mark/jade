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
        Tensor t1 = this.children.get(0).tensor;
        Tensor new_loss = new Tensor(t1.shape);

        for (int i = 0; i < new_loss.size; i++) {
            new_loss.data[i] = this.exponent * ((float) Math.pow(t1.data[i], this.exponent - 1.0f)) * loss.data[i];
        }
        t1.node.backward(new_loss);
    }
}
