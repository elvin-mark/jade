package com.data;

public class SigmoidBackward extends Node {
    public SigmoidBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t = this.children.get(0).tensor;
        for (int i = 0; i < t.data.length; i++) {
            t.grad.data[i] = t.data[i] * (1 - t.data[i]) * loss.data[i];
        }
    }
}
