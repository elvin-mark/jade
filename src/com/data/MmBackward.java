package com.data;

public class MmBackward extends Node {
    public MmBackward(Tensor t1, Tensor t2, Tensor result) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor t2 = this.children.get(1).tensor;
        Tensor t = this.tensor;
        // FIXME
        for (int i = 0; i < t.data.length; i++) {
            t.grad.data[i] = t1.data[i] * t2.data[i] * loss.data[i];
        }
    }
}