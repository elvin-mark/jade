package com.data;

public class SiLUBackward extends Node {
    public SiLUBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t = this.children.get(0).tensor;
        Tensor bp_loss = new Tensor(this.tensor.shape);
        // Get the gradient of the loss with respect to the input
        float tmp;
        for (int i = 0; i < this.tensor.data.length; i++) {
            tmp = this.tensor.data[i] / t.data[i];
            bp_loss.data[i] = (tmp + t.data[i] * tmp * (1 - tmp)) * loss.data[i];
        }
        t.node.backward(bp_loss);
    }
}
