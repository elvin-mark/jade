package com.data;

public class SigmoidBackward extends Node {
    public SigmoidBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t = this.children.get(0).tensor;
        Tensor bp_loss = new Tensor(this.tensor.shape);
        // Get the gradient of the loss with respect to the input
        for (int i = 0; i < this.tensor.data.length; i++) {
            bp_loss.data[i] = this.tensor.data[i] * (1 - this.tensor.data[i]) * loss.data[i];
        }
        t.node.backward(bp_loss);
    }
}
