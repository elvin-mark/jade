package com.data;

public class SoftmaxBackward extends Node {
    public SoftmaxBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t = this.children.get(0).tensor;
        Tensor bp_loss = new Tensor(this.tensor.shape);
        // Get the gradient of the loss with respect to the input
        // TODO: Implement this

        t.node.backward(bp_loss);
    }
}
