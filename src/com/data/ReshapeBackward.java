package com.data;

public class ReshapeBackward extends Node {

    public ReshapeBackward(Tensor input, Tensor result) {
        super();
        this.add_child(input.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor new_loss = loss.reshape(t1.shape);
        t1.node.backward(new_loss);
    }
}
