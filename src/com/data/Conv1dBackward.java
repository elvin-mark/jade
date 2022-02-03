package com.data;

public class Conv1dBackward extends Node {
    int stride;
    int padding;

    public Conv1dBackward(Tensor t1, Tensor t2, Tensor result, int stride, int padding) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.stride = stride;
        this.padding = padding;
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor t2 = this.children.get(1).tensor;

        Tensor new_loss_1 = new Tensor(t1.shape);
        Tensor new_loss_2 = new Tensor(t2.shape);
        // Get the gradient of the loss with respect to the input
        // TODO: Implement this

        t1.node.backward(new_loss_1);
        t2.node.backward(new_loss_2);
    }
}