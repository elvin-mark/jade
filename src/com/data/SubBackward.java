package com.data;

public class SubBackward extends Node {
    public SubBackward(Tensor t1, Tensor t2, Tensor result) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t1 = this.children.get(0).tensor;
        Tensor t2 = this.children.get(1).tensor;
        Tensor new_loss_1 = new Tensor(t1.shape);
        Tensor new_loss_2 = new Tensor(t2.shape);
        new_loss_1.zeros();
        new_loss_2.zeros();
        // Get the gradient of the loss with respect to the input t1
        for (int i = 0; i < loss.data.length; i++) {
            new_loss_1.data[i % new_loss_1.size] += loss.data[i];
            new_loss_2.data[i % new_loss_2.size] += -loss.data[i];
        }
        t1.node.backward(new_loss_1);
        t2.node.backward(new_loss_2);
    }
}
