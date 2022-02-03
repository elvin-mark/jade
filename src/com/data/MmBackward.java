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
        Tensor new_loss_1, new_loss_2;

        new_loss_1 = loss.mm(t2.transpose());
        new_loss_2 = t1.transpose().mm(loss);

        t1.node.backward(new_loss_1);
        t2.node.backward(new_loss_2);
    }
}