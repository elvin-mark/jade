package com.data;

public class NormBackward extends Node {
    public NormBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        Tensor t = this.children.get(0).tensor;
        Tensor new_loss = (new Tensor(t)).div(this.tensor);
        t.node.backward(new_loss);
    }
}
