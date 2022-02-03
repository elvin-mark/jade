package com.data;

public class ReluBackward extends Node {
    public ReluBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {
        // implement this
    }

}
