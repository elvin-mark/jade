package com.data;

public class SubBackward extends Node {
    public SubBackward(Tensor t1, Tensor t2, Tensor result) {
        super();
        this.add_child(t1.node);
        this.add_child(t2.node);
        this.tensor = result;
    }

    public void backward(Tensor loss) {

    }
}
