package com.data;

public class TanhBackward extends Node {
    public TanhBackward(Tensor t1, Tensor result) {
        super();
        this.add_child(t1.node);
        this.tensor = result;
    }
}
