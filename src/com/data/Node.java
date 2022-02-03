package com.data;

import java.util.*;

public class Node {
    public ArrayList<Node> children;
    public Tensor tensor;

    public Node() {
        this.children = new ArrayList<Node>();
    }

    public Node(Tensor tensor) {
        this.children = new ArrayList<Node>();
        this.tensor = tensor;
    }

    public void add_child(Node node) {
        this.children.add(node);
    }

    public void backward(Tensor loss) {
        // implement this
    }

    public boolean is_leaf() {
        return this.children.size() == 0;
    }
}
