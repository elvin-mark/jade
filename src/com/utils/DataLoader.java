package com.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Iterator;

import java.util.Collections;
import com.data.Tensor;

public class DataLoader implements Iterable<Tensor[]>, Iterator<Tensor[]> {
    Dataset dataset;
    int batchSize;
    int cursor;
    List<Integer> indices;

    public DataLoader(Dataset dataset, int batchSize, boolean shuffle) {
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.indices = new ArrayList<Integer>();
        for (int i = 0; i < this.dataset.size(); i++) {
            this.indices.add(i);
        }
        Collections.shuffle(this.indices);
    }

    public Tensor[] getBatch() {
        Tensor[] out = new Tensor[2];
        Tensor[] aux;
        int num_elems = this.dataset.size() - this.cursor;

        if (num_elems >= this.batchSize) {
            num_elems = this.batchSize;
        }
        aux = this.dataset.get(this.indices.get(this.cursor));
        this.cursor += 1;
        int[] x_new_shape = new int[aux[0].shape.length + 1];
        int[] y_new_shape = new int[aux[1].shape.length + 1];
        x_new_shape[0] = num_elems;
        y_new_shape[0] = num_elems;
        for (int i = 0; i < aux[0].shape.length; i++) {
            x_new_shape[i + 1] = aux[0].shape[i];
        }
        for (int i = 0; i < aux[1].shape.length; i++) {
            y_new_shape[i + 1] = aux[1].shape[i];
        }
        out[0] = new Tensor(x_new_shape);
        out[1] = new Tensor(y_new_shape);
        for (int i = 1; i < num_elems; i++) {
            aux = this.dataset.get(this.indices.get(this.cursor));
            this.cursor += 1;
            for (int j = 0; j < aux[0].size; j++) {
                out[0].data[i * aux[0].size + j] = aux[0].data[j];
            }
            for (int j = 0; j < aux[1].size; j++) {
                out[1].data[i * aux[1].size + j] = aux[1].data[j];
            }
        }
        return out;
    }

    public int size() {
        if (this.dataset.size() % this.batchSize == 0) {
            return this.dataset.size() / this.batchSize;
        }
        return this.dataset.size() / this.batchSize + 1;
    }

    public boolean hasNext() {
        return this.cursor < this.dataset.size();
    }

    public Iterator<Tensor[]> iterator() {
        this.cursor = 0;
        return this;
    }

    public Tensor[] next() {
        if (this.cursor >= this.dataset.size()) {
            throw new java.util.NoSuchElementException("No more batches in the dataset");
        }
        return this.getBatch();
    }
}
