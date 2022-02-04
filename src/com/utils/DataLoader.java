package com.utils;

import com.data.Tensor;

public class DataLoader {
    Dataset dataset;
    int batchSize;

    public DataLoader(Dataset dataset, int batchSize) {
        this.dataset = dataset;
        this.batchSize = batchSize;
    }

    public Tensor[] getBatch() {
        // FIX ME
        return null;
    }
}
