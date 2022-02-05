package com.utils;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.io.File;
import com.data.Tensor;

public class Misc {
    public static final int TYPE_CHAR = 0;
    public static final int TYPE_INT = 1;
    public static final int TYPE_FLOAT = 2;

    public static Tensor loadTensor(String path) {
        try {
            FileInputStream fis = new FileInputStream(new File(path));
            byte[] buffer = new byte[4];
            byte[] data_buffer = new byte[1];
            fis.read(buffer);
            int typeData = ByteBuffer.wrap(buffer).getInt();
            fis.read(buffer);
            int num_shape = ByteBuffer.wrap(buffer).getInt();
            int[] shape = new int[num_shape];
            for (int i = 0; i < num_shape; i++) {
                fis.read(buffer);
                shape[i] = ByteBuffer.wrap(buffer).getInt();
            }
            Tensor out = new Tensor(shape);
            for (int i = 0; i < out.size; i++) {
                if (typeData == TYPE_CHAR) {
                    fis.read(data_buffer);
                    out.data[i] = (float) data_buffer[0];
                } else if (typeData == TYPE_INT) {
                    fis.read(buffer);
                    out.data[i] = ByteBuffer.wrap(buffer).getInt();
                } else if (typeData == TYPE_FLOAT) {
                    fis.read(buffer);
                    out.data[i] = ByteBuffer.wrap(buffer).getFloat();
                }
            }
            return out;
        } catch (Exception e) {
            System.out.println("Error reading file: " + path);
        }
        return null;
    }

    public static void saveTensor(Tensor t, String path) {
        try {
            FileOutputStream fos = new FileOutputStream(new File(path));
            fos.write(ByteBuffer.allocate(4).putInt(TYPE_FLOAT).array());
            fos.write(ByteBuffer.allocate(4).putInt(t.shape.length).array());
            for (int i = 0; i < t.shape.length; i++) {
                fos.write(ByteBuffer.allocate(4).putInt(t.shape[i]).array());
            }
            for (int i = 0; i < t.size; i++) {
                fos.write(ByteBuffer.allocate(4).putFloat(t.data[i]).array());
            }
        } catch (Exception e) {
            System.out.println("Error writing file: " + path);
        }
    }
}