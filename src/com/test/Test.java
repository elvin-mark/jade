package com.test;

import com.data.Tensor;
import com.functions.F;
import com.nn.*;
import com.utils.*;
import com.optim.*;

import java.util.HashMap;
import java.util.Random;

public class Test {
        public static void testDraft() {
                // Some random tests
                Tensor t = new Tensor(new int[] { 1, 2, 3, 4 });
                NNModule bn = new BatchNorm2d(2);
                for (int i = 0; i < 24; i++) {
                        t.data[i] = i;
                }

                t.requires_grad(true);

                System.out.println(bn.forward(t));
                Tensor o = bn.forward(t).norm();
                System.out.println(o);

                o.backward();
                System.out.println(t.grad);
                for (Tensor param : bn.parameters()) {
                        System.out.println(param);
                        System.out.println(param.grad);
                }
        }

        public static void main(String args[]) {
                testDraft();
        }
}
