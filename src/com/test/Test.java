package com.test;

import com.data.Tensor;
import com.nn.*;
import com.utils.*;
import com.optim.*;

import java.util.HashMap;

public class Test {
        public static void testDraft() {
                // Some random tests
                Tensor a = new Tensor(new int[] { 5, 5 });
                a.random(0.0f, 1.0f);
                System.out.println(a);
                System.out.println(a.argmax(1));
        }

        public static void main(String args[]) {
                testDraft();
        }
}
