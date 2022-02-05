package com.test;

import com.data.Tensor;
import com.functions.F;
import com.nn.*;

public class Test {
    public static void testDraft() {
        // Tensor t = new Tensor(new int[] { 2, 2, 5 },
        // new float[] { 0.9712881f, 0.9923937f, 0.8624685f, 0.5829317f, 0.8995149f,
        // 0.72404754f, 0.46881002f,
        // 0.68940943f, 0.21434641f, 0.6536304f, 0.004905939f, 0.6508469f, 0.7345901f,
        // 0.44306612f,
        // 0.4500317f, 0.24441409f, 0.084246516f, 0.94214594f, 0.93758404f, 0.39994133f
        // });
        // Tensor f = new Tensor(new int[] { 3, 2, 2 }, new float[] { 0.41704082f,
        // 0.82422876f, 0.7174187f, 0.4910134f,
        // 0.008470118f, 0.20370054f, 0.36883116f, 0.8914987f, 0.29231638f, 0.3895387f,
        // 0.0074648857f,
        // 0.57268023f });

        // t.requires_grad(true);
        // f.requires_grad(true);

        // Tensor o = t.conv1d(f, 1, 0);
        // Tensor l = o.norm();

        // l.backward();

        // System.out.println(l);
        // System.out.println(t.grad);
        // System.out.println(f.grad);

        // for (int i : o.shape()) {
        // System.out.println(i);
        // }
        // System.out.println(o);

        Tensor t = new Tensor(new int[] { 2, 2, 5, 5 },
                new float[] { 0.3327623f, 0.25732797f, 0.7411025f, 0.8152998f, 0.4871192f, 0.018178046f, 0.27371395f,
                        0.46738803f, 0.08962768f, 0.24393004f, 0.1207397f, 0.97099733f, 0.92101806f, 0.19064635f,
                        0.13614321f, 0.9095381f, 0.2619018f, 0.107441306f, 0.74280334f, 0.9135377f, 0.34820253f,
                        0.2850039f, 0.97924906f, 0.5052691f, 0.17082971f, 0.47346598f, 0.6211467f, 0.88913685f,
                        0.4316635f, 0.5301048f, 0.25615674f, 0.037943125f, 0.3682357f, 0.9991708f, 0.12097317f,
                        0.41212767f, 0.14675605f, 0.11108911f, 0.22192353f, 0.06655347f, 0.28825855f, 0.77032775f,
                        0.9255996f, 0.79702026f, 0.6492246f, 0.58492565f, 0.6220956f, 0.90558267f, 0.8349316f,
                        0.79970294f, 0.55111945f, 0.5956184f, 0.63243926f, 0.5772865f, 0.88690454f, 0.24876696f,
                        0.18022233f, 0.2268815f, 0.2206794f, 0.28869206f, 0.61535746f, 0.8871881f, 0.3869788f,
                        0.31941915f, 0.9102659f, 0.6519549f, 0.15237159f, 0.29964232f, 0.6306594f, 0.71563053f,
                        0.22410965f, 0.18494874f, 0.45872617f, 0.11660689f, 0.1577543f, 0.21624964f, 0.3310613f,
                        0.6801327f, 0.3599763f, 0.964489f, 0.36949658f, 0.762786f, 0.3775332f, 0.91635376f, 0.20399445f,
                        0.50870353f, 0.41993666f, 0.020020962f, 0.40539396f, 0.6474827f, 0.9142037f, 0.88398737f,
                        0.4952643f, 0.003460586f, 0.5400901f, 0.79509777f, 0.55922526f, 0.2093482f, 0.44739783f,
                        0.4269716f });
        Tensor f = new Tensor(new int[] { 3, 2, 2, 2 }, new float[] { 0.63637525f, 0.90769744f, 0.63487697f,
                0.73867834f, 0.006724417f, 0.06864488f, 0.03238207f, 0.49349475f, 0.2679162f, 0.7569688f, 0.5991523f,
                0.8661887f, 0.6211604f, 0.9890883f, 0.44461554f, 0.77066934f, 0.31578702f, 0.2138865f, 0.4884444f,
                0.86112845f, 0.66648f, 0.96514577f, 0.4261791f, 0.9538317f });

        t.requires_grad(true);
        f.requires_grad(true);
        Tensor o = t.conv2d(f, new int[] { 1, 1 }, new int[] { 0, 0 });
        Tensor l = o.norm();
        System.out.println(l);
        l.backward();
        System.out.println(t.grad);
        System.out.println(f.grad);
        for (int i : o.shape()) {
            System.out.println(i);
        }
        System.out.println(o);
    }

    public static void testTensors() {
        int[] shape1 = { 2, 2 };
        int[] shape2 = { 2, 2 };
        int[] shape3 = { 2 };

        float[] data1 = { 1.0f, 2.0f, 3.0f, 4.0f };
        float[] data2 = { 2.0f, 3.0f, 4.0f, 5.0f };
        float[] data3 = { 8.0f, 9.0f };

        Tensor t1 = new Tensor(data1);
        Tensor t2 = new Tensor(data2);
        Tensor b = new Tensor(data3);
        Tensor c = new Tensor(2.3f);

        t1.requires_grad(true);

        t1.reshape(new int[] { 2, 2 });
        t2.reshape(new int[] { 2, 2 });

        Tensor t3 = F.linear(t1, t2, b);
        t1.print();
        t2.print();
        t3.print();
        System.out.println(t3.requires_grad_);
        System.out.println(t3);

        System.out.println(t1.mul(c));
    }

    public static void testModule() {
        NNModule seq = new Sequential();
        seq.add_module(new NNModule());

        seq.add_module((NNModule) new Linear(2, 2, true));
        seq.add_module((NNModule) new Sigmoid());
        seq.add_module((NNModule) new Linear(2, 1, true));
        seq.add_module((NNModule) new Sigmoid());

        Tensor input = new Tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f });
        Tensor target = new Tensor(new float[] { 0.0f, 1.0f });

        input.reshape(new int[] { 2, 2 });
        target.reshape(new int[] { 2, 1 });

        Tensor tmp = input.transpose();
        System.out.println(tmp.at(new int[] { 0, 1 }));

        System.out.println(input.transpose().mm(input));
        System.out.println(input.mm(input));

        Tensor output = seq.forward(input);

        System.out.println(output);
        System.out.println(output.requires_grad_);

        for (Tensor param : seq.parameters()) {
            param.print();
        }

        Loss loss_fn = new MSELoss();
        Tensor loss = loss_fn.criterion(output, target);
        loss.backward();
        System.out.println(loss);

        for (Tensor param : seq.parameters()) {
            System.out.println("param: " + param);
            System.out.println("grad: " + param.grad);
        }
    }

    public static void testBackward() {
        Tensor t1 = new Tensor(1.2f);
        Tensor t2 = new Tensor(2.3f);
        Tensor t3 = new Tensor(3.4f);
        Tensor t4 = new Tensor(4.5f);
        Tensor A = new Tensor(new int[] { 2, 2 });
        A.random(0.0f, 1.0f);
        A.requires_grad(true);
        Tensor tmp = A.mean();
        tmp.backward();
        System.out.println(A);
        System.out.println(tmp);
        System.out.println(A.grad);

        t1.requires_grad(true);
        t2.requires_grad(true);
        t3.requires_grad(true);
        t4.requires_grad(true);

        Tensor t5 = t4.div(t1.mul(t2).add(t3));
        t5.backward();
        System.out.println(t5);
        System.out.println(t1.grad);
        System.out.println(t2.grad);
        System.out.println(t3.grad);
        System.out.println(t4.grad);
    }

    public static void testLoss() {
        Tensor t1_ = new Tensor(new int[] { 3, 2, 2, 3 },
                new float[] { 0.5289f, 0.4626f, 0.5518f, 0.3476f, 0.5939f, 0.8576f, 0.4022f, 0.0636f, 0.8197f,
                        0.3296f, 0.8923f, 0.9391f, 0.6479f, 0.0469f, 0.9633f, 0.4420f, 0.1681f, 0.2387f,
                        0.0612f, 0.5698f, 0.6793f, 0.6686f, 0.5192f, 0.6727f, 0.4391f, 0.2874f, 0.0601f,
                        0.0318f, 0.4551f, 0.5309f, 0.1438f, 0.5036f, 0.6353f, 0.2046f, 0.4513f, 0.5982f });

        t1_.requires_grad(true);
        Tensor tmp = t1_.logsoftmax();
        Tensor t2_ = new Tensor(new int[] { 3, 1, 2, 3 },
                new float[] { 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1 });
        CrossEntropyLoss l_ = new CrossEntropyLoss();
        Tensor loss_ = l_.criterion(tmp, t2_);
        System.out.println(loss_);
        System.out.println(loss_.requires_grad_);
        loss_.backward();
        System.out.println(t1_.grad);
        Tensor t = new Tensor(new int[] { 4, 2 }, new float[] { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f });
        System.out.println(t.softmax());
        Tensor t1 = t.logsoftmax();
        Tensor t2 = new Tensor(new int[] { 4, 1 }, new float[] { 1, 0, 0, 1 });

        NLLLoss l = new NLLLoss();
        Tensor o = l.criterion(t1, t2);
        System.out.println(o.item());
    }

    public static void main(String args[]) {
        // testModule();
        // testBackward();
        // testLoss();
        testDraft();
    }
}
