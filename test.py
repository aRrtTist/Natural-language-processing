"""
测试神经网络模型
"""

from utils import *
from network import *


def test(model_path, test_data, vocab_size, id_to_word):
    # 测试的输入
    test_input = Input(batch_size=20, num_steps=35, data=test_data)

    # 创建测试的模型，基本的超参数需要和训练时用的一致，例如：
    # hidden_size，num_steps，num_layers，vocab_size，batch_size 等等
    # 因为要载入训练时保存的参数的文件，如果超参数不匹配 TensorFlow 会报错
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocab_size, num_layers=2)

    # 用 Saver 来恢复训练时生成的模型的变量
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Coordinator（协调器），用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 当前的状态
        # 第二维是 2 是因为测试时指定只有 2 层 LSTM
        # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
        # 一个是 前一时刻 LSTM 的输出 h(t-1)
        # 一个是 前一时刻的单元状态 C(t-1)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))

        # 恢复被训练的模型的变量
        saver.restore(sess, model_path)

        # 测试 30 个批次
        num_acc_batches = 30

        # 打印预测单词和实际单词的批次数
        check_batch_idx = 25

        # 超过 5 个批次才开始累加精度
        acc_check_thresh = 5

        # 初始精度的和，用于之后算平均精度
        accuracy = 0

        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                pred_words = [id_to_word[x] for x in pred[:m.num_steps]]
                true_words = [id_to_word[x] for x in true[0]]
                print("\n实际的单词:")
                print(" ".join(true_words))  # 真实的单词
                print("预测的单词:")
                print(" ".join(pred_words))  # 预测的单词
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc

        # 打印平均精度
        print("平均精度: {:.3f}".format(accuracy / (num_acc_batches - acc_check_thresh)))

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    if args.data_path:
        data_path = args.data_path
    if args.load_file:
        load_file = args.load_file
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)

    trained_model = save_path + "/" + load_file

    test(trained_model, test_data, vocab_size, id_to_word)
