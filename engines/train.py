
import tensorflow as tf
import numpy as np
import math
import time
import datetime
from tqdm import tqdm
from engines.model import NerModel
from engines.utils.metrics import metrics

from engines.utils.dice import DiceLoss
tf.compat.v1.enable_eager_execution()
from tensorflow_addons.text.crf import crf_decode
from keras.callbacks import TensorBoard
from tensorflow.python.ops import summary_ops_v2

logs_dir = 'fit_loggs/graph'
log_dir = 'fit_loggs/loss'

#def dice_loss(n_classes,logits, label, smooth=1.e-5):
#    epsilon = 1.e-6
#    alpha = 2.0  # 这个是dice coe的系数，见下边的解释
#    y_true = tf.one_hot(label, n_classes)
#    softmax_prob = tf.nn.softmax(logits)
#    #print("{}".format(softmax_prob.numpy()))
#    y_pred = tf.clip_by_value(softmax_prob, epsilon, 1. - epsilon)

#    y_pred_mask = tf.multiply(y_pred, y_true)
#    common = tf.multiply((tf.ones_like(y_true) - y_pred_mask), y_pred_mask)
#    nominator = tf.multiply(tf.multiply(common, y_true), alpha) + smooth
#    denominator = common + y_true + smooth
#    dice_coe = tf.divide(nominator, denominator)
#    return tf.reduce_mean(tf.reduce_max(1 - dice_coe, axis=-1))

class Train:
    def __init__(self, configs, data_manager, logger):
        self.logger = logger
        self.configs = configs
        self.data_manager = data_manager
        self.batch_size = configs.batch_size
        self.epoch = configs.epoch

        vocab_size = data_manager.max_token_number
        num_classes = data_manager.max_label_number
        learning_rate = configs.learning_rate
        max_to_keep = configs.checkpoints_max_to_keep
        checkpoints_dir = configs.checkpoints_dir
        checkpoint_name = configs.checkpoint_name

        if configs.optimizer == 'Adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif configs.optimizer == 'Adadelta':
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        elif configs.optimizer == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif configs.optimizer == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif configs.optimizer == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            from tensorflow_addons.optimizers import AdamW
            self.optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-2)

        if configs.use_pretrained_model and not configs.finetune:
            if configs.pretrained_model == 'Bert':
                from transformers import TFBertModel
                self.pretrained_model = TFBertModel.from_pretrained('bert-base-chinese')

        self.ner_model = NerModel(configs, vocab_size, num_classes)



        checkpoint = tf.train.Checkpoint(ner_model=self.ner_model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
        checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print('Restored from {}'.format(self.checkpoint_manager.latest_checkpoint))
        else:
            print('Initializing from scratch.')

    def train(self):
        train_dataset, val_dataset = self.data_manager.get_training_set()
        best_f1_val = 0.0
        best_at_epoch = 0
        unprocessed = 0
        very_start_time = time.time()

        self.logger.info(('+' * 20) + 'training starting' + ('+' * 20))


        for i in range(self.epoch):

            #summary_writer = tf.summary.create_file_writer(log_dir)  # 实例化记录器
            #tf.summary.trace_on(profiler=True)  # 开启Trace（可选）

            start_time = time.time()
            self.logger.info('epoch:{}/{}'.format(i + 1, self.epoch))
            for step, batch in tqdm(train_dataset.shuffle(len(train_dataset)).batch(self.batch_size).enumerate()):

                if self.configs.use_pretrained_model:
                    X_train_batch, y_train_batch, att_mask_batch = batch
                    if self.configs.finetune:
                        # 如果微调
                        model_inputs = (X_train_batch, att_mask_batch)
                    else:
                        # 不进行微调，预训练模型只做特征的增强
                        model_inputs = self.pretrained_model(X_train_batch, attention_mask=att_mask_batch)[0]
                else:
                    X_train_batch, y_train_batch = batch
                    model_inputs = X_train_batch
                # 计算没有加入pad之前的句子的长度
                inputs_length = tf.math.count_nonzero(X_train_batch, 1)


                #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs/", histogram_freq=1)


                with tf.GradientTape() as tape:
                    logits, log_likelihood, transition_params = self.ner_model(
                        inputs=model_inputs, inputs_length=inputs_length, targets=y_train_batch, training=1)
                    loss = -tf.reduce_mean(log_likelihood)


                    #log_dir = "logs_fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                    #callbacks = [tensorboard_callback]

                    #loss = DiceLoss(self.data_manager.max_label_number,logits,[y_train_batch,y_train_batch+1])
                    #loss = (-tf.reduce_mean(log_likelihood)+Dice_loss)/2

                    #with summary_writer.as_default():  # 指定记录器
                     #   tf.summary.scalar("loss", loss, step=step)  # 将当前损失函数的值写入记录器
                        #tf.summary.scalar("accuracy", val_res_str, step=step)

                # 定义好参加梯度的参数
                variables = self.ner_model.trainable_variables
                # 将预训练模型里面的pooler层的参数去掉
                variables = [var for var in variables if 'pooler' not in var.name]
                gradients = tape.gradient(loss, variables)

                if self.configs.use_gan:
                    if self.configs.gan_method == 'fgm':
                        # 使用FGM的对抗办法
                        epsilon = 1.0
                        embedding = variables[0]
                        embedding_gradients = gradients[0]
                        embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
                        delta = epsilon * embedding_gradients / tf.norm(embedding_gradients, ord=2)

                        accum_vars = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]
                        gradients = [accum_vars[i].assign_add(grad) for i, grad in enumerate(gradients)]
                        variables[0].assign_add(delta)

                        with tf.GradientTape() as gan_tape:
                            logits, log_likelihood, _ = self.ner_model(inputs=model_inputs, inputs_length=inputs_length,
                                                                  targets=y_train_batch, training=1)
                            loss = -tf.reduce_mean(log_likelihood)

                            #loss = DiceLoss(self.data_manager.max_label_number,logits,[y_train_batch,y_train_batch+1])
                            #loss = (-tf.reduce_mean(log_likelihood)+Dice_loss)/2
                        gan_gradients = gan_tape.gradient(loss, variables)
                        gradients = [gradients[i].assign_add(grad) for i, grad in enumerate(gan_gradients)]
                        variables[0].assign_sub(delta)

                    elif self.configs.gan_method == 'pgd':
                        # 使用PGD的对抗办法
                        K = 3
                        alpha = 0.3
                        epsilon = 1
                        origin_embedding = tf.Variable(variables[0])
                        accum_vars = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]
                        origin_gradients = [accum_vars[i].assign_add(grad) for i, grad in enumerate(gradients)]

                        for t in range(K):
                            embedding = variables[0]
                            embedding_gradients = gradients[0]
                            embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
                            delta = alpha * embedding_gradients / tf.norm(embedding_gradients, ord=2)
                            variables[0].assign_add(delta)

                            r = variables[0] - origin_embedding
                            if tf.norm(r, ord=2) > epsilon:
                                r = epsilon * r / tf.norm(r, ord=2)
                            variables[0].assign(origin_embedding + tf.Variable(r))

                            if t != K - 1:
                                gradients = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]
                            else:
                                gradients = origin_gradients
                            with tf.GradientTape() as gan_tape:
                                logits, log_likelihood, _ = self.ner_model(inputs=model_inputs, inputs_length=inputs_length,
                                                                      targets=y_train_batch, training=1)
                                loss = -tf.reduce_mean(log_likelihood)


                                #loss = (-tf.reduce_mean(log_likelihood)+Dice_loss)/2
                                # loss = DiceLoss(self.data_manager.max_label_number,logits,[y_train_batch,y_train_batch+1])

                            gan_gradients = gan_tape.gradient(loss, variables)
                            gradients = [gradients[i].assign_add(grad) for i, grad in enumerate(gan_gradients)]
                        variables[0].assign(origin_embedding)

                # 反向传播，自动微分计算
                self.optimizer.apply_gradients(zip(gradients, variables))
                if step % self.configs.print_per_batch == 0 and step != 0:
                    batch_pred_sequence, _ = crf_decode(logits, transition_params, inputs_length)
                    measures, _ = metrics(
                        X_train_batch, y_train_batch, batch_pred_sequence, self.configs, self.data_manager)
                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    self.logger.info('training batch: %5d, loss: %.5f, %s' % (step, loss, res_str))


            val_f1_avg, val_res_str = self.validate(val_dataset)
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))

            if np.array(val_f1_avg).mean() > best_f1_val:
                unprocessed = 0
                best_f1_val = np.array(val_f1_avg).mean()
                best_at_epoch = i + 1
                self.checkpoint_manager.save()
                self.logger.info('saved the new best model with f1: %.4f' % best_f1_val)
            else:
                unprocessed += 1

            if self.configs.is_early_stop:
                if unprocessed >= self.configs.patient:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(self.configs.patient))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                    self.logger.info('total training time consumption: %.4f(min)' % ((time.time() - very_start_time) / 60))
                    return


            #inputs = model_inputs
            #inputs_length = inputs_length
            #targets = y_train_batch
            # "graph"
            #graph_writer = tf.summary.create_file_writer(logdir=logs_dir)
            #with graph_writer.as_default():
            #    graph = self.ner_model.call.get_concrete_function(inputs, inputs_length, targets).graph
            #    summary_ops_v2.graph(graph.as_graph_def())
            #graph_writer.close()


             # "acc_loss"
            #with summary_writer.as_default():
            #    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)  # 保存Trace信息到文件（可选）


        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
        self.logger.info('total training time consumption: %.4f(min)' % ((time.time() - very_start_time) / 60))

    def validate(self, val_dataset):
        # validation

        #summary_writer = tf.summary.create_file_writer('./tensorboard')

        num_val_iterations = int(math.ceil(1.0 * len(val_dataset) / self.batch_size))
        self.logger.info('start evaluate engines...')
        loss_values = []
        val_results = {}
        val_labels_results = {}
        for label in self.data_manager.suffix:
            val_labels_results.setdefault(label, {})
        for measure in self.configs.measuring_metrics:
            val_results[measure] = 0
        for label, content in val_labels_results.items():
            for measure in self.configs.measuring_metrics:
                if measure != 'accuracy':
                    val_labels_results[label][measure] = 0

        for val_batch in tqdm(val_dataset.batch(self.batch_size)):
        #for val_batch in tqdm(val_dataset.shuffle(len(val_dataset)).batch(self.batch_size)):

            # ...（训练代码，当前batch的损失值放入变量loss中）
            #with summary_writer.as_default():  # 希望使用的记录器
            #    tf.summary.scalar("loss", loss, step=batch_index)
            #    tf.summary.scalar("MyScalar", my_scalar, step=batch_index)  # 还可以添加其他自定义的变量

            if self.configs.use_pretrained_model:
                X_val_batch, y_val_batch, att_mask_batch = val_batch
                if self.configs.finetune:
                    model_inputs = (X_val_batch, att_mask_batch)
                else:
                    model_inputs = self.pretrained_model(X_val_batch, attention_mask=att_mask_batch)[0]
            else:
                X_val_batch, y_val_batch = val_batch
                model_inputs = X_val_batch
            inputs_length_val = tf.math.count_nonzero(X_val_batch, 1)
            logits_val, log_likelihood_val, transition_params_val = self.ner_model(
                inputs=model_inputs, inputs_length=inputs_length_val, targets=y_val_batch)

            #val_loss = DiceLoss(self.data_manager.max_label_number,logits_val,[y_val_batch,y_val_batch+1])
            #val_loss = -tf.reduce_mean(log_likelihood_val)   原本应该在这

            val_loss = -tf.reduce_mean(log_likelihood_val)



            #val_loss = (-tf.reduce_mean(log_likelihood_val)+Dice_loss_val)/2
            batch_pred_sequence_val, _ = crf_decode(logits_val, transition_params_val, inputs_length_val)
            measures, lab_measures = metrics(
                X_val_batch, y_val_batch, batch_pred_sequence_val, self.configs, self.data_manager)


            #val_loss = -tf.reduce_mean(log_likelihood_val)

            for k, v in measures.items():
                val_results[k] += v
            for lab in lab_measures:
                for k, v in lab_measures[lab].items():
                    val_labels_results[lab][k] += v
            loss_values.append(val_loss)

        val_res_str = ''
        val_f1_avg = 0
        for k, v in val_results.items():
            val_results[k] /= num_val_iterations
            val_res_str += (k + ': %.4f ' % val_results[k])
            if k == 'f1':
                val_f1_avg = val_results[k]
        for label, content in val_labels_results.items():
            val_label_str = ''
            for k, v in content.items():
                val_labels_results[label][k] /= num_val_iterations
                val_label_str += (k + ': %.4f ' % val_labels_results[label][k])
            self.logger.info('label: %s, %s' % (label, val_label_str))
        return val_f1_avg, val_res_str
