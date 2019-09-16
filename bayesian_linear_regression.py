import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_polynomial_data(N):
    mean = [1, -0.1, 0.02, 0.001, 0.001]
    cov = np.diag([0.1, 0.0, 0.0, 0.0, 0.0])
    w = np.random.multivariate_normal(mean, cov, N).T
    
    x = np.random.uniform(low=-1.0, high=2.0, size=[N])
    x_poly = np.stack([x, x**2, x**3, x**4, x**5], axis=0)
    y =  np.sum(w * x_poly, axis=0) + np.random.normal(loc=0, scale=0.01, size=[N])
    return x, y


def get_sine_data(N):
    # x = [-0.8*pi, 0.2*pi ] and [1.0*pi, 1.4*pi]
    x = np.concatenate([np.random.uniform(low=-0.8, high=0.2, size=[N//2]),
                        np.random.uniform(low=1.0, high=1.4, size=[N//2])]) * np.pi

    y = 0.25 * np.sin(x) + np.random.normal(loc=0, scale=0.01, size=[N])
    return x, y


def create_linear_model(x):
    in_channels = x.shape[-1]
    tf_w = tf.get_variable('weights', shape=[in_channels, 1], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    tf_y = tf.matmul(x, tf_w)
    return tf_y


def create_nn(x, hidden_units, dropout, dropout_rate=None, is_training=None):
    in_channels = x.shape[-1]

    tf_w1 = tf.get_variable('w1', shape=[in_channels, hidden_units], initializer=tf.glorot_normal_initializer())
    tf_w2 = tf.get_variable('w2', shape=[hidden_units, 1], initializer=tf.glorot_normal_initializer())
    
    out = tf.nn.tanh(tf.matmul(x, tf_w1))

    if dropout:
        out = tf.layers.dropout(out, rate=dropout_rate, training=is_training)

    out = tf.matmul(out, tf_w2)

    return out, [tf_w1, tf_w2]


def create_nn_mvg(x, hidden_units, N, kl_weight, prior_variance):
    in_channels = x.shape[-1]

    tf_w1_mean = tf.get_variable('w1', shape=[in_channels, hidden_units], initializer=tf.glorot_normal_initializer())
    tf_w2_mean = tf.get_variable('w2', shape=[hidden_units, 1], initializer=tf.glorot_normal_initializer())

    tf_w1_fisher = tf.get_variable('w1_fisher', initializer=1e-5 * tf.eye(16), trainable=False)
    tf_w2_fisher = tf.get_variable('w2_fisher', initializer=1e-5 * tf.eye(16), trainable=False)

    tf_w1_cov = tf.linalg.inv(N / kl_weight * tf_w1_fisher + 1. / prior_variance * tf.eye(16))
    tf_w2_cov = tf.linalg.inv(N / kl_weight * tf_w2_fisher + 1. / prior_variance * tf.eye(16))

    tf_w1 = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=tf.reshape(tf_w1_mean, [-1]),
                                                                      covariance_matrix=tf_w1_cov).sample()
    tf_w1 = tf.reshape(tf_w1, [in_channels, hidden_units])

    tf_w2 = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=tf.reshape(tf_w2_mean, [-1]),
                                                                      covariance_matrix=tf_w2_cov).sample()
    tf_w2 = tf.reshape(tf_w2, [hidden_units, 1])

    out = tf.nn.tanh(tf.matmul(x, tf_w1))
    out = tf.matmul(out, tf_w2)

    return out, [(tf_w1_mean, tf_w1_fisher, tf_w1), (tf_w2_mean, tf_w2_fisher, tf_w2)]


def get_adam_optimizer(tf_loss, weights, learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_loss, var_list=weights)


def get_sgd_optimizer(tf_loss, weights, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(tf_loss, var_list=weights)


def get_ng_optimizer(tf_loss, weights, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    grad_and_vars = optimizer.compute_gradients(tf_loss, var_list=weights)

    natural_grad_and_vars = []

    for grad, var in grad_and_vars:
        prev_shape = grad.shape

        grad = tf.reshape(grad, shape=[-1, 1])
        fisher = tf.matmul(grad, grad, transpose_b=True) + tf.eye(tf.size(grad)) * 1e-3
        # e, v = tf.linalg.eigh(fisher)

        inverse_fisher = tf.linalg.inv(fisher)
        natural_gradient = tf.matmul(inverse_fisher, grad)
        natural_gradient = tf.reshape(natural_gradient, shape=prev_shape)

        natural_grad_and_vars.append((natural_gradient, var))

    update_op = optimizer.apply_gradients(natural_grad_and_vars)

    return update_op


def get_nng_optimizer(tf_loss, weights, alpha_, beta_, kl_weight, N, prior_variance, external_damping):
    var_list = [w for _, _, w  in weights]

    grads = tf.gradients(tf_loss, var_list)

    assign_ops = []

    for (mean, fisher, weight_sample), grad in zip(weights, grads):
        prev_shape = grad.shape
        grad = tf.reshape(grad, shape=[-1, 1]) # grad flattened
        cur_fisher = tf.matmul(grad, grad, transpose_b=True)

        new_fisher = beta_ * fisher + (1. - beta_) * cur_fisher
        damped_fisher = new_fisher + (kl_weight / N / prior_variance + external_damping) * tf.eye(tf.size(grad))

        step = tf.matmul(tf.linalg.inv(damped_fisher),
                         (grad - kl_weight / N / prior_variance * tf.reshape(weight_sample, [-1, 1])))

        step = tf.reshape(step, shape=prev_shape)

        assign_ops.append(tf.assign(mean, mean + alpha_ * step))
        assign_ops.append(tf.assign(fisher, new_fisher))

    return assign_ops


if __name__ == '__main__':

    N = 100

    x, y = get_sine_data(N=N)
    x_test = np.linspace(-2 * np.pi, 3 * np.pi, num=200)
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    x_mean = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)

    # x, y = get_polynomial_data(N=N)
    # x_test = np.linspace(-2 * np.pi, 3 * np.pi, num=200)
    # x = np.stack([x, x**2, x**3], axis=1)
    # y = np.expand_dims(y, axis=1)
    # x_test = np.stack([x_test, x_test**2, x_test**3], axis=1)

    tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 1])  # [100, 1]
    tf_y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1])  # [100, 1]

    tf_x_norm = (tf_x - tf.constant(x_mean, dtype=tf.float32)) / tf.constant(x_var, dtype=tf.float32)

    is_training = tf.placeholder(tf.bool)
    global_step = tf.train.create_global_step()

    alg = 'nng'

    if alg == 'sgd':
        tf_y, weights = create_nn(tf_x_norm, hidden_units=16, dropout=False)
        tf_loss = tf.reduce_mean((tf_y_true - tf_y) ** 2)

        learning_rate = tf.train.exponential_decay(3e-2, global_step, 50, 0.1, staircase=True)
        train_step = get_sgd_optimizer(tf_loss, weights, learning_rate=learning_rate)

    elif alg == 'ng':
        tf_y, weights = create_nn(tf_x_norm, hidden_units=16, dropout=False)
        tf_loss = tf.reduce_mean((tf_y_true - tf_y) ** 2)

        # TODO: This uses the empirical Fisher which may be a bad approximation. Use true Fisher.
        learning_rate = tf.train.exponential_decay(3e-4, global_step, 50, 0.1, staircase=True)
        train_step = get_ng_optimizer(tf_loss, weights, learning_rate)

    elif alg == 'nng':
        alpha_ = tf.train.exponential_decay(1e-3, global_step, 200, 0.9, staircase=True)
        beta_ = 0.995
        kl_weight = 0.001
        prior_variance = 0.001
        external_damping = 0

        tf_y, weights = create_nn_mvg(tf_x_norm, hidden_units=16, N=N, kl_weight=kl_weight, prior_variance=prior_variance)
        tf_loss = -1. * tf.reduce_mean((tf_y_true - tf_y) ** 2)

        # TODO: This uses the empirical Fisher which may be a bad approximation. Use true Fisher.
        train_step = get_nng_optimizer(tf_loss, weights, alpha_, beta_, kl_weight, N, prior_variance, external_damping)

    elif alg == 'mc_dropout':
        dropout_rate = 0.1
        length_scale = 1.
        precision = 100.

        tf_y, weights = create_nn(tf_x_norm, hidden_units=16, dropout=True, dropout_rate=dropout_rate, is_training=is_training)
        tf_loss = tf.reduce_mean((tf_y_true - tf_y) ** 2)

        tf_l2_loss = 0.
        for w in weights:
            tf_l2_loss += tf.norm(w, ord=2)

        tf_l2_loss = length_scale ** 2 * (1 - dropout_rate) / 2. / N / precision * tf_l2_loss

        learning_rate = tf.train.exponential_decay(0.03, global_step, 100, 0.1, staircase=True)
        train_step = get_adam_optimizer(tf_loss + tf_l2_loss, weights, learning_rate=learning_rate)

    else:
        raise Exception('Algorithm not supported.')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_fig, loss_ax = plt.subplots()
        data_fig, data_ax = plt.subplots()
        loss_list = []

        for i in range(2000):
            [_, loss] = sess.run([train_step, tf_loss],
                                 feed_dict={tf_x_norm: x, tf_y_true: y, is_training: True})

            increment_global_step_op = tf.assign(global_step, global_step+1)
            sess.run(global_step)

            print('Step {} Loss: {}'.format(i, loss))
            loss_list.append(loss)

            loss_ax.cla()
            loss_ax.plot(range(i + 1)[-200:], loss_list[-200:])
            plt.draw()
            plt.pause(1.0 / 60.0)

            if alg not in ['nng', 'mc_dropout', 'bbb']:
                y_pred = sess.run(tf_y, feed_dict={tf_x_norm: x_test, is_training: False})

                data_ax.cla()
                data_ax.plot(x_test, y_pred[:, 0], 'r', lw=0.5)
                data_ax.plot(x[:, 0], y[:, 0], '.')
                data_ax.set_xlim([-7, 10])
                data_ax.set_ylim([-0.75, 0.75])
                data_ax.set_title('Step: {}'.format(i))
                plt.draw()
                plt.pause(1.0 / 60.0)

            if alg in ['nng', 'mc_dropout', 'bbb']:
                y_pred_list = []
                for _ in range(50):
                    if alg == 'mc_dropout':
                        y_pred = sess.run(tf_y, feed_dict={tf_x_norm: x_test, is_training: True})
                    else:
                        y_pred = sess.run(tf_y, feed_dict={tf_x_norm: x_test, is_training: False})

                    y_pred_list.append(y_pred)

                y_pred_mean = np.mean(y_pred_list, axis=0).squeeze(axis=-1)
                y_pred_std = np.std(y_pred_list, axis=0).squeeze(axis=-1)
                x_plot = x_test.squeeze(axis=-1)

                data_ax.cla()
                data_ax.plot(x_plot, y_pred_mean, 'r', lw=0.5)
                data_ax.fill_between(x_plot, y_pred_mean - 0.5 * y_pred_std, y_pred_mean + 0.5 * y_pred_std, alpha=0.1, facecolor='r')
                data_ax.fill_between(x_plot, y_pred_mean - 1.0 * y_pred_std, y_pred_mean + 1.0 * y_pred_std, alpha=0.1, facecolor='r')
                data_ax.fill_between(x_plot, y_pred_mean - 1.5 * y_pred_std, y_pred_mean + 1.5 * y_pred_std, alpha=0.1, facecolor='r')
                data_ax.fill_between(x_plot, y_pred_mean - 2.0 * y_pred_std, y_pred_mean + 2.0 * y_pred_std, alpha=0.1, facecolor='r')
                data_ax.plot(x[:, 0], y[:, 0], '.')
                data_ax.set_xlim([-7, 10])
                data_ax.set_ylim([-0.75, 0.75])
                data_ax.set_title('Step: {}'.format(i))
                plt.draw()
                plt.pause(1.0 / 60.0)
