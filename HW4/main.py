from gan_model import create_discriminator, create_gan, create_generator
from tqdm import tqdm
import pickle
from utils import *


def part_a_train(args):
    import os
    import datetime
    import statistics
    tf.get_logger().setLevel('ERROR')

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f'./models/part_a/{time}'
    os.makedirs(dir)

    print(f'TF executing_eagerly: {tf.executing_eagerly()}')
    batch_size, epochs, lr, dim, optimizer, dp = handle_args_for_train(args)

    x, y = pre_process_arff_data(filepath=dp)
    total_samples, real_dim = x.shape
    x_train, x_test, y_train, y_test = split(x, y)
    gan_input_shape = 20

    # Creating the model
    generator = create_generator(batch_size, input_shape=gan_input_shape, dim=dim, real_dim=real_dim)
    discriminator = create_discriminator(batch_size, input_shape=real_dim, dim=dim)
    gan = create_gan(discriminator, generator, gan_input_shape)

    # compile the model
    # keras knows to choose binary-accuracy
    discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=create_optimizer(lr),
                          metrics=['accuracy'])
    gan.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=create_optimizer(lr))

    gan.summary()

    d_loss = []
    d_acc = []
    g_loss = []
    boost_generator = True

    for epoch in range(1, epochs + 1):
        batch_indices = generate_batch_indices(x_train.shape[0], batch_size)
        bar = tqdm(enumerate(batch_indices))
        d_epoch_loss = []
        d_epoch_acc = []
        g_epoch_loss = []
        if epoch % 10:
            boost_generator = not boost_generator

        for step, indices in bar:
            bar.set_description(f"Epoch: {epoch}")
            # generate noise
            noise = np.random.normal(0, 1, size=[batch_size, gan_input_shape])

            generated_rows = generator.predict(noise)  # (batch_size, real_dim)

            rows_batch = x_train[indices]

            x = np.vstack([rows_batch, generated_rows])

            # label start definition: 1 - real, 0 - fake
            y_disc = np.zeros(2 * batch_size)

            y_disc[:batch_size] = 0.9

            # train discriminator
            discriminator.trainable = True
            d_cur_loss, d_cur_acc = discriminator.train_on_batch(x, y_disc)

            d_epoch_loss.append(d_cur_loss)
            d_epoch_acc.append(d_cur_acc)

            # generate noise
            noise = np.random.normal(0, 1, size=[batch_size, gan_input_shape])

            if boost_generator:
                y_gen = np.ones(batch_size)  # Flip labels to trick the discriminator
            else:
                y_gen = np.zeros(batch_size)

            # lock discriminator
            discriminator.trainable = False

            # train generator
            g_cur_loss = gan.train_on_batch(noise, y_gen)

            g_epoch_loss.append(g_cur_loss)

            bar.set_postfix(step=step, d_loss=d_cur_loss, d_acc=100 * d_cur_acc, g_loss=g_cur_loss)

        generator.save(dir + f'/epoch_{epoch}/generator/gen_model')
        discriminator.save(dir + f'/epoch_{epoch}/discriminator/disc_model')

        d_loss.append(statistics.mean(d_epoch_loss))
        d_acc.append(statistics.mean(d_epoch_acc))
        g_loss.append(statistics.mean(g_epoch_loss))

    plot_training_stats(d_loss, d_acc, g_loss)

    analyse_gan_part_a(generator, discriminator, gan, x_test, y_test)

    print()


def analyse_gan_part_a(generator, discriminator, gan, x_test, y_test):
    """
    Analyse the generator and discriminator by:
    1. Projecting 100 randomly generated samples into 2-d plane.
    2. Measuring distance between a sample from the test and 10 generated samples.
    3. Compute the number of fooled generated samples.

    :param generator:
    :param discriminator:
    :param gan:
    :param x_test:
    :param y_test:
    :return:
    """
    from utils import project_samples
    num_samples, real_dim = x_test.shape
    gan_input_shape = 20

    noise = np.random.normal(0, 1, size=[100, gan_input_shape])
    generated_rows = generator.predict_on_batch(noise)

    project_samples(generated_rows)

    subset = generated_rows[:10]
    distances = [np.linalg.norm(x_test[0] - x) for x in subset]
    print(f'Measuring euclidean distance from a real sample: {list(x_test[0])}')
    for gen_sample, dist in zip(list(subset), distances):
        print(f'Distance: {dist}, sample: {list(gen_sample)}')
    print('=' * 80)

    predicted_on_generated = discriminator.predict_on_batch(generated_rows)

    num_fooled = sum(predicted_on_generated > 0.5)

    subset_fooled = generated_rows[np.squeeze(predicted_on_generated > 0.5), :][:5]

    print(f'Num of fooled generated samples: {num_fooled}')

    print(f'Subset fooled: \n {list(subset_fooled)}')

    print()


def train_classifier(args):
    import os
    import datetime
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import plot_roc_curve
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f'./models/part_b_classifier/{time}'
    os.makedirs(dir)
    datapath = args.dp
    x, y = pre_process_arff_data(filepath=datapath)
    y = np.squeeze(y)
    total_samples, real_dim = x.shape
    x_train, x_test, y_train, y_test = split(x, y, percent=0.3)

    # Training
    print(f'Start training classifier')
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    y_pred = model.predict_proba(x_test)
    print(f'Final acc on test: {acc * 100}%')

    plot_score_dist(y_pred)
    plot_roc_curve(model, x_test, y_test)
    filename = dir + '/classifier.sav'
    plt.savefig(dir + '/classifier.png')
    pickle.dump(model, open(filename, 'wb'))
    plt.show()
    return model, total_samples, real_dim


def prepare_disc_entry(clf, generated_samples, C, generator=False, trick=True):
    """
    Given a sklearn classifier, prepares the discriminator entry.
    :return:
    """
    batch_size = generated_samples.shape[0]  # assumes batch_size first

    Y_probs = clf.predict_proba(generated_samples)  # (batch_size, 2)

    Y = Y_probs[:, 1]  # (batch_size, 1) - taking the positive class score

    Y = np.expand_dims(Y, axis=-1)
    # Define the task of the discriminator:
    # 1 is the true label iff the true classification score is at index 1

    C_tensor = tf.convert_to_tensor(C, dtype=tf.float32)
    Y_tensor = tf.convert_to_tensor(Y, dtype=tf.float32)

    if generator:
        if trick:
            x = tf.concat([generated_samples, Y_tensor, C_tensor], axis=1)
            y = np.ones(batch_size)  # true classification is on index 0
            y_smooth = y
        else:
            x = tf.concat([generated_samples, Y_tensor, C_tensor], axis=1)
            y = np.zeros(batch_size)  # true classification is on index 0
            y_smooth = y
    else:
        if np.random.random() > 0.5:
            x = tf.concat([generated_samples, C_tensor, Y_tensor], axis=1)
            y = np.ones(batch_size)  # true classification is on index 1
            y_smooth = np.zeros(batch_size)
            y_smooth[:] = 0.8  # one sided label smoothing
        else:
            x = tf.concat([generated_samples, Y_tensor, C_tensor], axis=1)
            y = np.zeros(batch_size)  # true classification is on index 0
            y_smooth = y

    y = tf.convert_to_tensor(y, dtype=tf.float32)
    y_smooth = tf.convert_to_tensor(y_smooth, dtype=tf.float32)

    return x, y, y_smooth


def part_b_train(args):
    import os
    import datetime
    import statistics
    tf.get_logger().setLevel('ERROR')

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir = f'./models/part_b/{time}'
    os.makedirs(dir)
    classifier, total_samples, real_dim = train_classifier(args)

    batch_size, epochs, lr, dim, optimizer, dp = handle_args_for_train(args)
    gan_noise_shape = 20
    gan_input_shape = gan_noise_shape + 1  # the +1 is for the constant C - the wanted classification score

    # Creating the model
    generator = create_generator(batch_size, input_shape=gan_input_shape, dim=dim, real_dim=real_dim)
    discriminator = create_discriminator(batch_size, input_shape=real_dim + 2, dim=dim, minibatch_disc=True)
    # gan = create_gan(discriminator, generator, gan_input_shape)

    generator.summary()
    discriminator.summary()

    d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1 * lr, beta_1=0.5)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

    loss_fn = tf.keras.losses.BinaryCrossentropy()
    acc_fn = tf.keras.metrics.BinaryAccuracy()

    def train_step(trick_disc, epoch):
        """
        The training step defined for the gan architecture
        :return:
        """

        minval, maxval = 0.05, 0.8

        # generate noise
        noise = tf.random.normal(shape=[batch_size, gan_noise_shape], mean=0, stddev=1)
        # desired classification score for positive class
        C = tf.random.uniform(shape=[batch_size, 1], minval=minval, maxval=maxval)

        # train discriminator
        with tf.GradientTape() as tape:
            Z_C = tf.concat([noise, C], axis=1)
            generated_rows = generator(Z_C)  # (batch_size, real_dim)
            generated_rows += tf.random.normal(shape=generated_rows.shape, mean=0, stddev=0.2 * (1 / epoch))
            x, y_disc, y_disc_smooth = prepare_disc_entry(classifier, generated_rows, C)
            y_pred = discriminator(x)
            d_cur_loss = loss_fn(y_disc_smooth, y_pred)
            acc_fn.update_state(y_disc, y_pred)  # use the un-smoothed version

        # lock generator - UPDATE only discriminator.
        grads = tape.gradient(d_cur_loss, discriminator.trainable_weights)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

        # generate noise
        noise = tf.random.normal(shape=[batch_size, gan_noise_shape], mean=0, stddev=1)
        # desired classification score for positive class
        C = tf.random.uniform(shape=[batch_size, 1], minval=minval, maxval=maxval)

        # train generator
        with tf.GradientTape() as tape:
            Z_C = tf.concat([noise, C], axis=1)
            generated_rows = generator(Z_C)
            x, y_gen, _ = prepare_disc_entry(classifier, generated_rows, C, generator=True, trick=trick_disc)
            y_pred = discriminator(x)
            g_cur_loss = loss_fn(y_gen, y_pred)

        # lock discriminator - UPDATE only generator.
        grads = tape.gradient(g_cur_loss, generator.trainable_weights)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

        return d_cur_loss, g_cur_loss

    d_loss = []
    d_acc = []
    g_loss = []
    num_of_iterations = 100
    trick_disc = True
    for epoch in range(1, epochs + 1):
        bar = tqdm(enumerate(list(range(num_of_iterations))))
        bar.total = num_of_iterations
        d_epoch_loss = []
        g_epoch_loss = []
        acc_fn.reset_states()  # reset accuracy metric every epoch

        for step, _ in bar:
            bar.set_description(f"Epoch: {epoch}")

            d_cur_loss_t, g_cur_loss_t = train_step(trick_disc, epoch)

            d_cur_loss = d_cur_loss_t.numpy()
            g_cur_loss = g_cur_loss_t.numpy()

            d_epoch_loss.append(d_cur_loss)
            g_epoch_loss.append(g_cur_loss)

            bar.set_postfix(step=step, d_loss=d_cur_loss, d_acc=100 * acc_fn.result().numpy(), g_loss=g_cur_loss)

        generator.save(dir + f'/epoch_{epoch}/generator/gen_model')
        discriminator.save(dir + f'/epoch_{epoch}/discriminator/disc_model')

        d_loss.append(statistics.mean(d_epoch_loss))
        d_acc.append(acc_fn.result().numpy())
        g_loss.append(statistics.mean(g_epoch_loss))

    plot_training_stats(d_loss, d_acc, g_loss)

    analyse_gan_part_b(generator, discriminator, classifier)

    print()


def analyse_gan_part_b(generator, discriminator, classifier):
    """
    Analyse the generator and the classifier by:
    1. Projecting 1000 randomly generated samples with desired confidence of drawn from [0,1]
    2. Plot the classifier score distribution for the generated samples.
    :param generator:
    :param discriminator:
    :param classifier:
    :return:
    """
    from utils import project_samples
    gan_noise_shape = 20
    gan_input_shape = gan_noise_shape + 1  # the +1 is for the constant C - the wanted classification score
    num_samples = 1000
    noise = tf.random.normal(shape=[num_samples, gan_noise_shape], mean=0, stddev=1)
    C = tf.random.uniform(shape=[num_samples, 1], minval=0, maxval=1)
    Z_C = tf.concat([noise, C], axis=1)

    generated_samples = generator.predict_on_batch(Z_C)

    project_samples(generated_samples)

    y_pred = classifier.predict_proba(generated_samples)

    plot_score_dist(y_pred)

    print()


if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(prog='main.py', description='GAN Sampler')
        parser.add_argument('-q', '--question', dest='question', action='store', type=int,
                            help='Which question to run (note: 1 - part A, 2 - part B,  3 - refers to train a '
                                 'classifier individually)', default=1, choices=[1, 2, 3])
        parser.add_argument('-e', '--epochs', dest='epochs', action='store', type=int, default=100,
                            help='How many epochs the network will train')
        parser.add_argument('-opt', '--optimizer', dest='optimizer', metavar='optimizer', default='adam', type=str,
                            choices=['adam'],
                            help='The optimizer used for training')
        parser.add_argument('-b', '--batch_size', dest='batch_size', metavar='BATCH_SIZE', default=64, type=int,
                            help='the batch size used in train')
        parser.add_argument('-dim', '--dimension', dest='dim', metavar='DIMENSION', default=64, type=int,
                            help='The leading dimension in the GAN model')
        parser.add_argument('-dp', '--dataset_path', dest='dp', type=str, help='full path to the dataset file',
                            default='diabetes.arff')
        parser.add_argument('-lr', '--learning_rate', dest='lr', metavar='LEARNING_RATE', type=float,
                            default=1e-4, help='the learning rate used in training')
        args = parser.parse_args()

        return args


    questions = {1: part_a_train, 2: part_b_train, 3: train_classifier}

    args = parse_args()
    q = args.question
    if q in questions:
        func = questions[q]
        func(args)
    else:
        print(f'Wrong choice of question to run')
