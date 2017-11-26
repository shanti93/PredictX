import os

import numpy as np
import tensorflow as tf

from data_frame import DataFrame
from tf_base_model import TFBaseModel
from tf_utils import (
    timeDistributedDenseLayer, convolutionLayer,
    sequence_mean, sequence_smape, shape
)


class DataReader(object):

    def __init__(self, data_dir):
        data_cols = [
            'data',
            'is_nan',
            'page_id',
            'project',
            'access',
            'agent',
            'test_data',
            'test_is_nan'
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95)

        print 'train size', len(self.train_df)
        print 'val size', len(self.val_df)
        print 'test size', len(self.test_df)

    def train_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=10000,
            is_test=False
        )

    def test_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            is_test=True
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, is_test=False):
        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=is_test
        )
        data_col = 'test_data' if is_test else 'data'
        is_nan_col = 'test_is_nan' if is_test else 'is_nan'
        for batch in batch_gen:
            decodeCount = 64
            full_seq_len = batch[data_col].shape[1]
            max_encode_length = full_seq_len - decodeCount if not is_test else full_seq_len

            x_encode = np.zeros([len(batch), max_encode_length])
            y_decode = np.zeros([len(batch), decodeCount])
            is_nan_encode = np.zeros([len(batch), max_encode_length])
            is_nan_decode = np.zeros([len(batch), decodeCount])
            encode_len = np.zeros([len(batch)])
            decode_len = np.zeros([len(batch)])

            for i, (seq, nan_seq) in enumerate(zip(batch[data_col], batch[is_nan_col])):
                rand_len = np.random.randint(max_encode_length - 365 + 1, max_encode_length + 1)
                x_encode_len = max_encode_length if is_test else rand_len
                x_encode[i, :x_encode_len] = seq[:x_encode_len]
                is_nan_encode[i, :x_encode_len] = nan_seq[:x_encode_len]
                encode_len[i] = x_encode_len
                decode_len[i] = decodeCount
                if not is_test:
                    y_decode[i, :] = seq[x_encode_len: x_encode_len + decodeCount]
                    is_nan_decode[i, :] = nan_seq[x_encode_len: x_encode_len + decodeCount]

            batch['x_encode'] = x_encode
            batch['encode_len'] = encode_len
            batch['y_decode'] = y_decode
            batch['decode_len'] = decode_len
            batch['is_nan_encode'] = is_nan_encode
            batch['is_nan_decode'] = is_nan_decode

            yield batch


class cnn(TFBaseModel):

     def __init__(
        self,
        #residual channels being used. For an image we would have 3 channels, RGB
        num_of_residual_channels=32,
        #skip chanels are extra connections added between layers skipping some of the layers in between
        num_of_skip_channels=32,
        #dilationsCount is a hyper parameter introduced in the recent developement. here we choose dilation to be 1,
        #so it is skipping one level,
        #hence filter would be like x[0],x[2],x[4]...
        dilationsCount=[2**i for i in range(8)]*3,
        #filter width is 2*3 for all the filters
        widthOfFilter=[2 for i in range(8)]*3,
        decodeCount=64,
        #?? what is this?
        **kwargs
    ):
        self.num_of_residual_channels = num_of_residual_channels
        self.num_of_skip_channels = num_of_skip_channels
        self.dilationsCount = dilationsCount
        self.widthOfFilter = widthOfFilter
        self.decodeCount = decodeCount
        super(cnn, self).__init__(**kwargs)

    #difference of log(value+1) and log_encode_mean
    def transformFunction(self, x):
        return tf.log(x + 1) - tf.expand_dims(self.log_x_encode_mean, 1)

    def inverse_transformFunction(self, x):
        return tf.exp(x + tf.expand_dims(self.log_x_encode_mean, 1)) - 1


#placeholder is something that should always be fed with data when executed. When code is running at each step, data will be provided to placeholder.
    #format : placeholder( dtype, shape = None,name=None)
    #here we are inserting placeholder for tensor for which code has to feed data upon execution.
    def getData(self):
        self.x_encode = tf.placeholder(tf.float32, [None, None])
        self.encode_len = tf.placeholder(tf.int32, [None])
        self.y_decode = tf.placeholder(tf.float32, [None, self.decodeCount])
        self.decode_len = tf.placeholder(tf.int32, [None])
        self.is_nan_encode = tf.placeholder(tf.float32, [None, None])
        self.is_nan_decode = tf.placeholder(tf.float32, [None, self.decodeCount])

        self.page_id = tf.placeholder(tf.int32, [None])
        self.project = tf.placeholder(tf.int32, [None])
        self.access = tf.placeholder(tf.int32, [None])
        self.agent = tf.placeholder(tf.int32, [None])

        self.keepProbability = tf.placeholder(tf.float32)
        #to know if we are training on data.
        self.is_training = tf.placeholder(tf.bool)

        #log_x_encode_mean is log(mean of encode sequence)
        self.log_x_encode_mean = sequence_mean(tf.log(self.x_encode + 1), self.encode_len)

        #computing transformFunction on each encode input value
        self.log_x_encode = self.transformFunction(self.x_encode)


        #expanding dimension by 2
        self.x = tf.expand_dims(self.log_x_encode, 2)

#concat is to concatenate all these vectors around the axis. here axis is 2 which implies that this axis  is  2 dimensional
        #expand_dims adds an additional dimension 1 to the shape -- returns the given input but with an extra dimension.
        self.encodeTensorFeatures = tf.concat([
            tf.expand_dims(self.is_nan_encode, 2),
            #tf.cast to cast a tensor to required type.
            tf.expand_dims(tf.cast(tf.equal(self.x_encode, 0.0), tf.float32), 2),
            #-1 is to flatten.
            tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, tf.shape(self.x_encode)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, tf.shape(self.x_encode)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, tf.shape(self.x_encode)[1], 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, tf.shape(self.x_encode)[1], 1)),
        ], axis=2)

        decode_idx = tf.tile(tf.expand_dims(tf.range(self.decodeCount), 0), (tf.shape(self.y_decode)[0], 1))
        self.decodeTensorFeatures = tf.concat([
            tf.one_hot(decode_idx, self.decodeCount),
            tf.tile(tf.reshape(self.log_x_encode_mean, (-1, 1, 1)), (1, self.decodeCount, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.project, 9), 1), (1, self.decodeCount, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.access, 3), 1), (1, self.decodeCount, 1)),
            tf.tile(tf.expand_dims(tf.one_hot(self.agent, 2), 1), (1, self.decodeCount, 1)),
        ], axis=2)

        return self.x

    def encode(self, x, features):
        x = tf.concat([x, features], axis=2)

        inputs = timeDistributedDenseLayer(
            inputs=x,
            output_units=self.num_of_residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-encode'
        )

        skipChannels = []
        convolutionInputs = [inputs]
        for i, (dilation, widthOfFilter) in enumerate(zip(self.dilationsCount, self.widthOfFilter)):
            dilatedConvolution = convolutionLayer(
                inputs=inputs,
                output_units=2*self.num_of_residual_channels,
                convolution_width=widthOfFilter,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-encode-{}'.format(i)
            )
            filterConvolution, gateConvolution = tf.split(dilatedConvolution, 2, axis=2)
            dilatedConvolution = tf.nn.tanh(filterConvolution)*tf.nn.sigmoid(gateConvolution)

            outputs = timeDistributedDenseLayer(
                inputs=dilatedConvolution,
                output_units=self.num_of_skip_channels + self.num_of_residual_channels,
                scope='dilated-conv-proj-encode-{}'.format(i)
            )
            skips, residuals = tf.split(outputs, [self.num_of_skip_channels, self.num_of_residual_channels], axis=2)

            inputs += residuals
            convolutionInputs.append(inputs)
            skipChannels.append(skips)

        skipChannels = tf.nn.relu(tf.concat(skipChannels, axis=2))
        h = timeDistributedDenseLayer(
            skipChannels, 128,
            scope='dense-encode-1',
            activation=tf.nn.relu
        )
        y_hat = timeDistributedDenseLayer(h, 1, scope='dense-encode-2')

        return y_hat, convolutionInputs[:-1]

    def decodeParametersInitialize(self, x, features):
        x = tf.concat([x, features], axis=2)

        inputs = timeDistributedDenseLayer(
            inputs=x,
            output_units=self.num_of_residual_channels,
            activation=tf.nn.tanh,
            scope='x-proj-decode'
        )

        skipChannels = []
        convolutionInputs = [inputs]
        for i, (dilation, widthOfFilter) in enumerate(zip(self.dilationsCount, self.widthOfFilter)):
            dilatedConvolution = convolutionLayer(
                inputs=inputs,
                output_units=2*self.num_of_residual_channels,
                convolution_width=widthOfFilter,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-decode-{}'.format(i)
            )
            filterConvolution, gateConvolution = tf.split(dilatedConvolution, 2, axis=2)
            dilatedConvolution = tf.nn.tanh(filterConvolution)*tf.nn.sigmoid(gateConvolution)

            outputs = timeDistributedDenseLayer(
                inputs=dilatedConvolution,
                output_units=self.num_of_skip_channels + self.num_of_residual_channels,
                scope='dilated-conv-proj-decode-{}'.format(i)
            )
            skips, resid = tf.split(outputs, [self.num_of_skip_channels, self.num_of_residual_channels], axis=2)

            inputs += resid
            convolutionInputs.append(inputs)
            skipChannels.append(skips)

        skipChannels = tf.nn.relu(tf.concat(skipChannels, axis=2))
        h = timeDistributedDenseLayer(skipChannels, 128, scope='dense-decode-1', activation=tf.nn.relu)
        y_hat = timeDistributedDenseLayer(h, 1, scope='dense-decode-2')
        return y_hat

    def decode(self, x, convolutionInputs, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation) in enumerate(zip(convolutionInputs, self.dilations)):
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation))
            batch_idx = tf.reshape(batch_idx, [-1])

            queue_begin_time = self.encode_len - dilation - 1
            Indextemp = tf.expand_dims(queue_begin_time, 1) + tf.expand_dims(tf.range(dilation), 0)
            Indextemp = tf.reshape(Indextemp, [-1])


            idx = tf.stack([batch_idx, Indextemp], axis=1)

            #collects all slices from conv_input within specified index into tensor of shape as indicated
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation + self.decodeCount)
            #unpacks the tensor into individual tensors
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        features_ta = tf.TensorArray(dtype=tf.float32, size=self.decodeCount)
        features_ta = features_ta.unstack(tf.transpose(features, (1, 0, 2)))

        # initialize output tensor array
        FinalemittedArray = tf.TensorArray(size=self.decodeCount, dtype=tf.float32)

        # initialize other loop vars
        finishedElements = 0 >= self.decode_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        current_idx = tf.stack([tf.range(tf.shape(self.encode_len)[0]), self.encode_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)

        def loopfunction(time, current_input, queues):
            current_features = features_ta.read(time)
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('x-proj-decode', reuse=True):
                w_xProjection = tf.get_variable('weights')
                b_xProjection = tf.get_variable('biases')
                #calcluating feature map at every level
                # This is obtained by doing convolution on input image on sub regions with filter
                # and adding bias and applying non linear filter function.
                xProjection = tf.nn.tanh(tf.matmul(current_input, w_xProjection) + b_xProjection)

                skipChannels, updated_queues = [], []
            for i, (conv_input, queue, dilation) in enumerate(zip(convolutionInputs, queues, self.dilationsCount)):

                state = queue.read(time)
                with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights'.format(i))
                    b_conv = tf.get_variable('biases'.format(i))
                    #doing dilated convolution at every point
                    dilatedConvolution = tf.matmul(state, w_conv[0, :, :]) + tf.matmul(xProjection, w_conv[1, :, :]) + b_conv
                filterConvolution, gateConvolution = tf.split(dilatedConvolution, 2, axis=1)
                #applying tanh to get feature map
                dilatedConvolution = tf.nn.tanh(filterConvolution)*tf.nn.sigmoid(gateConvolution)

                with tf.variable_scope('dilated-conv-proj-decode-{}'.format(i), reuse=True):
                    wProjection = tf.get_variable('weights'.format(i))
                    bProjection = tf.get_variable('biases'.format(i))
                    #final convolution
                    concat_outputs = tf.matmul(dilatedConvolution, wProjection) + bProjection
                skips, residuals = tf.split(concat_outputs, [self.num_of_skip_channels, self.num_of_residual_channels], axis=1)

                xProjection += residuals
                skipChannels.append(skips)
                updated_queues.append(queue.write(time + dilation, xProjection))

            skipChannels = tf.nn.relu(tf.concat(skipChannels, axis=1))
            with tf.variable_scope('dense-decode-1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                #doing convolution on skip outputs
                h = tf.nn.relu(tf.matmul(skipChannels, w_h) + b_h)

            with tf.variable_scope('dense-decode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                #final convolution
                y_hat = tf.matmul(h, w_y) + b_y

            finishedElements = (time >= self.decode_len)
            finished = tf.reduce_all(finishedElements)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 1], dtype=tf.float32),
                lambda: y_hat
            )
            next_finishedElements = (time >= self.decode_len - 1)

            return (next_finishedElements, next_input, updated_queues)

        def condition(unused_time, finishedElements, *_):
            return tf.logical_not(tf.reduce_all(finishedElements))

        def body(time, finishedElements, FinalemittedArray, *state_queues):
            (next_finished, FinalemittedOutput, state_queues) = loopfunction(time, initial_input, state_queues)

            emit = tf.where(finishedElements, tf.zeros_like(FinalemittedOutput), FinalemittedOutput)
            FinalemittedArray = FinalemittedArray.write(time, emit)

            finishedElements = tf.logical_or(finishedElements, next_finished)
            return [time + 1, finishedElements, FinalemittedArray] + list(state_queues)

        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, finishedElements, FinalemittedArray] + state_queues
        )

        outputs_ta = returned[2]
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
        return y_hat

    def calculate_loss(self):
        x = self.getData()

        y_hat_encode, convolutionInputs = self.encode(x, features=self.encodeTensorFeatures)
        self.decodeParametersInitialize(x, features=self.decodeTensorFeatures)
        y_decodedhat = self.decode(y_hat_encode, convolutionInputs, features=self.decodeTensorFeatures)
        y_decodedhat = self.inverse_transformFunction(tf.squeeze(y_decodedhat, 2))
        y_decodedhat = tf.nn.relu(y_decodedhat)

        self.labels = self.y_decode
        self.preds = y_decodedhat
        self.loss = sequence_smape(self.labels, self.preds, self.decode_len, self.is_nan_decode)

        self.prediction_tensors = {
            'priors': self.x_encode,
            'labels': self.labels,
            'preds': self.preds,
            'page_id': self.page_id,
        }

        return self.loss


if __name__ == '__main__':
    base_dir = './'

    dr = DataReader(data_dir=os.path.join(base_dir, 'data/processed/'))

    nn = cnn(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.001,
        batch_size=128,
        num_training_steps=200000,
        early_stopping_steps=5000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keepProbability=1.0,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=500,
        log_interval=10,
        num_validation_batches=1,
        grad_clip=20,
        num_of_residual_channels=32,
        num_of_skip_channels=32,
        dilationsCount=[2**i for i in range(8)]*3,
        widthOfFilter=[2 for i in range(8)]*3,
        decodeCount=64,
    )
    nn.fit()
    nn.restore()
    nn.predict()

