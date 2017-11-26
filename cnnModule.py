{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf100
{\fonttbl\f0\fnil\fcharset0 Menlo-Bold;\f1\fnil\fcharset0 Menlo-Regular;\f2\fnil\fcharset0 Menlo-Italic;
}
{\colortbl;\red255\green255\blue255;\red0\green0\blue109;\red160\green0\blue163;\red128\green63\blue122;
\red15\green112\blue3;\red82\green0\blue135;\red0\green0\blue254;\red109\green109\blue109;}
{\*\expandedcolortbl;;\csgenericrgb\c0\c0\c42745;\csgenericrgb\c62745\c0\c63922;\csgenericrgb\c50196\c24706\c47843;
\csgenericrgb\c5882\c43922\c1176;\csgenericrgb\c32157\c0\c52941;\csgenericrgb\c0\c0\c99608;\csgenericrgb\c42745\c42745\c42745;}
\margl1440\margr1440\vieww18160\viewh13620\viewkind0
\pard\tx560\tx1120\tx1680\tx2240\tx2800\tx3360\tx3920\tx4480\tx5040\tx5600\tx6160\tx6720\pardirnatural\partightenfactor0

\f0\b\fs24 \cf2 import 
\f1\b0 \cf0 os\
\

\f0\b \cf2 import 
\f1\b0 \cf0 numpy 
\f0\b \cf2 as 
\f1\b0 \cf0 np\

\f0\b \cf2 import 
\f1\b0 \cf0 tensorflow 
\f0\b \cf2 as 
\f1\b0 \cf0 tf\
\

\f0\b \cf2 from 
\f1\b0 \cf0 data_frame 
\f0\b \cf2 import 
\f1\b0 \cf0 DataFrame\

\f0\b \cf2 from 
\f1\b0 \cf0 tf_base_model 
\f0\b \cf2 import 
\f1\b0 \cf0 TFBaseModel\

\f0\b \cf2 from 
\f1\b0 \cf0 tf_utils 
\f0\b \cf2 import 
\f1\b0 \cf0 (\
    timeDistributedDenseLayer, convolutionLayer,\
    sequence_mean, sequence_smape, shape\
)\
\
\

\f0\b \cf2 class 
\f1\b0 \cf0 DataReader(\cf2 object\cf0 ):\
\
    
\f0\b \cf2 def 
\f1\b0 \cf3 __init__\cf0 (\cf4 self\cf0 , data_dir):\
        data_cols = [\
            
\f0\b \cf5 'data'
\f1\b0 \cf0 ,\
            
\f0\b \cf5 'is_nan'
\f1\b0 \cf0 ,\
            
\f0\b \cf5 'page_id'
\f1\b0 \cf0 ,\
            
\f0\b \cf5 'project'
\f1\b0 \cf0 ,\
            
\f0\b \cf5 'access'
\f1\b0 \cf0 ,\
            
\f0\b \cf5 'agent'
\f1\b0 \cf0 ,\
            
\f0\b \cf5 'test_data'
\f1\b0 \cf0 ,\
            
\f0\b \cf5 'test_is_nan'\
        
\f1\b0 \cf0 ]\
        data = [np.load(os.path.join(data_dir, 
\f0\b \cf5 '\{\}.npy'
\f1\b0 \cf0 .format(i))) 
\f0\b \cf2 for 
\f1\b0 \cf0 i 
\f0\b \cf2 in 
\f1\b0 \cf0 data_cols]\
\
        \cf4 self\cf0 .test_df = DataFrame(\cf6 columns\cf0 =data_cols, \cf6 data\cf0 =data)\
        \cf4 self\cf0 .train_df, \cf4 self\cf0 .val_df = \cf4 self\cf0 .test_df.train_test_split(\cf6 train_size\cf0 =\cf7 0.95\cf0 )\
\
        
\f0\b \cf2 print \cf5 'train size'
\f1\b0 \cf0 , \cf2 len\cf0 (\cf4 self\cf0 .train_df)\
        
\f0\b \cf2 print \cf5 'val size'
\f1\b0 \cf0 , \cf2 len\cf0 (\cf4 self\cf0 .val_df)\
        
\f0\b \cf2 print \cf5 'test size'
\f1\b0 \cf0 , \cf2 len\cf0 (\cf4 self\cf0 .test_df)\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 train_batch_generator(\cf4 self\cf0 , batch_size):\
        
\f0\b \cf2 return 
\f1\b0 \cf4 self\cf0 .batch_generator(\
            \cf6 batch_size\cf0 =batch_size,\
            \cf6 df\cf0 =\cf4 self\cf0 .train_df,\
            \cf6 shuffle\cf0 =\cf2 True\cf0 ,\
            \cf6 num_epochs\cf0 =\cf7 10000\cf0 ,\
            \cf6 is_test\cf0 =\cf2 False\
        \cf0 )\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 val_batch_generator(\cf4 self\cf0 , batch_size):\
        
\f0\b \cf2 return 
\f1\b0 \cf4 self\cf0 .batch_generator(\
            \cf6 batch_size\cf0 =batch_size,\
            \cf6 df\cf0 =\cf4 self\cf0 .val_df,\
            \cf6 shuffle\cf0 =\cf2 True\cf0 ,\
            \cf6 num_epochs\cf0 =\cf7 10000\cf0 ,\
            \cf6 is_test\cf0 =\cf2 False\
        \cf0 )\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 test_batch_generator(\cf4 self\cf0 , batch_size):\
        
\f0\b \cf2 return 
\f1\b0 \cf4 self\cf0 .batch_generator(\
            \cf6 batch_size\cf0 =batch_size,\
            \cf6 df\cf0 =\cf4 self\cf0 .test_df,\
            \cf6 shuffle\cf0 =\cf2 True\cf0 ,\
            \cf6 num_epochs\cf0 =\cf7 1\cf0 ,\
            \cf6 is_test\cf0 =\cf2 True\
        \cf0 )\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 batch_generator(\cf4 self\cf0 , batch_size, df, shuffle=\cf2 True\cf0 , num_epochs=\cf7 10000\cf0 , is_test=\cf2 False\cf0 ):\
        batch_gen = df.batch_generator(\
            \cf6 batch_size\cf0 =batch_size,\
            \cf6 shuffle\cf0 =shuffle,\
            \cf6 num_epochs\cf0 =num_epochs,\
            \cf6 allow_smaller_final_batch\cf0 =is_test\
        )\
        data_col = 
\f0\b \cf5 'test_data' \cf2 if 
\f1\b0 \cf0 is_test 
\f0\b \cf2 else \cf5 'data'\
        
\f1\b0 \cf0 is_nan_col = 
\f0\b \cf5 'test_is_nan' \cf2 if 
\f1\b0 \cf0 is_test 
\f0\b \cf2 else \cf5 'is_nan'\
        \cf2 for 
\f1\b0 \cf0 batch 
\f0\b \cf2 in 
\f1\b0 \cf0 batch_gen:\
            decodeCount = \cf7 64\
            \cf0 full_seq_len = batch[data_col].shape[\cf7 1\cf0 ]\
            max_encode_length = full_seq_len - decodeCount 
\f0\b \cf2 if not 
\f1\b0 \cf0 is_test 
\f0\b \cf2 else 
\f1\b0 \cf0 full_seq_len\
\
            x_encode = np.zeros([\cf2 len\cf0 (batch), max_encode_length])\
            y_decode = np.zeros([\cf2 len\cf0 (batch), decodeCount])\
            is_nan_encode = np.zeros([\cf2 len\cf0 (batch), max_encode_length])\
            is_nan_decode = np.zeros([\cf2 len\cf0 (batch), decodeCount])\
            encode_len = np.zeros([\cf2 len\cf0 (batch)])\
            decode_len = np.zeros([\cf2 len\cf0 (batch)])\
\
            
\f0\b \cf2 for 
\f1\b0 \cf0 i, (seq, nan_seq) 
\f0\b \cf2 in 
\f1\b0 enumerate\cf0 (\cf2 zip\cf0 (batch[data_col], batch[is_nan_col])):\
                rand_len = np.random.randint(max_encode_length - \cf7 365 \cf0 + \cf7 1\cf0 , max_encode_length + \cf7 1\cf0 )\
                x_encode_len = max_encode_length 
\f0\b \cf2 if 
\f1\b0 \cf0 is_test 
\f0\b \cf2 else 
\f1\b0 \cf0 rand_len\
                x_encode[i, :x_encode_len] = seq[:x_encode_len]\
                is_nan_encode[i, :x_encode_len] = nan_seq[:x_encode_len]\
                encode_len[i] = x_encode_len\
                decode_len[i] = decodeCount\
                
\f0\b \cf2 if not 
\f1\b0 \cf0 is_test:\
                    y_decode[i, :] = seq[x_encode_len: x_encode_len + decodeCount]\
                    is_nan_decode[i, :] = nan_seq[x_encode_len: x_encode_len + decodeCount]\
\
            batch[
\f0\b \cf5 'x_encode'
\f1\b0 \cf0 ] = x_encode\
            batch[
\f0\b \cf5 'encode_len'
\f1\b0 \cf0 ] = encode_len\
            batch[
\f0\b \cf5 'y_decode'
\f1\b0 \cf0 ] = y_decode\
            batch[
\f0\b \cf5 'decode_len'
\f1\b0 \cf0 ] = decode_len\
            batch[
\f0\b \cf5 'is_nan_encode'
\f1\b0 \cf0 ] = is_nan_encode\
            batch[
\f0\b \cf5 'is_nan_decode'
\f1\b0 \cf0 ] = is_nan_decode\
\
            
\f0\b \cf2 yield 
\f1\b0 \cf0 batch\
\
\

\f0\b \cf2 class 
\f1\b0 \cf0 cnn(TFBaseModel):\
\
     
\f0\b \cf2 def 
\f1\b0 \cf3 __init__\cf0 (\
        \cf4 self\cf0 ,\
        
\f2\i \cf8 #residual channels being used. For an image we would have 3 channels, RGB\
        
\f1\i0 \cf0 num_of_residual_channels=\cf7 32\cf0 ,\
        
\f2\i \cf8 #skip chanels are extra connections added between layers skipping some of the layers in between\
        
\f1\i0 \cf0 num_of_skip_channels=\cf7 32\cf0 ,\
        
\f2\i \cf8 #dilationsCount is a hyper parameter introduced in the recent developement. here we choose dilation to be 1,\
        #so it is skipping one level,\
        #hence filter would be like x[0],x[2],x[4]...\
        
\f1\i0 \cf0 dilationsCount=[\cf7 2\cf0 **i 
\f0\b \cf2 for 
\f1\b0 \cf0 i 
\f0\b \cf2 in 
\f1\b0 range\cf0 (\cf7 8\cf0 )]*\cf7 3\cf0 ,\
        
\f2\i \cf8 #filter width is 2*3 for all the filters\
        
\f1\i0 \cf0 widthOfFilter=[\cf7 2 
\f0\b \cf2 for 
\f1\b0 \cf0 i 
\f0\b \cf2 in 
\f1\b0 range\cf0 (\cf7 8\cf0 )]*\cf7 3\cf0 ,\
        decodeCount=\cf7 64\cf0 ,\
        
\f2\i \cf8 #?? what is this?\
        
\f1\i0 \cf0 **kwargs\
    ):\
        \cf4 self\cf0 .num_of_residual_channels = num_of_residual_channels\
        \cf4 self\cf0 .num_of_skip_channels = num_of_skip_channels\
        \cf4 self\cf0 .dilationsCount = dilationsCount\
        \cf4 self\cf0 .widthOfFilter = widthOfFilter\
        \cf4 self\cf0 .decodeCount = decodeCount\
        \cf2 super\cf0 (cnn, \cf4 self\cf0 ).\cf3 __init__\cf0 (**kwargs)\
\
    
\f2\i \cf8 #difference of log(value+1) and log_encode_mean\
    
\f0\i0\b \cf2 def 
\f1\b0 \cf0 transformFunction(self, x):\
        
\f0\b \cf2 return 
\f1\b0 \cf0 tf.log(x + \cf7 1\cf0 ) - tf.expand_dims(self.log_x_encode_mean, \cf7 1\cf0 )\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 inverse_transformFunction(self, x):\
        
\f0\b \cf2 return 
\f1\b0 \cf0 tf.exp(x + tf.expand_dims(self.log_x_encode_mean, \cf7 1\cf0 )) - \cf7 1\
\
\

\f2\i \cf8 #placeholder is something that should always be fed with data when executed. When code is running at each step, data will be provided to placeholder.\
    #format : placeholder( dtype, shape = None,name=None)\
    #here we are inserting placeholder for tensor for which code has to feed data upon execution.\
    
\f0\i0\b \cf2 def 
\f1\b0 \cf0 getData(self):\
        self.x_encode = tf.placeholder(tf.float32, [\cf2 None\cf0 , \cf2 None\cf0 ])\
        self.encode_len = tf.placeholder(tf.int32, [\cf2 None\cf0 ])\
        self.y_decode = tf.placeholder(tf.float32, [\cf2 None\cf0 , self.decodeCount])\
        self.decode_len = tf.placeholder(tf.int32, [\cf2 None\cf0 ])\
        self.is_nan_encode = tf.placeholder(tf.float32, [\cf2 None\cf0 , \cf2 None\cf0 ])\
        self.is_nan_decode = tf.placeholder(tf.float32, [\cf2 None\cf0 , self.decodeCount])\
\
        self.page_id = tf.placeholder(tf.int32, [\cf2 None\cf0 ])\
        self.project = tf.placeholder(tf.int32, [\cf2 None\cf0 ])\
        self.access = tf.placeholder(tf.int32, [\cf2 None\cf0 ])\
        self.agent = tf.placeholder(tf.int32, [\cf2 None\cf0 ])\
\
        self.keepProbability = tf.placeholder(tf.float32)\
        
\f2\i \cf8 #to know if we are training on data.\
        
\f1\i0 \cf0 self.is_training = tf.placeholder(tf.bool)\
\
        
\f2\i \cf8 #log_x_encode_mean is log(mean of encode sequence)\
        
\f1\i0 \cf0 self.log_x_encode_mean = sequence_mean(tf.log(self.x_encode + \cf7 1\cf0 ), self.encode_len)\
\
        
\f2\i \cf8 #computing transformFunction on each encode input value\
        
\f1\i0 \cf0 self.log_x_encode = self.transformFunction(self.x_encode)\
\
\
        
\f2\i \cf8 #expanding dimension by 2\
        
\f1\i0 \cf0 self.x = tf.expand_dims(self.log_x_encode, \cf7 2\cf0 )\
\

\f2\i \cf8 #concat is to concatenate all these vectors around the axis. here axis is 2 which implies that this axis  is  2 dimensional\
        #expand_dims adds an additional dimension 1 to the shape -- returns the given input but with an extra dimension.\
        
\f1\i0 \cf0 self.encodeTensorFeatures = tf.concat([\
            tf.expand_dims(self.is_nan_encode, \cf7 2\cf0 ),\
            
\f2\i \cf8 #tf.cast to cast a tensor to required type.\
            
\f1\i0 \cf0 tf.expand_dims(tf.cast(tf.equal(self.x_encode, \cf7 0.0\cf0 ), tf.float32), \cf7 2\cf0 ),\
            
\f2\i \cf8 #-1 is to flatten.\
            
\f1\i0 \cf0 tf.tile(tf.reshape(self.log_x_encode_mean, (-\cf7 1\cf0 , \cf7 1\cf0 , \cf7 1\cf0 )), (\cf7 1\cf0 , tf.shape(self.x_encode)[\cf7 1\cf0 ], \cf7 1\cf0 )),\
            tf.tile(tf.expand_dims(tf.one_hot(self.project, \cf7 9\cf0 ), \cf7 1\cf0 ), (\cf7 1\cf0 , tf.shape(self.x_encode)[\cf7 1\cf0 ], \cf7 1\cf0 )),\
            tf.tile(tf.expand_dims(tf.one_hot(self.access, \cf7 3\cf0 ), \cf7 1\cf0 ), (\cf7 1\cf0 , tf.shape(self.x_encode)[\cf7 1\cf0 ], \cf7 1\cf0 )),\
            tf.tile(tf.expand_dims(tf.one_hot(self.agent, \cf7 2\cf0 ), \cf7 1\cf0 ), (\cf7 1\cf0 , tf.shape(self.x_encode)[\cf7 1\cf0 ], \cf7 1\cf0 )),\
        ], \cf6 axis\cf0 =\cf7 2\cf0 )\
\
        decode_idx = tf.tile(tf.expand_dims(tf.range(self.decodeCount), \cf7 0\cf0 ), (tf.shape(self.y_decode)[\cf7 0\cf0 ], \cf7 1\cf0 ))\
        self.decodeTensorFeatures = tf.concat([\
            tf.one_hot(decode_idx, self.decodeCount),\
            tf.tile(tf.reshape(self.log_x_encode_mean, (-\cf7 1\cf0 , \cf7 1\cf0 , \cf7 1\cf0 )), (\cf7 1\cf0 , self.decodeCount, \cf7 1\cf0 )),\
            tf.tile(tf.expand_dims(tf.one_hot(self.project, \cf7 9\cf0 ), \cf7 1\cf0 ), (\cf7 1\cf0 , self.decodeCount, \cf7 1\cf0 )),\
            tf.tile(tf.expand_dims(tf.one_hot(self.access, \cf7 3\cf0 ), \cf7 1\cf0 ), (\cf7 1\cf0 , self.decodeCount, \cf7 1\cf0 )),\
            tf.tile(tf.expand_dims(tf.one_hot(self.agent, \cf7 2\cf0 ), \cf7 1\cf0 ), (\cf7 1\cf0 , self.decodeCount, \cf7 1\cf0 )),\
        ], \cf6 axis\cf0 =\cf7 2\cf0 )\
\
        
\f0\b \cf2 return 
\f1\b0 \cf0 self.x\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 encode(self, x, features):\
        x = tf.concat([x, features], \cf6 axis\cf0 =\cf7 2\cf0 )\
\
        inputs = timeDistributedDenseLayer(\
            \cf6 inputs\cf0 =x,\
            \cf6 output_units\cf0 =self.num_of_residual_channels,\
            \cf6 activation\cf0 =tf.nn.tanh,\
            \cf6 scope\cf0 =
\f0\b \cf5 'x-proj-encode'\
        
\f1\b0 \cf0 )\
\
        skipChannels = []\
        convolutionInputs = [inputs]\
        
\f0\b \cf2 for 
\f1\b0 \cf0 i, (dilation, widthOfFilter) 
\f0\b \cf2 in 
\f1\b0 enumerate\cf0 (\cf2 zip\cf0 (self.dilationsCount, self.widthOfFilter)):\
            dilatedConvolution = convolutionLayer(\
                \cf6 inputs\cf0 =inputs,\
                \cf6 output_units\cf0 =\cf7 2\cf0 *self.num_of_residual_channels,\
                \cf6 convolution_width\cf0 =widthOfFilter,\
                \cf6 causal\cf0 =\cf2 True\cf0 ,\
                \cf6 dilation_rate\cf0 =[dilation],\
                \cf6 scope\cf0 =
\f0\b \cf5 'dilated-conv-encode-\{\}'
\f1\b0 \cf0 .format(i)\
            )\
            filterConvolution, gateConvolution = tf.split(dilatedConvolution, \cf7 2\cf0 , \cf6 axis\cf0 =\cf7 2\cf0 )\
            dilatedConvolution = tf.nn.tanh(filterConvolution)*tf.nn.sigmoid(gateConvolution)\
\
            outputs = timeDistributedDenseLayer(\
                \cf6 inputs\cf0 =dilatedConvolution,\
                \cf6 output_units\cf0 =self.num_of_skip_channels + self.num_of_residual_channels,\
                \cf6 scope\cf0 =
\f0\b \cf5 'dilated-conv-proj-encode-\{\}'
\f1\b0 \cf0 .format(i)\
            )\
            skips, residuals = tf.split(outputs, [self.num_of_skip_channels, self.num_of_residual_channels], \cf6 axis\cf0 =\cf7 2\cf0 )\
\
            inputs += residuals\
            convolutionInputs.append(inputs)\
            skipChannels.append(skips)\
\
        skipChannels = tf.nn.relu(tf.concat(skipChannels, \cf6 axis\cf0 =\cf7 2\cf0 ))\
        h = timeDistributedDenseLayer(\
            skipChannels, \cf7 128\cf0 ,\
            \cf6 scope\cf0 =
\f0\b \cf5 'dense-encode-1'
\f1\b0 \cf0 ,\
            \cf6 activation\cf0 =tf.nn.relu\
        )\
        y_hat = timeDistributedDenseLayer(h, \cf7 1\cf0 , \cf6 scope\cf0 =
\f0\b \cf5 'dense-encode-2'
\f1\b0 \cf0 )\
\
        
\f0\b \cf2 return 
\f1\b0 \cf0 y_hat, convolutionInputs[:-\cf7 1\cf0 ]\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 decodeParametersInitialize(self, x, features):\
        x = tf.concat([x, features], \cf6 axis\cf0 =\cf7 2\cf0 )\
\
        inputs = timeDistributedDenseLayer(\
            \cf6 inputs\cf0 =x,\
            \cf6 output_units\cf0 =self.num_of_residual_channels,\
            \cf6 activation\cf0 =tf.nn.tanh,\
            \cf6 scope\cf0 =
\f0\b \cf5 'x-proj-decode'\
        
\f1\b0 \cf0 )\
\
        skipChannels = []\
        convolutionInputs = [inputs]\
        
\f0\b \cf2 for 
\f1\b0 \cf0 i, (dilation, widthOfFilter) 
\f0\b \cf2 in 
\f1\b0 enumerate\cf0 (\cf2 zip\cf0 (self.dilationsCount, self.widthOfFilter)):\
            dilatedConvolution = convolutionLayer(\
                \cf6 inputs\cf0 =inputs,\
                \cf6 output_units\cf0 =\cf7 2\cf0 *self.num_of_residual_channels,\
                \cf6 convolution_width\cf0 =widthOfFilter,\
                \cf6 causal\cf0 =\cf2 True\cf0 ,\
                \cf6 dilation_rate\cf0 =[dilation],\
                \cf6 scope\cf0 =
\f0\b \cf5 'dilated-conv-decode-\{\}'
\f1\b0 \cf0 .format(i)\
            )\
            filterConvolution, gateConvolution = tf.split(dilatedConvolution, \cf7 2\cf0 , \cf6 axis\cf0 =\cf7 2\cf0 )\
            dilatedConvolution = tf.nn.tanh(filterConvolution)*tf.nn.sigmoid(gateConvolution)\
\
            outputs = timeDistributedDenseLayer(\
                \cf6 inputs\cf0 =dilatedConvolution,\
                \cf6 output_units\cf0 =self.num_of_skip_channels + self.num_of_residual_channels,\
                \cf6 scope\cf0 =
\f0\b \cf5 'dilated-conv-proj-decode-\{\}'
\f1\b0 \cf0 .format(i)\
            )\
            skips, residuals = tf.split(outputs, [self.num_of_skip_channels, self.num_of_residual_channels], \cf6 axis\cf0 =\cf7 2\cf0 )\
\
            inputs += residuals\
            convolutionInputs.append(inputs)\
            skipChannels.append(skips)\
\
        skipChannels = tf.nn.relu(tf.concat(skipChannels, \cf6 axis\cf0 =\cf7 2\cf0 ))\
        h = timeDistributedDenseLayer(skipChannels, \cf7 128\cf0 , \cf6 scope\cf0 =
\f0\b \cf5 'dense-decode-1'
\f1\b0 \cf0 , \cf6 activation\cf0 =tf.nn.relu)\
        y_hat = timeDistributedDenseLayer(h, \cf7 1\cf0 , \cf6 scope\cf0 =
\f0\b \cf5 'dense-decode-2'
\f1\b0 \cf0 )\
        
\f0\b \cf2 return 
\f1\b0 \cf0 y_hat\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 decode(self, x, convolutionInputs, features):\
        batch_size = tf.shape(x)[\cf7 0\cf0 ]\
\
        
\f2\i \cf8 # initialize state tensor arrays\
        
\f1\i0 \cf0 state_queues = []\
        
\f0\b \cf2 for 
\f1\b0 \cf0 i, (conv_Indexbatchbatch_idx = tf.range(batch_size)\
            batch_idx = tf.tile(tf.expand_dims(batch_idx, \cf7 1\cf0 ), (\cf7 1\cf0 , dilation))\
            batch_idx = tf.reshape(batch_idx, [-\cf7 1\cf0 ])\
\
            queue_begin_time = self.encode_len - dilation - \cf7 1\
            \cf0 Indextemp = tf.expand_dims(queue_begin_time, \cf7 1\cf0 ) + tf.expand_dims(tf.range(dilation), \cf7 0\cf0 )\
            Indextemp = tf.reshape(Indextemp, [-\cf7 1\cf0 ])\
\
\
            idx = tf.stack([batch_idx, Indextemp], \cf6 axis\cf0 =\cf7 1\cf0 )\
\
            
\f2\i \cf8 #collects all slices from conv_input within specified index into tensor of shape as indicated\
            
\f1\i0 \cf0 slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation, shape(conv_input, \cf7 2\cf0 )))\
\
            layer_ta = tf.TensorArray(\cf6 dtype\cf0 =tf.float32, \cf6 size\cf0 =dilation + self.decodeCount)\
            
\f2\i \cf8 #unpacks the tensor into individual tensors\
            
\f1\i0 \cf0 layer_ta = layer_ta.unstack(tf.transpose(slices, (\cf7 1\cf0 , \cf7 0\cf0 , \cf7 2\cf0 )))\
            state_queues.append(layer_ta)\
\
        
\f2\i \cf8 # initialize feature tensor array\
        
\f1\i0 \cf0 Tensorfeatures = tf.TensorArray(\cf6 dtype\cf0 =tf.float32, \cf6 size\cf0 =self.decodeCount)\
        Tensorfeatures = Tensorfeatures.unstack(tf.transpose(features, (\cf7 1\cf0 , \cf7 0\cf0 , \cf7 2\cf0 )))\
\
        
\f2\i \cf8 # initialize output tensor array\
        
\f1\i0 \cf0 FinalemittedArray = tf.TensorArray(\cf6 size\cf0 =self.decodeCount, \cf6 dtype\cf0 =tf.float32)\
\
        
\f2\i \cf8 # initialize other loop vars\
        
\f1\i0 \cf0 finishedElements = \cf7 0 \cf0 >= self.decode_len\
        time = tf.constant(\cf7 0\cf0 , \cf6 dtype\cf0 =tf.int32)\
\
        
\f2\i \cf8 # get initial x input\
        
\f1\i0 \cf0 current_idx = tf.stack([tf.range(tf.shape(self.encode_len)[\cf7 0\cf0 ]), self.encode_len - \cf7 1\cf0 ], \cf6 axis\cf0 =\cf7 1\cf0 )\
        initial_input = tf.gather_nd(x, current_idx)\
\
        
\f0\b \cf2 def 
\f1\b0 \cf0 loopfunction(time, current_input, queues):\
            current_features = Tensorfeatures.read(time)\
            current_input = tf.concat([current_input, current_features], \cf6 axis\cf0 =\cf7 1\cf0 )\
\
            
\f0\b \cf2 with 
\f1\b0 \cf0 tf.variable_scope(
\f0\b \cf5 'x-proj-decode'
\f1\b0 \cf0 , \cf6 reuse\cf0 =\cf2 True\cf0 ):\
                w_xProjection = tf.get_variable(
\f0\b \cf5 'weights'
\f1\b0 \cf0 )\
                b_xProjection = tf.get_variable(
\f0\b \cf5 'biases'
\f1\b0 \cf0 )\
                
\f2\i \cf8 #calcluating feature map at every level\
                # This is obtained by doing convolution on input image on sub regions with filter\
                # and adding bias and applying non linear filter function.\
                
\f1\i0 \cf0 xProjection = tf.nn.tanh(tf.matmul(current_input, w_xProjection) + b_xProjection)\
\
                skipChannels, updated_queues = [], []\
            
\f0\b \cf2 for 
\f1\b0 \cf0 i, (conv_input, queue, dilation) 
\f0\b \cf2 in 
\f1\b0 enumerate\cf0 (\cf2 zip\cf0 (convolutionInputs, queues, self.dilationsCount)):\
\
                state = queue.read(time)\
                
\f0\b \cf2 with 
\f1\b0 \cf0 tf.variable_scope(
\f0\b \cf5 'dilated-conv-decode-\{\}'
\f1\b0 \cf0 .format(i), \cf6 reuse\cf0 =\cf2 True\cf0 ):\
                    w_conv = tf.get_variable(
\f0\b \cf5 'weights'
\f1\b0 \cf0 .format(i))\
                    b_conv = tf.get_variable(
\f0\b \cf5 'biases'
\f1\b0 \cf0 .format(i))\
                    
\f2\i \cf8 #doing dilated convolution at every point\
                    
\f1\i0 \cf0 dilatedConvolution = tf.matmul(state, w_conv[\cf7 0\cf0 , :, :]) + tf.matmul(xProjection, w_conv[\cf7 1\cf0 , :, :]) + b_conv\
                filterConvolution, gateConvolution = tf.split(dilatedConvolution, \cf7 2\cf0 , \cf6 axis\cf0 =\cf7 1\cf0 )\
                
\f2\i \cf8 #applying tanh to get feature map\
                
\f1\i0 \cf0 dilatedConvolution = tf.nn.tanh(filterConvolution)*tf.nn.sigmoid(gateConvolution)\
\
                
\f0\b \cf2 with 
\f1\b0 \cf0 tf.variable_scope(
\f0\b \cf5 'dilated-conv-proj-decode-\{\}'
\f1\b0 \cf0 .format(i), \cf6 reuse\cf0 =\cf2 True\cf0 ):\
                    wProjection = tf.get_variable(
\f0\b \cf5 'weights'
\f1\b0 \cf0 .format(i))\
                    bProjection = tf.get_variable(
\f0\b \cf5 'biases'
\f1\b0 \cf0 .format(i))\
                    
\f2\i \cf8 #final convolution\
                    
\f1\i0 \cf0 concat_outputs = tf.matmul(dilatedConvolution, wProjection) + bProjection\
                skips, residuals = tf.split(concat_outputs, [self.num_of_skip_channels, self.num_of_residual_channels], \cf6 axis\cf0 =\cf7 1\cf0 )\
\
                xProjection += residuals\
                skipChannels.append(skips)\
                updated_queues.append(queue.write(time + dilation, xProjection))\
\
            skipChannels = tf.nn.relu(tf.concat(skipChannels, \cf6 axis\cf0 =\cf7 1\cf0 ))\
            
\f0\b \cf2 with 
\f1\b0 \cf0 tf.variable_scope(
\f0\b \cf5 'dense-decode-1'
\f1\b0 \cf0 , \cf6 reuse\cf0 =\cf2 True\cf0 ):\
                w_h = tf.get_variable(
\f0\b \cf5 'weights'
\f1\b0 \cf0 )\
                b_h = tf.get_variable(
\f0\b \cf5 'biases'
\f1\b0 \cf0 )\
                
\f2\i \cf8 #doing convolution on skip outputs\
                
\f1\i0 \cf0 h = tf.nn.relu(tf.matmul(skipChannels, w_h) + b_h)\
\
            
\f0\b \cf2 with 
\f1\b0 \cf0 tf.variable_scope(
\f0\b \cf5 'dense-decode-2'
\f1\b0 \cf0 , \cf6 reuse\cf0 =\cf2 True\cf0 ):\
                w_y = tf.get_variable(
\f0\b \cf5 'weights'
\f1\b0 \cf0 )\
                b_y = tf.get_variable(
\f0\b \cf5 'biases'
\f1\b0 \cf0 )\
                
\f2\i \cf8 #final convolution\
                
\f1\i0 \cf0 y_hat = tf.matmul(h, w_y) + b_y\
\
            finishedElements = (time >= self.decode_len)\
            finished = tf.reduce_all(finishedElements)\
\
            next_input = tf.cond(\
                finished,\
                
\f0\b \cf2 lambda
\f1\b0 \cf0 : tf.zeros([batch_size, \cf7 1\cf0 ], \cf6 dtype\cf0 =tf.float32),\
                
\f0\b \cf2 lambda
\f1\b0 \cf0 : y_hat\
            )\
            next_finishedElements = (time >= self.decode_len - \cf7 1\cf0 )\
\
            
\f0\b \cf2 return 
\f1\b0 \cf0 (next_finishedElements, next_input, updated_queues)\
\
        
\f0\b \cf2 def 
\f1\b0 \cf0 condition(\cf8 unused_time\cf0 , finishedElements, *_):\
            
\f0\b \cf2 return 
\f1\b0 \cf0 tf.logical_not(tf.reduce_all(finishedElements))\
\
        
\f0\b \cf2 def 
\f1\b0 \cf0 body(time, finishedElements, FinalemittedArray, *state_queues):\
            (next_finished, FinalemittedOutput, state_queues) = loopfunction(time, initial_input, state_queues)\
\
            emit = tf.where(finishedElements, tf.zeros_like(FinalemittedOutput), FinalemittedOutput)\
            FinalemittedArray = FinalemittedArray.write(time, emit)\
\
            finishedElements = tf.logical_or(finishedElements, next_finished)\
            
\f0\b \cf2 return 
\f1\b0 \cf0 [time + \cf7 1\cf0 , finishedElements, FinalemittedArray] + \cf2 list\cf0 (state_queues)\
\
        returned = tf.while_loop(\
            \cf6 cond\cf0 =condition,\
            \cf6 body\cf0 =body,\
            \cf6 loop_vars\cf0 =[time, finishedElements, FinalemittedArray] + state_queues\
        )\
\
        outputs_ta = returned[\cf7 2\cf0 ]\
        y_hat = tf.transpose(outputs_ta.stack(), (\cf7 1\cf0 , \cf7 0\cf0 , \cf7 2\cf0 ))\
        
\f0\b \cf2 return 
\f1\b0 \cf0 y_hat\
\
    
\f0\b \cf2 def 
\f1\b0 \cf0 calculate_loss(self):\
        x = self.getData()\
\
        y_hat_encode, convolutionInputs = self.encode(x, \cf6 features\cf0 =self.encodeTensorFeatures)\
        self.decodeParametersInitialize(x, \cf6 features\cf0 =self.decodeTensorFeatures)\
        y_decodedhat = self.decode(y_hat_encode, convolutionInputs, \cf6 features\cf0 =self.decodeTensorFeatures)\
        y_decodedhat = self.inverse_transformFunction(tf.squeeze(y_decodedhat, \cf7 2\cf0 ))\
        y_decodedhat = tf.nn.relu(y_decodedhat)\
\
        self.labels = self.y_decode\
        self.preds = y_decodedhat\
        self.loss = sequence_smape(self.labels, self.preds, self.decode_len, self.is_nan_decode)\
\
        self.prediction_tensors = \{\
            
\f0\b \cf5 'priors'
\f1\b0 \cf0 : self.x_encode,\
            
\f0\b \cf5 'labels'
\f1\b0 \cf0 : self.labels,\
            
\f0\b \cf5 'preds'
\f1\b0 \cf0 : self.preds,\
            
\f0\b \cf5 'page_id'
\f1\b0 \cf0 : self.page_id,\
        \}\
\
        
\f0\b \cf2 return 
\f1\b0 \cf0 self.loss\
\
\

\f0\b \cf2 if 
\f1\b0 \cf0 __name__ == 
\f0\b \cf5 '__main__'
\f1\b0 \cf0 :\
    base_dir = 
\f0\b \cf5 './'\
\
    
\f1\b0 \cf0 dr = DataReader(\cf6 data_dir\cf0 =os.path.join(base_dir, 
\f0\b \cf5 'data/processed/'
\f1\b0 \cf0 ))\
\
    nn = cnn(\
        \cf6 reader\cf0 =dr,\
        \cf6 log_dir\cf0 =os.path.join(base_dir, 
\f0\b \cf5 'logs'
\f1\b0 \cf0 ),\
        \cf6 checkpoint_dir\cf0 =os.path.join(base_dir, 
\f0\b \cf5 'checkpoints'
\f1\b0 \cf0 ),\
        \cf6 prediction_dir\cf0 =os.path.join(base_dir, 
\f0\b \cf5 'predictions'
\f1\b0 \cf0 ),\
        \cf6 optimizer\cf0 =
\f0\b \cf5 'adam'
\f1\b0 \cf0 ,\
        \cf6 learning_rate\cf0 =\cf7 .001\cf0 ,\
        \cf6 batch_size\cf0 =\cf7 128\cf0 ,\
        \cf6 num_training_steps\cf0 =\cf7 200000\cf0 ,\
        \cf6 early_stopping_steps\cf0 =\cf7 5000\cf0 ,\
        \cf6 warm_start_init_step\cf0 =\cf7 0\cf0 ,\
        \cf6 regularization_constant\cf0 =\cf7 0.0\cf0 ,\
        \cf6 keepProbability\cf0 =\cf7 1.0\cf0 ,\
        \cf6 enable_parameter_averaging\cf0 =\cf2 False\cf0 ,\
        \cf6 num_restarts\cf0 =\cf7 2\cf0 ,\
        \cf6 min_steps_to_checkpoint\cf0 =\cf7 500\cf0 ,\
        \cf6 log_interval\cf0 =\cf7 10\cf0 ,\
        \cf6 num_validation_batches\cf0 =\cf7 1\cf0 ,\
        \cf6 grad_clip\cf0 =\cf7 20\cf0 ,\
        \cf6 num_of_residual_channels\cf0 =\cf7 32\cf0 ,\
        \cf6 num_of_skip_channels\cf0 =\cf7 32\cf0 ,\
        \cf6 dilationsCount\cf0 =[\cf7 2\cf0 **i 
\f0\b \cf2 for 
\f1\b0 \cf0 i 
\f0\b \cf2 in 
\f1\b0 range\cf0 (\cf7 8\cf0 )]*\cf7 3\cf0 ,\
        \cf6 widthOfFilter\cf0 =[\cf7 2 
\f0\b \cf2 for 
\f1\b0 \cf0 i 
\f0\b \cf2 in 
\f1\b0 range\cf0 (\cf7 8\cf0 )]*\cf7 3\cf0 ,\
        \cf6 decodeCount\cf0 =\cf7 64\cf0 ,\
    )\
    nn.fit()\
    nn.restore()\
    nn.predict()\
\
}