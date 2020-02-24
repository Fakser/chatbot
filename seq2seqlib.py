from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from copy import deepcopy
from dataload import preprocess_sentence, load_convos, merge_convos, convos_to_questions_and_answers
from sklearn.model_selection import train_test_split
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
import numpy as np
import os
import time

def tokenize(lang):
    max_num_words = 25000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        num_words = max_num_words,
        oov_token='<out>')

    tokenizer.fit_on_texts(lang)

    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= tokenizer.num_words} # <= because tokenizer is 1 indexed
    tokenizer.word_index[tokenizer.oov_token] =  1

    tensor = tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, tokenizer


# Encoder - Decoder architecture was inspired by https://www.tensorflow.org/tutorials/text/nmt_with_attention
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))
    
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

# End of inspiration
class Seq2seq(object):
    def __init__(self):
        conversations = load_convos()
        print('files loaded')
        # splitting conversations to questions list and answers list, if there are more messages from one sender/reciever in a row they should be added as one string
        conversations_merged = merge_convos(conversations)
        print('messages merged')
        questions, answers = convos_to_questions_and_answers(conversations_merged, user_name= 'Krzysztof Kramarz')
        print('lists of questions and answers created')
        self.input_tensor, self.input_dict = tokenize(questions)
        self.target_tensor, self.target_dict = tokenize(answers)
        self.input_tensor_train, self.input_tensor_val, self.target_tensor_train, self.target_tensor_val = train_test_split(self.input_tensor, self.target_tensor, test_size=0.05)
        # print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
        # hyper parametrs 
        self.BUFFER_SIZE = len(self.input_tensor_train)
        self.BATCH_SIZE = 64
        self.steps_per_epoch = len(self.input_tensor_train)//self.BATCH_SIZE
        self.embedding_dim = 256
        self.units = 1024
        self.vocab_inp_size = len(self.input_dict.word_index)+1
        self.vocab_tar_size = len(self.target_dict.word_index)+1
        # cerating dataset
        self.dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_train, self.target_tensor_train)).shuffle(self.BUFFER_SIZE)
        self.dataset = self.dataset.batch(self.BATCH_SIZE, drop_remainder=True)
        # more parameters
        max_len = 20 #this is just stupid ignore it
        self.max_length_inp = max_len
        self.max_length_targ = max_len

        # encoder, attention, decoder
        self.encoder = Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)

        self.decoder = Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)

    
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')


        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                            encoder=self.encoder,
                                            decoder=self.decoder)

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)


    def train(self, EPOCHS):

        @tf.function
        def train_step(inp, targ, enc_hidden):
            loss = 0

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = self.encoder(inp, enc_hidden)

                dec_hidden = enc_hidden

                dec_input = tf.expand_dims([self.target_dict.word_index['<start>']] * self.BATCH_SIZE, 1)

                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                    loss += self.loss_function(targ[:, t], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)

            batch_loss = (loss / int(targ.shape[1]))

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables

            gradients = tape.gradient(loss, variables)

            self.optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f} time {:.2f} min'.format(epoch + 1,
                                                                                batch,
                                                                                batch_loss.numpy(),
                                                                                (time.time()-start)/60))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 5 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} min\n'.format((time.time() - start)/60))


    def load(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))



    def evaluate(self, sentence):
        attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))

        sentence = preprocess_sentence(sentence)

        inputs = [self.input_dict.word_index[i] if i in self.input_dict.word_index else self.input_dict.word_index['<out>'] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                                maxlen=self.max_length_inp,
                                                                padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.target_dict.word_index['<start>']], 0)

        for t in range(self.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                dec_hidden,
                                                                enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()


            if self.target_dict.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot


            if self.target_dict.index_word[predicted_id] != '<out>':
                result += self.target_dict.index_word[predicted_id] + ' '
            else:
                result += 'xd '

        
            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot  

    def chat(self, sentence):
        result, sentence, _ = self.evaluate(sentence)
        print('User: %s' % (sentence))
        print('Bot: {}'.format(result))
        return result
