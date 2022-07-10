import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification


class BertTrainer:
    """
    Builds and trains a BERT model
    """

    def __init__(self, df, classes, labels):
        self.df = df
        self.classes = classes 
        self.labels = labels


    def build_data(self):
        """
        load the data and tranform them into TensorSlices

        :return: <tensor slices> the 3 splitted tensor slicec tht correpsond to the train, validtion and test set
        """

        titles = pd.read_csv(self.df, index_col=0)

        # execute only to undersample the classes 
        train = pd.concat([titles[titles['isco3']==self.classes[0]].sample(1000),
                    titles[titles['isco3']==self.classes[1]].sample(1000),
                    titles[titles['isco3']==self.classes[2]].sample(1000),
                    titles[titles['isco3']==self.classes[3]].sample(1000),
                    titles[titles['isco3']==self.classes[4]].sample(1000),
                    titles[titles['isco3']==self.classes[5]].sample(1000)], ignore_index = True)
        
        labels = pd.DataFrame(data ={'class':self.classes, 'label':self.labels})

        # match each escoe code to a label
        train = train.merge(labels, left_on='isco3', right_on='class')

        X = list(train.vacancyTitle.values) # the texts --> X
        y = list(train.label.values) # the labels we want to predict --> Y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128) # convert input strings to BERT encodings
        test_encodings = tokenizer(X_test, truncation=True, padding=True,  max_length=128)
        val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=128)

        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings),y_train)).shuffle(1000).batch(16) # convert the encodings to Tensorflow objects
        val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_encodings),y_val)).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings),y_test)).batch(64)

        return train_dataset, val_dataset, test_dataset, y_test


    def build_model(self, train_dataset, val_dataset):
        """
        compiles and traines a bert model

        :param labels: <list> the labels of the supervised task
        :return: <h5> the trained BERT model
        """

        model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', 
                                                           num_labels=len(self.labels))
        callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, 
                            mode='min', baseline=None, 
                            restore_best_weights=True)]

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=optimizer, loss=loss)

        model.fit(train_dataset, 
                    epochs=10,
                    callbacks=callbacks, 
                    validation_data=val_dataset,
                    batch_size=128)

        return model


    def evaluate_model(self, model, test_dataset, y_test):
        """
        evaluates the model and prints the classification report

        :param model: <h5> the trained BERT model
        :param test_dataset: <tensor slice> the test dataset 
        :param y_test: <numpy> the initial test set 
        """

        logits = model.predict(test_dataset)
        y_preds = np.argmax(logits[0], axis=1)
        print(classification_report(y_test, y_preds))

if __name__ == '__main__':

    classes = [251, 243, 214, 522, 242,334]
    labels = [0, 1, 2, 3, 4, 5]

    bert = BertTrainer('data/titles_final.csv', classes, labels)

    train_dataset, val_dataset, test_dataset, y_test = bert.build_data()
    model = bert.build_model(labels, train_dataset, val_dataset)
    bert.evaluate_model(model, test_dataset, y_test)

